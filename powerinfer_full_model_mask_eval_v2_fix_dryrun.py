#!/usr/bin/env python3
# powerinfer_full_model_mask_eval_v2_fix_dryrun.py
#
# Full-model mask evaluation with metadata, safe-exp, dry-run and quick modes.
# Based on v2_fix with:
#  - output metadata fields (metric_chunk_size, seq_len, eval_batch_size, topk_list, num_random_batches)
#  - safe_exp for PPL
#  - --dry_run to validate profile/model/hooks with a tiny forward (includes oracle_original test)
#  - --quick / --global_only convenience mode
#
# Usage (normal):
# python powerinfer_full_model_mask_eval_v2_fix_dryrun.py --model_path ./models/TinyLlama-1.1B-Chat-v1.0 \
#   --profile_json activation_profile_with_indices.json --prompts_file prompts.txt --seq_len 512 \
#   --eval_batch_size 4 --hot_fracs "0.01,0.02,0.05" --mask_modes "global_by_freq,global_by_mass" \
#   --topk_list "1,5,10" --dtype float16 --device cuda --metric_chunk_size 256 \
#   --output_json full_model_mask_eval.json --output_csv full_model_mask_eval.csv
#
# Quick smoke test:
# python powerinfer_full_model_mask_eval_v2_fix_dryrun.py --model_path ./models/TinyLlama-1.1B-Chat-v1.0 \
#   --profile_json activation_profile_with_indices.json --quick
#
# Dry run (validate hooks and a tiny forward):
# python powerinfer_full_model_mask_eval_v2_fix_dryrun.py --model_path ... --profile_json ... --dry_run

import argparse
import json
import math
import csv
import random
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_list(s: str, cast=float):
    if s is None or s == "":
        return []
    return [cast(x.strip()) for x in s.split(",") if x.strip()]


def extract_profile_layers(profile: Dict) -> Tuple[Dict[int, Dict], Dict]:
    layers = {}
    metadata = {}
    for k, v in profile.items():
        if isinstance(k, str) and k.isdigit() and isinstance(v, dict):
            layers[int(k)] = v
        elif isinstance(k, int) and isinstance(v, dict):
            layers[int(k)] = v
        else:
            metadata[k] = v
    if not layers:
        raise RuntimeError("Profile JSON contains no numeric layer entries. Re-run profiler with --save_indices True.")
    for layer_idx, entry in layers.items():
        if "order_by_freq" not in entry or "order_by_mass" not in entry:
            raise RuntimeError(
                f"Profile JSON missing order_by_freq/order_by_mass for layer {layer_idx}. Re-run profiler with --save_indices True."
            )
    return layers, metadata


def find_down_proj_modules(model, config):
    hidden_size = getattr(config, "hidden_size", None)
    candidates = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        lname = name.lower()
        is_ffn_path = (".mlp." in lname) or (".ffn." in lname) or ("feed_forward" in lname)
        is_down_name = (
            lname.endswith(".down_proj")
            or lname.endswith(".fc2")
            or lname.endswith(".w2")
            or (lname.endswith(".wo") and is_ffn_path)
        )
        shape_ok = (
            hidden_size is None
            or (module.out_features == hidden_size and module.in_features > module.out_features)
        )
        if is_ffn_path and is_down_name and shape_ok:
            candidates.append((name, module))
    return candidates


def ensure_pad_token(tokenizer, model):
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id


def ceil_k(frac: float, hidden: int):
    return max(1, int(math.ceil(frac * hidden)))


def safe_exp(x: float):
    # avoid overflow in math.exp
    if x != x:  # NaN
        return float("nan")
    if x > 700:
        return float("inf")
    return math.exp(x)


def compute_nll_from_logits_chunked(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attn_mask: torch.Tensor,
    chunk_size: int = 256,
):
    B, T, V = logits.shape
    logits_2d = logits.reshape(B * T, V)
    labels_1d = labels.reshape(B * T)
    mask_1d = attn_mask.reshape(B * T)
    valid_idx = mask_1d.nonzero(as_tuple=False).squeeze(1)
    if valid_idx.numel() == 0:
        return 0.0, 0
    nll_sum = 0.0
    total = 0
    for start in range(0, valid_idx.numel(), chunk_size):
        idx = valid_idx[start:start + chunk_size]
        l = logits_2d[idx].float()
        y = labels_1d[idx]
        nll_sum += F.cross_entropy(l, y, reduction="sum").item()
        total += y.shape[0]
    return nll_sum, total


def compute_kl_mse_topk_chunked(orig_logits: torch.Tensor, mask_logits: torch.Tensor, topk_list: List[int],
                               attn_mask: torch.Tensor, chunk_size: int = 256):
    B, T, V = orig_logits.shape
    mask_flat = attn_mask.reshape(B * T)
    valid_idx = mask_flat.nonzero(as_tuple=False).squeeze(1)
    if valid_idx.numel() == 0:
        return {
            "kl_mean": None,
            "logit_mse_mean": None,
            "top1_agreement": None,
            "topk_overlap": {k: None for k in topk_list},
        }
    topk_list = [k for k in topk_list if 0 < k <= V]
    if not topk_list:
        return {
            "kl_mean": None,
            "logit_mse_mean": None,
            "top1_agreement": None,
            "topk_overlap": {},
        }
    max_k = min(max(topk_list), V)

    kl_sum = 0.0
    mse_sum = 0.0
    top1_sum = 0.0
    topk_sums = {k: 0.0 for k in topk_list}
    total = 0

    orig_2d = orig_logits.reshape(B * T, V)
    mask_2d = mask_logits.reshape(B * T, V)

    for start in range(0, valid_idx.numel(), chunk_size):
        idx = valid_idx[start:start + chunk_size]
        o = orig_2d[idx].float()
        m = mask_2d[idx].float()
        n = o.shape[0]
        logp = F.log_softmax(o, dim=-1)
        logq = F.log_softmax(m, dim=-1)
        p = logp.exp()
        kl_sum += (p * (logp - logq)).sum(dim=1).sum().item()
        mse_sum += ((o - m) ** 2).mean(dim=1).sum().item()
        top1_sum += (torch.argmax(o, dim=-1) == torch.argmax(m, dim=-1)).float().sum().item()
        top_o = torch.topk(o, k=max_k, dim=-1).indices
        top_m = torch.topk(m, k=max_k, dim=-1).indices
        for k in topk_list:
            oo = top_o[:, :k]
            mm = top_m[:, :k]
            eq = oo.unsqueeze(2) == mm.unsqueeze(1)
            match_any = eq.any(dim=2)
            overlap_counts = match_any.sum(dim=1).float()  # values in [0..k]
            topk_sums[k] += (overlap_counts / float(k)).sum().item()  # accumulate fraction
        total += n

    return {
        "kl_mean": kl_sum / total,
        "logit_mse_mean": mse_sum / total,
        "top1_agreement": top1_sum / total,
        "topk_overlap": {k: topk_sums[k] / total for k in topk_list},
    }


def make_global_hook(mask_vec: torch.Tensor):
    def hook(module, inp):
        x = inp[0]
        mv = mask_vec.to(device=x.device, dtype=x.dtype)
        return (x * mv, )
    return hook


def make_oracle_hook(mask_tensor: torch.Tensor):
    def hook(module, inp):
        x = inp[0]
        return (x * mask_tensor, )
    return hook


def make_oracle_dynamic_hook(k: int):
    def hook(module, inp):
        x = inp[0]
        _, idxs = torch.topk(x.abs(), k=k, dim=-1)
        mask = torch.zeros_like(x)
        mask.scatter_(-1, idxs, 1.0)
        return (x * mask, )
    return hook


def stack_reservoir(reservoir: List[torch.Tensor]) -> torch.Tensor:
    if not reservoir:
        return None
    rows = []
    for s in reservoir:
        t = s.cpu()
        if t.dim() == 1:
            rows.append(t)
        elif t.dim() == 2 and t.shape[0] == 1:
            rows.append(t.squeeze(0))
        else:
            rows.append(t.reshape(-1))
    return torch.stack(rows, dim=0).float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--profile_json", type=str, required=True)
    parser.add_argument("--prompts_file", type=str, required=False, default=None)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--num_random_batches", type=int, default=8)
    parser.add_argument("--hot_fracs", "--hot-fracs", dest="hot_fracs", type=str, default="0.01,0.02,0.05,0.1")
    parser.add_argument("--mask_modes", type=str, default="global_by_freq,global_by_mass,oracle_original_topk_abs,oracle_dynamic_topk_abs")
    parser.add_argument("--topk_list", type=str, default="1,5,10")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--metric_chunk_size", type=int, default=256)
    parser.add_argument("--dry_run", action="store_true", help="Validate profile/model/hooks with a tiny forward and exit")
    parser.add_argument("--quick", action="store_true", help="Quick convenience mode (small seq_len, batch_size, metric_chunk_size, limited masks)")
    parser.add_argument("--global_only", action="store_true", help="Shortcut: only run global_by_freq and global_by_mass")
    parser.add_argument("--output_json", type=str, default="full_model_mask_eval_v2_fix_dryrun.json")
    parser.add_argument("--output_csv", type=str, default="full_model_mask_eval_v2_fix_dryrun.csv")
    args = parser.parse_args()

    if args.metric_chunk_size <= 0:
        raise RuntimeError("--metric_chunk_size must be > 0.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Quick mode overrides (unless user explicitly set values)
    if args.quick:
        print("Quick mode: overriding some parameters for a fast smoke test.")
        args.seq_len = 128
        args.eval_batch_size = 1
        args.metric_chunk_size = min(args.metric_chunk_size, 64)
        args.hot_fracs = "0.05,0.1"
        args.mask_modes = "global_by_freq,global_by_mass,oracle_dynamic_topk_abs"
        args.topk_list = "1,5"

    if args.global_only:
        args.mask_modes = "global_by_freq,global_by_mass"

    hot_fracs = parse_list(args.hot_fracs, float)
    hot_fracs = [f for f in hot_fracs if 0.0 < f <= 1.0]
    if not hot_fracs and not args.dry_run:
        raise RuntimeError("No valid hot_fracs provided. Use values in (0,1].")

    mask_modes = [m.strip() for m in args.mask_modes.split(",") if m.strip()]
    allowed_modes = {
        "global_by_freq",
        "global_by_mass",
        "oracle_original_topk_abs",
        "oracle_dynamic_topk_abs",
    }
    bad_modes = [m for m in mask_modes if m not in allowed_modes]
    if bad_modes:
        raise RuntimeError(f"Unknown mask modes: {bad_modes}")

    topk_list = sorted(set(int(x) for x in parse_list(args.topk_list, int) if int(x) > 0))
    if not topk_list and not args.dry_run:
        raise RuntimeError("No valid topk values provided.")

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    model_dtype = dtype_map.get(args.dtype, torch.float32)
    device = torch.device(args.device)

    # Load profile JSON
    with open(args.profile_json, "r", encoding="utf-8") as f:
        profile_raw = json.load(f)

    profile_layers, profile_meta = extract_profile_layers(profile_raw)
    profile_layer_keys = sorted(profile_layers.keys())

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=model_dtype)
    ensure_pad_token(tokenizer, model)
    model.to(device)
    model.eval()

    # Robust vocab detection
    out_emb = model.get_output_embeddings()
    vocab_out = None
    if out_emb is not None and hasattr(out_emb, "weight"):
        try:
            vocab_out = out_emb.weight.shape[0]
        except Exception:
            vocab_out = None
    if vocab_out is not None and topk_list:
        topk_list = [k for k in topk_list if k <= vocab_out]
        if not topk_list and not args.dry_run:
            raise RuntimeError("No valid topk values remain after vocab-size filtering.")

    # Find down_proj modules
    down_modules = find_down_proj_modules(model, model.config)
    if not down_modules:
        raise RuntimeError("No FFN down_proj modules found in the model by heuristics.")

    max_profile_idx = max(profile_layer_keys)
    if max_profile_idx >= len(down_modules):
        raise RuntimeError("Profile references layer index >= number of down_proj modules in the model. Model mismatch.")

    # Build order maps and validate
    order_by_freq_map = {}
    order_by_mass_map = {}
    for layer_idx in profile_layer_keys:
        entry = profile_layers[layer_idx]
        order_by_freq = np.array(entry["order_by_freq"], dtype=np.int64)
        order_by_mass = np.array(entry["order_by_mass"], dtype=np.int64)
        model_name, model_module = down_modules[layer_idx]
        expected_hidden = model_module.in_features
        if order_by_freq.shape[0] != expected_hidden:
            raise RuntimeError(
                f"Layer {layer_idx} hidden mismatch: profile has {order_by_freq.shape[0]}, model module {model_name} has {expected_hidden}."
            )
        profile_name = entry.get("name")
        if profile_name is not None and profile_name != model_name:
            print(f"Warning: layer {layer_idx} name mismatch: profile={profile_name}, model={model_name}")
        order_by_freq_map[layer_idx] = order_by_freq
        order_by_mass_map[layer_idx] = order_by_mass

    # Prepare data
    if args.prompts_file:
        with open(args.prompts_file, "r", encoding="utf-8") as f:
            prompts = [l.strip() for l in f if l.strip()]
        if len(prompts) == 0:
            raise RuntimeError("Prompts file empty.")
        steps = math.ceil(len(prompts) / args.eval_batch_size)
    else:
        prompts = None
        steps = args.num_random_batches

    def make_batch(start_idx, batch_size):
        if prompts:
            batch_prompts = prompts[start_idx:start_idx + batch_size]
            enc = tokenizer(batch_prompts, return_tensors="pt", padding="max_length",
                            truncation=True, max_length=args.seq_len)
            input_ids = enc["input_ids"]
            attn_mask = enc["attention_mask"]
        else:
            input_ids = torch.randint(0, model.get_input_embeddings().num_embeddings, (batch_size, args.seq_len), dtype=torch.long)
            attn_mask = torch.ones_like(input_ids)
        return input_ids, attn_mask

    # If dry_run: validate hooks and do a tiny forward (including oracle_original)
    if args.dry_run:
        print("DRY RUN: validating profile/model/hooks with a tiny forward.")
        tiny_seq = min(8, args.seq_len)
        tiny_batch = 1
        input_ids = torch.randint(0, model.get_input_embeddings().num_embeddings, (tiny_batch, tiny_seq), dtype=torch.long).to(device)
        attn_mask = torch.ones_like(input_ids).to(device)

        # Test global hooks
        handles = []
        try:
            for layer_idx in profile_layer_keys:
                name, module = down_modules[layer_idx]
                hidden = module.in_features
                mask_vec = torch.ones(hidden, dtype=torch.float32, device=device)
                h = module.register_forward_pre_hook(make_global_hook(mask_vec))
                handles.append(h)
            with torch.no_grad():
                _ = model(input_ids, attention_mask=attn_mask, use_cache=False)
            print("Tiny forward succeeded with global hooks registered and removed.")
        except Exception as e:
            print("Dry-run failed during tiny forward with global hooks:", repr(e))
            raise
        finally:
            for h in handles:
                try:
                    h.remove()
                except Exception:
                    pass

        # Test oracle_dynamic hooks
        handles = []
        try:
            for layer_idx in profile_layer_keys:
                name, module = down_modules[layer_idx]
                hidden = module.in_features
                k = ceil_k(0.05, hidden)
                h = module.register_forward_pre_hook(make_oracle_dynamic_hook(k))
                handles.append(h)
            with torch.no_grad():
                _ = model(input_ids, attention_mask=attn_mask, use_cache=False)
            print("Tiny forward succeeded with oracle_dynamic hooks.")
        except Exception as e:
            print("Dry-run failed during tiny forward with oracle_dynamic hooks:", repr(e))
            raise
        finally:
            for h in handles:
                try:
                    h.remove()
                except Exception:
                    pass

        # Test oracle_original: capture + mask application
        captured_inputs = {}
        capture_handles = []
        try:
            def make_capture_hook(layer_idx):
                def hook(module, inp):
                    captured_inputs[layer_idx] = inp[0].detach()
                return hook
            for layer_idx in profile_layer_keys:
                name, module = down_modules[layer_idx]
                h = module.register_forward_pre_hook(make_capture_hook(layer_idx))
                capture_handles.append(h)
            with torch.no_grad():
                _ = model(input_ids, attention_mask=attn_mask, use_cache=False)
        finally:
            for h in capture_handles:
                try:
                    h.remove()
                except Exception:
                    pass

        # Build masks from captured inputs and test masked forward
        handles = []
        try:
            for layer_idx in profile_layer_keys:
                if layer_idx not in captured_inputs:
                    raise RuntimeError(f"Captured inputs missing for layer {layer_idx} during dry-run oracle_original test.")
                x = captured_inputs[layer_idx]
                hidden = x.shape[-1]
                k = ceil_k(0.05, hidden)
                _, idxs = torch.topk(x.abs(), k=k, dim=-1)
                mask_tensor = torch.zeros_like(x)
                mask_tensor.scatter_(-1, idxs, 1.0)
                name, module = down_modules[layer_idx]
                h = module.register_forward_pre_hook(make_oracle_hook(mask_tensor))
                handles.append(h)
            with torch.no_grad():
                _ = model(input_ids, attention_mask=attn_mask, use_cache=False)
            print("Tiny forward succeeded with oracle_original hooks.")
        except Exception as e:
            print("Dry-run failed during tiny forward with oracle_original hooks:", repr(e))
            raise
        finally:
            for h in handles:
                try:
                    h.remove()
                except Exception:
                    pass

        print("DRY RUN: all checks passed. Exiting (dry_run).")
        return

    # Accumulators
    results = {}
    for mode in mask_modes:
        results[mode] = {}
        for hf in hot_fracs:
            results[mode][hf] = {
                "nll_orig_sum": 0.0,
                "n_tokens": 0,
                "nll_mask_sum": 0.0,
                "kl_sum": 0.0,
                "logit_mse_sum": 0.0,
                "top1_agreement_sum": 0.0,
                "topk_agreement_sums": {k: 0.0 for k in topk_list},
                "count_batches": 0,
            }

    # Helper to build global masks (device float32 placeholders; hook will cast to x.dtype)
    def build_global_masks_for_hot_frac(hf):
        masks = {}
        ks = {}
        for layer_idx in profile_layer_keys:
            order_freq = order_by_freq_map[layer_idx]
            order_mass = order_by_mass_map[layer_idx]
            hidden = order_freq.shape[0]
            k = ceil_k(hf, hidden)
            ks[layer_idx] = k
            if "global_by_freq" in mask_modes:
                top_idx = order_freq[:k]
                mask_vec = torch.zeros(hidden, dtype=torch.float32, device=device)
                mask_vec[top_idx.tolist()] = 1.0
                masks.setdefault("global_by_freq", {})[layer_idx] = mask_vec
            if "global_by_mass" in mask_modes:
                top_idx_m = order_mass[:k]
                mask_vec_m = torch.zeros(hidden, dtype=torch.float32, device=device)
                mask_vec_m[top_idx_m.tolist()] = 1.0
                masks.setdefault("global_by_mass", {})[layer_idx] = mask_vec_m
        return masks, ks

    # Main loop
    try:
        for step in tqdm(range(steps), desc="Eval batches"):
            if prompts:
                input_ids, attn_mask = make_batch(step * args.eval_batch_size, args.eval_batch_size)
            else:
                input_ids, attn_mask = make_batch(0, args.eval_batch_size)
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)

            # Original forward
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attn_mask, use_cache=False)
                logits_orig = outputs.logits

            labels = input_ids
            logits_orig_shift = logits_orig[:, :-1, :].detach()
            labels_shift = labels[:, 1:].detach()
            attn_mask_shift = attn_mask[:, 1:].detach().bool()

            # Chunked NLL original
            nll_orig, n_tokens = compute_nll_from_logits_chunked(
                logits_orig_shift, labels_shift, attn_mask_shift, chunk_size=args.metric_chunk_size
            )
            for mode in mask_modes:
                for hf in hot_fracs:
                    results[mode][hf]["nll_orig_sum"] += nll_orig
                    results[mode][hf]["n_tokens"] += n_tokens

            # For oracle_original_topk_abs we need original activations; capture them safely
            captured_inputs = {}
            if "oracle_original_topk_abs" in mask_modes:
                capture_handles = []
                try:
                    def make_capture_hook(layer_idx):
                        def hook(module, inp):
                            x = inp[0]
                            captured_inputs[layer_idx] = x.detach()
                        return hook
                    for layer_idx in profile_layer_keys:
                        name, module = down_modules[layer_idx]
                        h = module.register_forward_pre_hook(make_capture_hook(layer_idx))
                        capture_handles.append(h)
                    with torch.no_grad():
                        _ = model(input_ids, attention_mask=attn_mask, use_cache=False)
                finally:
                    for h in capture_handles:
                        try:
                            h.remove()
                        except Exception:
                            pass

            # Evaluate each mode/hf
            for mode in mask_modes:
                for hf in hot_fracs:
                    if mode in ("global_by_freq", "global_by_mass"):
                        masks_for_mode, ks = build_global_masks_for_hot_frac(hf)
                        per_layer_masks = masks_for_mode.get(mode, {})
                        handles = []
                        try:
                            for layer_idx, mask_vec in per_layer_masks.items():
                                name, module = down_modules[layer_idx]
                                h = module.register_forward_pre_hook(make_global_hook(mask_vec))
                                handles.append(h)
                            with torch.no_grad():
                                outputs_mask = model(input_ids, attention_mask=attn_mask, use_cache=False)
                                logits_mask = outputs_mask.logits
                        finally:
                            for h in handles:
                                try:
                                    h.remove()
                                except Exception:
                                    pass

                        logits_mask_shift = logits_mask[:, :-1, :].detach()
                        # Chunked NLL masked
                        nll_mask, _ = compute_nll_from_logits_chunked(
                            logits_mask_shift, labels_shift, attn_mask_shift, chunk_size=args.metric_chunk_size
                        )
                        metrics = compute_kl_mse_topk_chunked(
                            logits_orig_shift, logits_mask_shift, topk_list, attn_mask_shift, chunk_size=args.metric_chunk_size
                        )

                        res = results[mode][hf]
                        res["nll_mask_sum"] += nll_mask
                        res["kl_sum"] += (metrics["kl_mean"] * n_tokens) if metrics["kl_mean"] is not None else 0.0
                        res["logit_mse_sum"] += (metrics["logit_mse_mean"] * n_tokens) if metrics["logit_mse_mean"] is not None else 0.0
                        res["top1_agreement_sum"] += (metrics["top1_agreement"] * n_tokens) if metrics["top1_agreement"] is not None else 0.0
                        for k in topk_list:
                            val = metrics["topk_overlap"].get(k, None)
                            if val is not None:
                                res["topk_agreement_sums"][k] += (val * n_tokens)
                        res["count_batches"] += 1

                    elif mode == "oracle_original_topk_abs":
                        per_layer_mask_tensors = {}
                        for layer_idx in profile_layer_keys:
                            if layer_idx not in captured_inputs:
                                raise RuntimeError(f"Captured inputs missing for layer {layer_idx} required by oracle_original_topk_abs.")
                            x = captured_inputs[layer_idx]
                            hidden = x.shape[-1]
                            k = ceil_k(hf, hidden)
                            vals, idxs = torch.topk(x.abs(), k=k, dim=-1)
                            mask_tensor = torch.zeros_like(x, dtype=x.dtype, device=x.device)
                            mask_tensor.scatter_(-1, idxs, 1.0)
                            per_layer_mask_tensors[layer_idx] = mask_tensor

                        handles = []
                        try:
                            for layer_idx, mask_tensor in per_layer_mask_tensors.items():
                                name, module = down_modules[layer_idx]
                                h = module.register_forward_pre_hook(make_oracle_hook(mask_tensor))
                                handles.append(h)
                            with torch.no_grad():
                                outputs_mask = model(input_ids, attention_mask=attn_mask, use_cache=False)
                                logits_mask = outputs_mask.logits
                        finally:
                            for h in handles:
                                try:
                                    h.remove()
                                except Exception:
                                    pass

                        logits_mask_shift = logits_mask[:, :-1, :].detach()
                        nll_mask, _ = compute_nll_from_logits_chunked(
                            logits_mask_shift, labels_shift, attn_mask_shift, chunk_size=args.metric_chunk_size
                        )
                        metrics = compute_kl_mse_topk_chunked(
                            logits_orig_shift, logits_mask_shift, topk_list, attn_mask_shift, chunk_size=args.metric_chunk_size
                        )

                        res = results[mode][hf]
                        res["nll_mask_sum"] += nll_mask
                        res["kl_sum"] += (metrics["kl_mean"] * n_tokens) if metrics["kl_mean"] is not None else 0.0
                        res["logit_mse_sum"] += (metrics["logit_mse_mean"] * n_tokens) if metrics["logit_mse_mean"] is not None else 0.0
                        res["top1_agreement_sum"] += (metrics["top1_agreement"] * n_tokens) if metrics["top1_agreement"] is not None else 0.0
                        for k in topk_list:
                            val = metrics["topk_overlap"].get(k, None)
                            if val is not None:
                                res["topk_agreement_sums"][k] += (val * n_tokens)
                        res["count_batches"] += 1

                    elif mode == "oracle_dynamic_topk_abs":
                        ks = {}
                        for layer_idx in profile_layer_keys:
                            hidden = order_by_freq_map[layer_idx].shape[0]
                            ks[layer_idx] = ceil_k(hf, hidden)
                        handles = []
                        try:
                            for layer_idx, k in ks.items():
                                name, module = down_modules[layer_idx]
                                h = module.register_forward_pre_hook(make_oracle_dynamic_hook(k))
                                handles.append(h)
                            with torch.no_grad():
                                outputs_mask = model(input_ids, attention_mask=attn_mask, use_cache=False)
                                logits_mask = outputs_mask.logits
                        finally:
                            for h in handles:
                                try:
                                    h.remove()
                                except Exception:
                                    pass

                        logits_mask_shift = logits_mask[:, :-1, :].detach()
                        nll_mask, _ = compute_nll_from_logits_chunked(
                            logits_mask_shift, labels_shift, attn_mask_shift, chunk_size=args.metric_chunk_size
                        )
                        metrics = compute_kl_mse_topk_chunked(
                            logits_orig_shift, logits_mask_shift, topk_list, attn_mask_shift, chunk_size=args.metric_chunk_size
                        )

                        res = results[mode][hf]
                        res["nll_mask_sum"] += nll_mask
                        res["kl_sum"] += (metrics["kl_mean"] * n_tokens) if metrics["kl_mean"] is not None else 0.0
                        res["logit_mse_sum"] += (metrics["logit_mse_mean"] * n_tokens) if metrics["logit_mse_mean"] is not None else 0.0
                        res["top1_agreement_sum"] += (metrics["top1_agreement"] * n_tokens) if metrics["top1_agreement"] is not None else 0.0
                        for k in topk_list:
                            val = metrics["topk_overlap"].get(k, None)
                            if val is not None:
                                res["topk_agreement_sums"][k] += (val * n_tokens)
                        res["count_batches"] += 1

                    else:
                        raise RuntimeError(f"Unknown mask mode: {mode}")

    finally:
        pass

    # Aggregate results and include metadata
    out = {
        "profile_json": args.profile_json,
        "profile_metadata": profile_meta,
        "model_path": args.model_path,
        "dtype": args.dtype,
        "mask_modes": mask_modes,
        "hot_fracs": hot_fracs,
        "topk_list": topk_list,
        "seq_len": args.seq_len,
        "eval_batch_size": args.eval_batch_size,
        "num_random_batches": args.num_random_batches if prompts is None else None,
        "metric_chunk_size": args.metric_chunk_size,
        "profile_version": profile_meta.get("profile_version"),
    }
    rows = []
    for mode in mask_modes:
        out[mode] = {}
        for hf in hot_fracs:
            res = results[mode][hf]
            n_tokens = res["n_tokens"]
            if n_tokens == 0:
                print(f"Warning: no tokens evaluated for mode {mode} hf {hf}")
                continue
            nll_orig = res["nll_orig_sum"]
            nll_mask = res["nll_mask_sum"]
            ppl_orig = safe_exp(nll_orig / n_tokens) if n_tokens > 0 else None
            ppl_mask = safe_exp(nll_mask / n_tokens) if n_tokens > 0 else None
            ppl_delta = (ppl_mask - ppl_orig) if (ppl_orig is not None and ppl_mask is not None) else None
            kl_mean = res["kl_sum"] / n_tokens if n_tokens > 0 else None
            logit_mse_mean = res["logit_mse_sum"] / n_tokens if n_tokens > 0 else None
            top1_agreement = res["top1_agreement_sum"] / n_tokens if n_tokens > 0 else None
            topk_agreements = {k: (res["topk_agreement_sums"][k] / n_tokens if n_tokens > 0 else None) for k in topk_list}

            out[mode][hf] = {
                "ppl_original": ppl_orig,
                "ppl_masked": ppl_mask,
                "ppl_delta": ppl_delta,
                "kl_mean": kl_mean,
                "logit_mse_mean": logit_mse_mean,
                "top1_agreement": top1_agreement,
                "topk_overlap": topk_agreements,
                "n_tokens": n_tokens,
                "count_batches": res["count_batches"],
            }

            row = {
                "mask_mode": mode,
                "hot_frac": hf,
                "ppl_original": ppl_orig,
                "ppl_masked": ppl_mask,
                "ppl_delta": ppl_delta,
                "kl_mean": kl_mean,
                "logit_mse_mean": logit_mse_mean,
                "top1_agreement": top1_agreement,
                "n_tokens": n_tokens,
                "count_batches": res["count_batches"],
            }
            for k in topk_list:
                row[f"top{k}_overlap"] = topk_agreements[k]
            rows.append(row)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    fieldnames = ["mask_mode", "hot_frac", "ppl_original", "ppl_masked", "ppl_delta", "kl_mean", "logit_mse_mean", "top1_agreement", "n_tokens", "count_batches"] + [f"top{k}_overlap" for k in topk_list]
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print("Done. Results written to:", args.output_json, args.output_csv)


if __name__ == "__main__":
    main()
