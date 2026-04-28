#!/usr/bin/env python3
# powerinfer_full_model_mask_eval.py
#
# Full-model mask evaluation for PowerInfer-style Hot/Cold analysis.
#
# Modes:
# - global_by_freq
# - global_by_mass
# - oracle_per_token_topk_abs
#
# Metrics (per mask mode & hot_frac):
# - perplexity_original
# - perplexity_masked
# - perplexity_delta
# - kl_mean (original || masked)
# - logit_mse_mean
# - top1_agreement
# - topk_agreements (for each topk in --topk_list)
#
# Usage example:
# python powerinfer_full_model_mask_eval.py \
#   --model_path ./models/TinyLlama-1.1B-Chat-v1.0 \
#   --profile_json activation_profile_with_indices.json \
#   --prompts_file prompts.txt \
#   --seq_len 512 --eval_batch_size 4 \
#   --hot_fracs "0.01,0.02,0.05,0.1" \
#   --mask_modes "global_by_freq,global_by_mass,oracle_per_token_topk_abs" \
#   --topk_list "1,5,10" --dtype float16 --device "cuda" \
#   --seed 1234 --output_json full_model_mask_eval.json --output_csv full_model_mask_eval.csv
#
# Notes:
# - The profile JSON must contain order_by_freq and order_by_mass arrays per layer index.
# - The profile JSON should include metadata: selected_layers, model_path, dtype, profile_version (recommended).
# - This script runs original forward and masked forward(s) per batch; it's intentionally conservative and robust.
# - For oracle_per_token_topk_abs the per-token top-k is computed from the down_proj input of the original forward pass.
#
# Requirements: torch, transformers, tqdm, numpy

import argparse
import csv
import json
import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_list(s: str, cast=float):
    if s is None or s == "":
        return []
    return [cast(x.strip()) for x in s.split(",") if x.strip()]


def find_down_proj_modules(model, config):
    """
    Strict FFN down_proj detection (LLaMA/TinyLlama-typical).
    Returns list of (name, module) in model.named_modules() order.
    """
    hidden_size = getattr(config, "hidden_size", None)
    candidates = []

    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue

        lname = name.lower()
        is_ffn_path = (
            (".mlp." in lname)
            or (".ffn." in lname)
            or ("feed_forward" in lname)
        )

        is_down_name = (
            lname.endswith(".down_proj")
            or lname.endswith(".fc2")
            or lname.endswith(".w2")
            or (lname.endswith(".wo") and is_ffn_path)
        )

        shape_ok = (
            hidden_size is None
            or (
                module.out_features == hidden_size
                and module.in_features > module.out_features
            )
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


def compute_perplexity_from_logits(logits, labels, attn_mask):
    """
    logits: (B, T, V) float32
    labels: (B, T) long
    attn_mask: (B, T) bool (valid positions)

    Returns: (nll_sum, token_count)
    """
    B, T, V = logits.shape
    logits_flat = logits.reshape(B * T, V)
    labels_flat = labels.reshape(B * T)
    mask_flat = attn_mask.reshape(B * T)

    valid_idx = mask_flat.nonzero(as_tuple=False).squeeze(1)
    if valid_idx.numel() == 0:
        return 0.0, 0

    logits_sel = logits_flat[valid_idx]
    labels_sel = labels_flat[valid_idx]

    log_probs = F.log_softmax(logits_sel, dim=-1)
    nll = -log_probs[
        torch.arange(labels_sel.shape[0], device=labels_sel.device),
        labels_sel,
    ].sum().item()

    return nll, int(labels_sel.shape[0])


def compute_kl_and_logit_mse_and_topk(orig_logits, mask_logits, topk_list, attn_mask):
    """
    Compute per-token KL(original || masked), logit MSE, top1/topk agreements.

    orig_logits, mask_logits: (B, T, V) float32
    attn_mask: (B, T) bool

    Returns aggregated metrics (means).
    """
    B, T, V = orig_logits.shape
    mask_flat = attn_mask.reshape(B * T)
    valid_idx = mask_flat.nonzero(as_tuple=False).squeeze(1)

    if valid_idx.numel() == 0:
        return {
            "kl_mean": None,
            "logit_mse_mean": None,
            "top1_agreement": None,
            "topk_agreements": {k: None for k in topk_list},
        }

    orig_flat = orig_logits.reshape(B * T, V)[valid_idx].float()
    mask_flat_logits = mask_logits.reshape(B * T, V)[valid_idx].float()

    logp = F.log_softmax(orig_flat, dim=-1)
    logq = F.log_softmax(mask_flat_logits, dim=-1)
    p = logp.exp()

    kl_per_token = (p * (logp - logq)).sum(dim=1)
    kl_mean = float(kl_per_token.mean().item())

    mse_per_token = ((orig_flat - mask_flat_logits) ** 2).mean(dim=1)
    logit_mse_mean = float(mse_per_token.mean().item())

    top1_orig = torch.argmax(orig_flat, dim=-1)
    top1_mask = torch.argmax(mask_flat_logits, dim=-1)
    top1_agreement = float((top1_orig == top1_mask).float().mean().item())

    topk_agreements = {}
    max_k = max(topk_list) if topk_list else 1
    max_k = min(max_k, V)

    topk_orig = torch.topk(orig_flat, k=max_k, dim=-1).indices
    topk_mask = torch.topk(mask_flat_logits, k=max_k, dim=-1).indices

    topk_orig_np = topk_orig.cpu().numpy()
    topk_mask_np = topk_mask.cpu().numpy()
    N = topk_orig_np.shape[0]

    for k in topk_list:
        if k <= 0:
            topk_agreements[k] = None
            continue

        kk = min(k, max_k)
        overlap = 0.0

        for i in range(N):
            set_o = set(topk_orig_np[i, :kk].tolist())
            set_m = set(topk_mask_np[i, :kk].tolist())
            overlap += len(set_o.intersection(set_m)) / float(kk)

        topk_agreements[k] = float(overlap / N)

    return {
        "kl_mean": kl_mean,
        "logit_mse_mean": logit_mse_mean,
        "top1_agreement": top1_agreement,
        "topk_agreements": topk_agreements,
    }


def make_global_hook(mask_vec):
    def hook(module, inp):
        x = inp[0]
        return (x * mask_vec,)

    return hook


def make_oracle_hook(mask_tensor):
    def hook(module, inp):
        x = inp[0]
        return (x * mask_tensor,)

    return hook


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--profile_json", type=str, required=True)
    parser.add_argument("--prompts_file", type=str, required=False, default=None)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=4)

    parser.add_argument(
        "--hot_fracs",
        "--hot-fracs",
        dest="hot_fracs",
        type=str,
        default="0.01,0.02,0.05,0.1",
    )

    parser.add_argument(
        "--mask_modes",
        type=str,
        default="global_by_freq,global_by_mass,oracle_per_token_topk_abs",
    )

    parser.add_argument("--topk_list", type=str, default="1,5,10")

    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "float16", "bfloat16"],
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output_json", type=str, default="full_model_mask_eval.json")
    parser.add_argument("--output_csv", type=str, default="full_model_mask_eval.csv")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    hot_fracs = parse_list(args.hot_fracs, float)
    hot_fracs = [f for f in hot_fracs if 0.0 < f <= 1.0]

    if not hot_fracs:
        raise RuntimeError("No valid hot_fracs provided. Use values in (0,1].")

    mask_modes = [m.strip() for m in args.mask_modes.split(",") if m.strip()]
    topk_list = [int(x) for x in args.topk_list.split(",") if x.strip()]

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    model_dtype = dtype_map.get(args.dtype, torch.float32)
    device = torch.device(args.device)

    with open(args.profile_json, "r", encoding="utf-8") as f:
        profile = json.load(f)

    profile_layer_keys = sorted([int(k) for k in profile.keys() if str(k).isdigit()])

    if not profile_layer_keys:
        raise RuntimeError("Profile JSON contains no layer entries.")

    first_entry = profile[str(profile_layer_keys[0])]
    if "order_by_freq" not in first_entry or "order_by_mass" not in first_entry:
        raise RuntimeError(
            "Profile JSON has no saved indices. Re-run profiler with --save_indices True."
        )

    profile_selected_layers = profile.get("selected_layers", None)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=model_dtype,
    )

    ensure_pad_token(tokenizer, model)
    model.to(device)
    model.eval()

    down_modules = find_down_proj_modules(model, model.config)

    if not down_modules:
        raise RuntimeError("No FFN down_proj modules found in the model by heuristics.")

    if profile_selected_layers is not None and profile_selected_layers != profile_layer_keys:
        print("Warning: profile 'selected_layers' metadata differs from profile keys.")

    max_profile_idx = max(profile_layer_keys)
    if max_profile_idx >= len(down_modules):
        raise RuntimeError(
            "Profile references layer index >= number of down_proj modules in the model. Model mismatch."
        )

    order_by_freq_map = {}
    order_by_mass_map = {}

    for k in profile_layer_keys:
        entry = profile[str(k)] if str(k) in profile else profile[k]

        if "order_by_freq" not in entry or "order_by_mass" not in entry:
            raise RuntimeError(
                f"Profile JSON missing order_by_freq/order_by_mass for layer {k}"
            )

        order_by_freq_map[k] = np.array(entry["order_by_freq"], dtype=np.int64)
        order_by_mass_map[k] = np.array(entry["order_by_mass"], dtype=np.int64)

    if args.prompts_file:
        with open(args.prompts_file, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]

        if len(prompts) == 0:
            raise RuntimeError("Prompts file empty.")
    else:
        prompts = None

    eval_batch_size = args.eval_batch_size
    seq_len = args.seq_len

    def make_batch(start_idx, batch_size):
        if prompts:
            batch_prompts = prompts[start_idx:start_idx + batch_size]
            enc = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=seq_len,
            )
            input_ids = enc["input_ids"]
            attn_mask = enc["attention_mask"]
        else:
            input_ids = torch.randint(
                0,
                model.get_input_embeddings().num_embeddings,
                (batch_size, seq_len),
                dtype=torch.long,
            )
            attn_mask = torch.ones_like(input_ids)

        return input_ids, attn_mask

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

    if prompts:
        total_items = len(prompts)
        steps = math.ceil(total_items / eval_batch_size)
    else:
        steps = 1

    for step in tqdm(range(steps), desc="Eval batches"):
        if prompts:
            input_ids, attn_mask = make_batch(step * eval_batch_size, eval_batch_size)
        else:
            input_ids, attn_mask = make_batch(0, eval_batch_size)

        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attn_mask, use_cache=False)
            logits_orig = outputs.logits

        labels = input_ids
        logits_orig_shift = logits_orig[:, :-1, :].detach()
        labels_shift = labels[:, 1:].detach()
        attn_mask_shift = attn_mask[:, 1:].detach().bool()

        nll_orig, n_tokens = compute_perplexity_from_logits(
            logits_orig_shift.float(),
            labels_shift,
            attn_mask_shift,
        )

        for mode in mask_modes:
            for hf in hot_fracs:
                results[mode][hf]["nll_orig_sum"] += nll_orig
                results[mode][hf]["n_tokens"] += n_tokens

        captured_inputs = {}

        if "oracle_per_token_topk_abs" in mask_modes:
            capture_handles = []

            def make_capture_hook(layer_idx):
                def hook(module, inp):
                    x = inp[0]
                    captured_inputs[layer_idx] = x.detach()

                return hook

            for layer_idx in profile_layer_keys:
                name, module = down_modules[layer_idx]
                h = module.register_forward_pre_hook(make_capture_hook(layer_idx))
                capture_handles.append(h)

            try:
                with torch.no_grad():
                    _ = model(input_ids, attention_mask=attn_mask, use_cache=False)
            finally:
                for h in capture_handles:
                    try:
                        h.remove()
                    except Exception:
                        pass

        for mode in mask_modes:
            for hf in hot_fracs:
                if mode in ("global_by_freq", "global_by_mass"):
                    masks_for_mode, _ = build_global_masks_for_hot_frac(hf)
                    per_layer_masks = masks_for_mode.get(mode, {})
                    handles = []

                    for layer_idx, mask_vec in per_layer_masks.items():
                        name, module = down_modules[layer_idx]
                        h = module.register_forward_pre_hook(make_global_hook(mask_vec))
                        handles.append(h)

                    try:
                        with torch.no_grad():
                            outputs_mask = model(
                                input_ids,
                                attention_mask=attn_mask,
                                use_cache=False,
                            )
                            logits_mask = outputs_mask.logits
                    finally:
                        for h in handles:
                            try:
                                h.remove()
                            except Exception:
                                pass

                    logits_mask_shift = logits_mask[:, :-1, :].detach()
                    nll_mask, _ = compute_perplexity_from_logits(
                        logits_mask_shift.float(),
                        labels_shift,
                        attn_mask_shift,
                    )

                    metrics = compute_kl_and_logit_mse_and_topk(
                        logits_orig_shift.float(),
                        logits_mask_shift.float(),
                        topk_list,
                        attn_mask_shift,
                    )

                    res = results[mode][hf]
                    res["nll_mask_sum"] += nll_mask
                    res["kl_sum"] += (
                        metrics["kl_mean"] * n_tokens
                        if metrics["kl_mean"] is not None
                        else 0.0
                    )
                    res["logit_mse_sum"] += (
                        metrics["logit_mse_mean"] * n_tokens
                        if metrics["logit_mse_mean"] is not None
                        else 0.0
                    )
                    res["top1_agreement_sum"] += (
                        metrics["top1_agreement"] * n_tokens
                        if metrics["top1_agreement"] is not None
                        else 0.0
                    )

                    for k in topk_list:
                        val = metrics["topk_agreements"].get(k, None)
                        if val is not None:
                            res["topk_agreement_sums"][k] += val * n_tokens

                    res["count_batches"] += 1

                elif mode == "oracle_per_token_topk_abs":
                    per_layer_mask_tensors = {}

                    for layer_idx in profile_layer_keys:
                        if layer_idx not in captured_inputs:
                            raise RuntimeError(
                                f"Captured inputs missing for layer {layer_idx} required by oracle mode."
                            )

                        x = captured_inputs[layer_idx]
                        hidden = x.shape[-1]
                        k = ceil_k(hf, hidden)
                        _, idxs = torch.topk(x.abs(), k=k, dim=-1)
                        mask_tensor = torch.zeros_like(x, dtype=x.dtype, device=x.device)
                        mask_tensor.scatter_(-1, idxs, 1.0)
                        per_layer_mask_tensors[layer_idx] = mask_tensor

                    handles = []

                    for layer_idx, mask_tensor in per_layer_mask_tensors.items():
                        name, module = down_modules[layer_idx]
                        h = module.register_forward_pre_hook(make_oracle_hook(mask_tensor))
                        handles.append(h)

                    try:
                        with torch.no_grad():
                            outputs_mask = model(
                                input_ids,
                                attention_mask=attn_mask,
                                use_cache=False,
                            )
                            logits_mask = outputs_mask.logits
                    finally:
                        for h in handles:
                            try:
                                h.remove()
                            except Exception:
                                pass

                    logits_mask_shift = logits_mask[:, :-1, :].detach()
                    nll_mask, _ = compute_perplexity_from_logits(
                        logits_mask_shift.float(),
                        labels_shift,
                        attn_mask_shift,
                    )

                    metrics = compute_kl_and_logit_mse_and_topk(
                        logits_orig_shift.float(),
                        logits_mask_shift.float(),
                        topk_list,
                        attn_mask_shift,
                    )

                    res = results[mode][hf]
                    res["nll_mask_sum"] += nll_mask
                    res["kl_sum"] += (
                        metrics["kl_mean"] * n_tokens
                        if metrics["kl_mean"] is not None
                        else 0.0
                    )
                    res["logit_mse_sum"] += (
                        metrics["logit_mse_mean"] * n_tokens
                        if metrics["logit_mse_mean"] is not None
                        else 0.0
                    )
                    res["top1_agreement_sum"] += (
                        metrics["top1_agreement"] * n_tokens
                        if metrics["top1_agreement"] is not None
                        else 0.0
                    )

                    for k in topk_list:
                        val = metrics["topk_agreements"].get(k, None)
                        if val is not None:
                            res["topk_agreement_sums"][k] += val * n_tokens

                    res["count_batches"] += 1

                else:
                    raise RuntimeError(f"Unknown mask mode: {mode}")

    out = {
        "profile_json": args.profile_json,
        "model_path": args.model_path,
        "dtype": args.dtype,
        "mask_modes": mask_modes,
        "hot_fracs": hot_fracs,
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

            ppl_orig = math.exp(nll_orig / n_tokens) if n_tokens > 0 else None
            ppl_mask = math.exp(nll_mask / n_tokens) if n_tokens > 0 else None
            ppl_delta = (
                ppl_mask - ppl_orig
                if ppl_orig is not None and ppl_mask is not None
                else None
            )

            kl_mean = res["kl_sum"] / n_tokens if n_tokens > 0 else None
            logit_mse_mean = res["logit_mse_sum"] / n_tokens if n_tokens > 0 else None
            top1_agreement = res["top1_agreement_sum"] / n_tokens if n_tokens > 0 else None
            topk_agreements = {
                k: (
                    res["topk_agreement_sums"][k] / n_tokens
                    if n_tokens > 0
                    else None
                )
                for k in topk_list
            }

            out[mode][hf] = {
                "ppl_original": ppl_orig,
                "ppl_masked": ppl_mask,
                "ppl_delta": ppl_delta,
                "kl_mean": kl_mean,
                "logit_mse_mean": logit_mse_mean,
                "top1_agreement": top1_agreement,
                "topk_agreements": topk_agreements,
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
                row[f"top{k}_agreement"] = topk_agreements[k]

            rows.append(row)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    fieldnames = [
        "mask_mode",
        "hot_frac",
        "ppl_original",
        "ppl_masked",
        "ppl_delta",
        "kl_mean",
        "logit_mse_mean",
        "top1_agreement",
        "n_tokens",
        "count_batches",
    ] + [f"top{k}_agreement" for k in topk_list]

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print("Done. Results written to:", args.output_json, args.output_csv)


if __name__ == "__main__":
    main()
