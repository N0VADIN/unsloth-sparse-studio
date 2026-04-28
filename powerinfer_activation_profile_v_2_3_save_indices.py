#!/usr/bin/env python3
# powerinfer_activation_profile_v2_3_save_indices.py
#
# v2.3_save_indices: v2.3_patch + optional save_indices + try/finally hook cleanup
#
# Usage example:
# python powerinfer_activation_profile_v2_3_save_indices.py \
#   --model_path ./models/TinyLlama-1.1B-Chat-v1.0 \
#   --layers "0,1,2,3" --num_sequences 128 --seq_len 128 --batch_size 8 \
#   --hot_fracs "0.01,0.02,0.05,0.1" --do_error_analysis True --sample_error_tokens 4096 \
#   --dtype float16 --save_indices True \
#   --output_json activation_profile_with_indices.json

import argparse
import json
import math
import random
from typing import List

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_list(s: str, cast=float):
    if s is None:
        return []
    return [cast(x.strip()) for x in s.split(",") if x.strip()]


def find_down_proj_modules(model, config):
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


class StreamingStats:
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        self.hidden = int(hidden_size)
        self.eps = float(eps)
        self.total_tokens = 0
        self.active_counts = np.zeros(self.hidden, dtype=np.int64)
        self.sum_abs = np.zeros(self.hidden, dtype=np.float64)

    def update(self, x_cpu: torch.Tensor, token_mask):
        # token_mask may be numpy array or torch tensor
        if isinstance(token_mask, np.ndarray):
            token_mask_t = torch.from_numpy(token_mask).bool()
        else:
            token_mask_t = token_mask.bool().cpu()

        if token_mask_t.sum().item() == 0:
            return

        x_sel = x_cpu[token_mask_t]  # (N_valid, H)
        n = x_sel.shape[0]
        self.total_tokens += int(n)

        # ensure float32 for numpy conversion
        abs_x = x_sel.float().abs().numpy()
        self.sum_abs += abs_x.sum(axis=0)
        self.active_counts += (abs_x > self.eps).sum(axis=0).astype(np.int64)


def flatten_last_dim(tensor: torch.Tensor):
    if tensor.dim() == 3:
        B, S, H = tensor.shape
        return tensor.reshape(B * S, H), B, S
    elif tensor.dim() == 2:
        return tensor, None, None
    else:
        return tensor.reshape(-1, tensor.shape[-1]), None, None


def reservoir_add(
    reservoir: List[torch.Tensor],
    reservoir_seen: int,
    candidates: torch.Tensor,
    k: int,
):
    N = candidates.shape[0]

    for i in range(N):
        reservoir_seen += 1

        if len(reservoir) < k:
            reservoir.append(candidates[i].clone())
        else:
            j = random.randrange(reservoir_seen)
            if j < k:
                reservoir[j] = candidates[i].clone()

    return reservoir, reservoir_seen


def down_proj_error(down_proj, sample_inputs: List[torch.Tensor], hot_idx):
    """
    Batched, numerically stable down-proj error.

    sample_inputs: list of tensors (N_chunk, H) on CPU (float32)
    hot_idx: iterable of indices
    """
    if not sample_inputs:
        return {}

    # Cast weights/bias to float32 on CPU for stable matmuls
    W = down_proj.weight.detach().cpu().float()
    b = (
        down_proj.bias.detach().cpu().float()
        if down_proj.bias is not None
        else None
    )

    hot_idx_t = torch.as_tensor(list(hot_idx), dtype=torch.long)
    mask = torch.zeros(W.shape[1], dtype=torch.float32)
    mask[hot_idx_t] = 1.0

    mse_sum = 0.0
    rel_sum = 0.0
    cos_sum = 0.0
    total = 0

    for X in sample_inputs:
        X = X.cpu().float()

        Y_orig = X @ W.t()
        if b is not None:
            Y_orig = Y_orig + b

        X_mask = X * mask
        Y_mask = X_mask @ W.t()
        if b is not None:
            Y_mask = Y_mask + b

        diff = Y_orig - Y_mask

        mse_sum += (diff.pow(2).mean().item()) * X.shape[0]

        orig_norm = Y_orig.norm(dim=1)
        diff_norm = diff.norm(dim=1)
        rel_sum += (
            diff_norm / (orig_norm + 1e-12)
        ).sum().item()

        cos = torch.nn.functional.cosine_similarity(
            Y_orig,
            Y_mask,
            dim=1,
            eps=1e-8,
        )
        cos_sum += cos.sum().item()
        total += X.shape[0]

    return {
        "mse": mse_sum / total,
        "relative_norm_mean": rel_sum / total,
        "cosine_mean": cos_sum / total,
        "tokens": total,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--num_sequences", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--layers", type=str, default="all")

    parser.add_argument(
        "--hot_fracs",
        "--hot-fracs",
        dest="hot_fracs",
        type=str,
        default="0.01,0.02,0.05,0.1",
    )

    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--use_random", type=str, default="True")
    parser.add_argument("--prompts_file", type=str, default=None)
    parser.add_argument("--do_error_analysis", type=str, default="False")
    parser.add_argument("--sample_error_tokens", type=int, default=2048)

    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
    )

    parser.add_argument(
        "--save_indices",
        "--save-indices",
        dest="save_indices",
        type=str,
        default="True",
        help="Save order_by_freq and order_by_mass in the output JSON",
    )

    parser.add_argument(
        "--output_json",
        type=str,
        default="activation_profile_v2_3_save_indices.json",
    )

    args = parser.parse_args()

    device = torch.device(args.device)
    use_random = args.use_random.lower() in ("true", "1", "yes")
    do_error = args.do_error_analysis.lower() in ("true", "1", "yes")
    save_indices = args.save_indices.lower() in ("true", "1", "yes")

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    model_dtype = dtype_map.get(args.dtype, torch.float32)

    hot_fracs = parse_list(args.hot_fracs, float)
    hot_fracs = [f for f in hot_fracs if 0.0 < f <= 1.0]

    if not hot_fracs:
        raise RuntimeError(
            "No valid hot_fracs provided. Use values in (0,1]."
        )

    print(
        "Loading model:",
        args.model_path,
        "dtype:",
        args.dtype,
        "device:",
        device,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=model_dtype,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    model.to(device)
    model.eval()

    down_modules = find_down_proj_modules(model, model.config)

    if not down_modules:
        raise RuntimeError(
            "No FFN down_proj modules found by heuristics."
        )

    if args.layers == "all":
        selected = list(range(len(down_modules)))
    else:
        selected = [
            int(x)
            for x in args.layers.split(",")
            if x.strip()
        ]

    selected = [
        i for i in selected
        if 0 <= i < len(down_modules)
    ]

    if not selected:
        raise RuntimeError("No valid layers selected.")

    print("Selected layers:", selected)

    stats = {}
    for idx in selected:
        name, module = down_modules[idx]
        stats[idx] = StreamingStats(
            module.in_features,
            eps=args.eps,
        )

    hook_handles = []
    hook_storage = {idx: [] for idx in selected}

    reservoirs = {idx: [] for idx in selected}
    reservoir_seen = {idx: 0 for idx in selected}
    reservoir_k = {
        idx: int(args.sample_error_tokens)
        for idx in selected
    }

    def make_hook(idx):
        def hook(mod, inp):
            x = inp[0].detach().cpu()
            hook_storage[idx].append(x)
        return hook

    for idx in selected:
        name, module = down_modules[idx]
        h = module.register_forward_pre_hook(make_hook(idx))
        hook_handles.append(h)

    if not use_random and args.prompts_file:
        with open(args.prompts_file, "r", encoding="utf-8") as f:
            prompts = [l.strip() for l in f if l.strip()]

        while len(prompts) < args.num_sequences:
            prompts = prompts + prompts

        prompts = prompts[: args.num_sequences]
    else:
        prompts = None

    vocab_size = model.get_input_embeddings().num_embeddings
    total = args.num_sequences
    steps = math.ceil(total / args.batch_size)

    print(
        f"Forward pass: sequences={total}, "
        f"seq_len={args.seq_len}, "
        f"batch_size={args.batch_size}, "
        f"steps={steps}"
    )

    try:
        with torch.no_grad():
            for step in tqdm(range(steps), desc="Forward"):
                bstart = step * args.batch_size
                bend = min(total, bstart + args.batch_size)
                b = bend - bstart

                if prompts:
                    enc = tokenizer(
                        prompts[bstart:bend],
                        return_tensors="pt",
                        padding="max_length",
                        max_length=args.seq_len,
                        truncation=True,
                    )

                    input_ids = enc["input_ids"]
                    attn_mask = enc["attention_mask"]
                else:
                    input_ids = torch.randint(
                        0,
                        vocab_size,
                        (b, args.seq_len),
                        dtype=torch.long,
                    )
                    attn_mask = torch.ones_like(
                        input_ids,
                        dtype=torch.long,
                    )

                input_ids = input_ids.to(device)
                attn_mask = attn_mask.to(device)

                model(
                    input_ids,
                    attention_mask=attn_mask,
                    use_cache=False,
                )

                for idx in selected:
                    for x in hook_storage[idx]:
                        flat, B, S = flatten_last_dim(x)

                        if B is not None and S is not None:
                            token_mask = (
                                attn_mask.reshape(B * S)
                                .cpu()
                                .bool()
                                .numpy()
                            )
                        else:
                            token_mask = np.ones(
                                (flat.shape[0],),
                                dtype=bool,
                            )

                        stats[idx].update(flat, token_mask)

                        if do_error and reservoir_k[idx] > 0:
                            valid_idx = np.nonzero(token_mask)[0]
                            if valid_idx.size > 0:
                                candidates = flat[valid_idx]
                                reservoirs[idx], reservoir_seen[idx] = reservoir_add(
                                    reservoirs[idx],
                                    reservoir_seen[idx],
                                    candidates,
                                    reservoir_k[idx],
                                )

                    hook_storage[idx].clear()

    finally:
        for h in hook_handles:
            try:
                h.remove()
            except Exception:
                pass

    results = {}

    for idx in selected:
        st = stats[idx]

        if st.total_tokens == 0:
            print(
                f"Warning: layer {idx} observed zero tokens; skipping."
            )
            continue

        freq = st.active_counts / max(1, st.total_tokens)
        order_by_freq = np.argsort(-freq)
        order_by_mass = np.argsort(-st.sum_abs)

        cov_freq = []
        cov_mass = []

        for f in hot_fracs:
            k = max(1, int(math.ceil(f * st.hidden)))

            top_freq = order_by_freq[:k]
            top_mass = order_by_mass[:k]

            mass_cov_freq = (
                st.sum_abs[top_freq].sum()
                / (st.sum_abs.sum() + 1e-12)
            )

            mass_cov_mass = (
                st.sum_abs[top_mass].sum()
                / (st.sum_abs.sum() + 1e-12)
            )

            cov_freq.append({
                "hot_frac": f,
                "k": int(k),
                "mass_coverage_by_freq": float(mass_cov_freq),
            })

            cov_mass.append({
                "hot_frac": f,
                "k": int(k),
                "mass_coverage_by_mass": float(mass_cov_mass),
            })

        name, module = down_modules[idx]

        info = {
            "name": name,
            "hidden_size": int(st.hidden),
            "total_tokens": int(st.total_tokens),
            "mean_freq": float(freq.mean()),
            "median_freq": float(np.median(freq)),
            "almost_inactive": int((freq <= 0.001).sum()),
            "activity_definition": f"abs(x) > {args.eps}",
            "coverage_by_freq": cov_freq,
            "coverage_by_mass": cov_mass,
            "reservoir_seen": int(reservoir_seen[idx]),
            "reservoir_size": int(len(reservoirs[idx])),
        }

        if info["mean_freq"] > 0.95:
            print(
                f"Warning: Layer {idx} ({name}) almost fully active "
                f"(mean_freq={info['mean_freq']:.4f})."
            )

        if save_indices:
            info["order_by_freq"] = (
                order_by_freq.astype(int).tolist()
            )
            info["order_by_mass"] = (
                order_by_mass.astype(int).tolist()
            )

        if do_error and reservoirs[idx]:
            try:
                sample_tensor = torch.stack(
                    [
                        s if s.dim() == 1 else s.squeeze(0)
                        for s in reservoirs[idx]
                    ],
                    dim=0,
                ).float()
                sample_chunks = [sample_tensor]
            except Exception:
                sample_chunks = [
                    s.unsqueeze(0).float()
                    if s.dim() == 1
                    else s.float()
                    for s in reservoirs[idx]
                ]

            info["error_analysis"] = {}

            for entry in cov_freq:
                k = entry["k"]
                hot_idx = order_by_freq[:k]
                err = down_proj_error(
                    module,
                    sample_chunks,
                    hot_idx,
                )

                info["error_analysis"][
                    f"top_{entry['hot_frac']}_by_freq"
                ] = {
                    "k": int(k),
                    **err,
                }

            for entry in cov_mass:
                k = entry["k"]
                hot_idx = order_by_mass[:k]
                err = down_proj_error(
                    module,
                    sample_chunks,
                    hot_idx,
                )

                info["error_analysis"][
                    f"top_{entry['hot_frac']}_by_mass"
                ] = {
                    "k": int(k),
                    **err,
                }

        results[idx] = info

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Done. Results saved to", args.output_json)


if __name__ == "__main__":
    main()
