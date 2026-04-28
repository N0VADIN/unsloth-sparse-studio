#!/usr/bin/env python3
# powerinfer_plot_results_v2.py
#
# Plot results from powerinfer_full_model_mask_eval and optional profiler JSON.
# Produces:
#  - PPL delta vs hot_frac (per mask mode)
#  - PPL masked vs hot_frac (retained fraction)
#  - KL vs hot_frac
#  - Top-1 agreement vs hot_frac
#  - Top-k overlap vs hot_frac (for each requested k)
#  - (optional) Activation mass coverage curves if profiler contains cumulative_mass_by_mass or sum_abs
#  - (optional) HTML report with embedded images
#
# Usage example:
# python powerinfer_plot_results_v2.py \
#   --eval_json full_model_mask_eval_v3.json \
#   --profile_json activation_profile_with_indices.json \
#   --output_dir plots \
#   --formats "png,pdf" \
#   --html_report report.html \
#   --show
#
# Requirements: matplotlib, seaborn, numpy, pandas (optional but recommended)
# Install: pip install matplotlib seaborn numpy pandas

import argparse
import json
import math
import os
import base64
from typing import Dict, Any, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import pandas as pd
    _HAS_PANDAS = True
except Exception:
    _HAS_PANDAS = False

sns.set(style="whitegrid")


def parse_list(s: Optional[str], cast=float):
    if s is None or s == "":
        return []
    return [cast(x.strip()) for x in s.split(",") if x.strip()]


def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def load_eval_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_profile_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_finite_number(x):
    """Helper to filter non-finite values (None, nan, inf)"""
    return x is not None and isinstance(x, (int, float)) and np.isfinite(x)


def extract_hot_fracs_from_eval(eval_json: Dict[str, Any]) -> List[float]:
    # eval_json structure: top-level keys include mask modes; each mode maps hot_frac -> metrics
    # Try to find hot_fracs from metadata first
    meta = eval_json.get("profile_metadata") or eval_json.get("profile_meta") or {}
    if "hot_fracs" in meta and isinstance(meta["hot_fracs"], list):
        return [float(x) for x in meta["hot_fracs"]]
    # fallback: collect from first mask_mode
    for k, v in eval_json.items():
        if isinstance(v, dict):
            # find numeric keys convertible to float
            try:
                fracs = sorted([float(x) for x in v.keys()])
                if fracs:
                    return fracs
            except Exception:
                continue
    return []


def gather_table(eval_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert eval JSON into a flat table list of dicts:
    {mask_mode, hot_frac, ppl_original, ppl_masked, ppl_delta, kl_mean, logit_mse_mean, top1_agreement, topk_overlap(dict), n_tokens}
    """
    rows = []
    # top-level keys: metadata + mask modes
    for mode, mode_data in eval_json.items():
        if mode in ("profile_json", "profile_metadata", "model_path", "dtype", "mask_modes", "hot_fracs", "profile_version"):
            continue
        if not isinstance(mode_data, dict):
            continue
        for hf_str, metrics in mode_data.items():
            try:
                hf = float(hf_str)
            except Exception:
                # sometimes hot_fracs are stored as numbers already
                try:
                    hf = float(metrics.get("hot_frac", hf_str))
                except Exception:
                    continue
            row = {
                "mask_mode": mode,
                "hot_frac": hf,
                "ppl_original": metrics.get("ppl_original"),
                "ppl_masked": metrics.get("ppl_masked"),
                "ppl_delta": metrics.get("ppl_delta"),
                "kl_mean": metrics.get("kl_mean"),
                "logit_mse_mean": metrics.get("logit_mse_mean"),
                "top1_agreement": metrics.get("top1_agreement"),
                "topk_overlap": metrics.get("topk_overlap"),
                "n_tokens": metrics.get("n_tokens"),
                "count_batches": metrics.get("count_batches"),
            }
            rows.append(row)
    # sort rows by mode then hot_frac
    rows = sorted(rows, key=lambda r: (r["mask_mode"], r["hot_frac"]))
    return rows


def plot_ppl_delta(rows: List[Dict[str, Any]], out_dir: str, fmt_list: List[str]):
    plt.figure(figsize=(8, 5))
    modes = sorted(set(r["mask_mode"] for r in rows))
    for mode in modes:
        xs = [r["hot_frac"] for r in rows if r["mask_mode"] == mode and is_finite_number(r["ppl_delta"])]
        ys = [r["ppl_delta"] for r in rows if r["mask_mode"] == mode and is_finite_number(r["ppl_delta"])]
        if not xs:
            continue
        plt.plot(xs, ys, marker="o", label=mode)
    plt.xscale("log")
    plt.xlabel("hot_frac")
    plt.ylabel("PPL delta (masked - original)")
    plt.title("Perplexity Delta vs Hot Fraction")
    plt.legend()
    plt.tight_layout()
    for fmt in fmt_list:
        plt.savefig(os.path.join(out_dir, f"ppl_delta.{fmt}"))
    plt.close()


def plot_ppl_masked(rows: List[Dict[str, Any]], out_dir: str, fmt_list: List[str]):
    """Plot masked PPL vs retained fraction (hot_frac)"""
    plt.figure(figsize=(8, 5))
    modes = sorted(set(r["mask_mode"] for r in rows))
    for mode in modes:
        xs = [r["hot_frac"] for r in rows if r["mask_mode"] == mode and is_finite_number(r["ppl_masked"])]
        ys = [r["ppl_masked"] for r in rows if r["mask_mode"] == mode and is_finite_number(r["ppl_masked"])]
        if not xs:
            continue
        plt.plot(xs, ys, marker="o", label=mode)
    plt.xscale("log")
    plt.xlabel("Retained fraction (hot_frac)")
    plt.ylabel("PPL (masked model)")
    plt.title("Masked Model Perplexity vs Retained Fraction")
    plt.legend()
    plt.tight_layout()
    for fmt in fmt_list:
        plt.savefig(os.path.join(out_dir, f"ppl_masked_vs_retained.{fmt}"))
    plt.close()


def plot_kl(rows: List[Dict[str, Any]], out_dir: str, fmt_list: List[str]):
    plt.figure(figsize=(8, 5))
    modes = sorted(set(r["mask_mode"] for r in rows))
    for mode in modes:
        xs = [r["hot_frac"] for r in rows if r["mask_mode"] == mode and is_finite_number(r["kl_mean"])]
        ys = [r["kl_mean"] for r in rows if r["mask_mode"] == mode and is_finite_number(r["kl_mean"])]
        if not xs:
            continue
        plt.plot(xs, ys, marker="o", label=mode)
    plt.xscale("log")
    plt.xlabel("hot_frac")
    plt.ylabel("KL (original || masked)")
    plt.title("KL Divergence vs Hot Fraction")
    plt.legend()
    plt.tight_layout()
    for fmt in fmt_list:
        plt.savefig(os.path.join(out_dir, f"kl_vs_hot_frac.{fmt}"))
    plt.close()


def plot_top1(rows: List[Dict[str, Any]], out_dir: str, fmt_list: List[str]):
    plt.figure(figsize=(8, 5))
    modes = sorted(set(r["mask_mode"] for r in rows))
    for mode in modes:
        xs = [r["hot_frac"] for r in rows if r["mask_mode"] == mode and is_finite_number(r["top1_agreement"])]
        ys = [r["top1_agreement"] for r in rows if r["mask_mode"] == mode and is_finite_number(r["top1_agreement"])]
        if not xs:
            continue
        plt.plot(xs, ys, marker="o", label=mode)
    plt.xscale("log")
    plt.xlabel("hot_frac")
    plt.ylabel("Top-1 agreement")
    plt.title("Top-1 Agreement vs Hot Fraction")
    plt.legend()
    plt.tight_layout()
    for fmt in fmt_list:
        plt.savefig(os.path.join(out_dir, f"top1_vs_hot_frac.{fmt}"))
    plt.close()


def plot_topk_overlap(rows: List[Dict[str, Any]], out_dir: str, fmt_list: List[str], topk_list: List[int]):
    # For each k, plot modes vs hot_frac
    for k in topk_list:
        plt.figure(figsize=(8, 5))
        modes = sorted(set(r["mask_mode"] for r in rows))
        any_plot = False
        for mode in modes:
            xs = []
            ys = []
            for r in rows:
                if r["mask_mode"] != mode:
                    continue
                tk = r.get("topk_overlap")
                if not isinstance(tk, dict):
                    continue
                val = tk.get(str(k)) if str(k) in tk else tk.get(k)
                if not is_finite_number(val):
                    continue
                xs.append(r["hot_frac"])
                ys.append(val)
            if xs:
                any_plot = True
                plt.plot(xs, ys, marker="o", label=mode)
        if not any_plot:
            plt.close()
            continue
        plt.xscale("log")
        plt.xlabel("hot_frac")
        plt.ylabel(f"Top-{k} overlap (fraction)")
        plt.title(f"Top-{k} Overlap vs Hot Fraction")
        plt.legend()
        plt.tight_layout()
        for fmt in fmt_list:
            plt.savefig(os.path.join(out_dir, f"top{k}_overlap_vs_hot_frac.{fmt}"))
        plt.close()


def plot_mass_coverage_from_profile(profile_json: Dict[str, Any], out_dir: str, fmt_list: List[str]):
    """
    If profile contains cumulative_mass_by_mass or cumulative_mass_by_freq or raw sum_abs, plot cumulative curves.
    Fixed to start at 0% and 0.0 for all curves.
    """
    # profile may contain per-layer entries and metadata
    # try to find a representative layer (first numeric key)
    layers = {}
    meta = {}
    for k, v in profile_json.items():
        if isinstance(k, str) and k.isdigit() and isinstance(v, dict):
            layers[int(k)] = v
        elif isinstance(k, int) and isinstance(v, dict):
            layers[int(k)] = v
        else:
            meta[k] = v

    if not layers:
        return

    # Plot per-layer curves for first few layers
    layer_indices = sorted(layers.keys())[:min(3, len(layers))]
    plt.figure(figsize=(8, 5))
    plotted = False

    # Also compute mean curve across all layers if possible
    mean_cum_mass = None
    mean_cum_freq = None
    n_layers = 0

    for layer_idx in layer_indices:
        entry = layers[layer_idx]

        # prefer cumulative_mass_by_mass or cumulative_mass_by_freq if present
        cm_mass = entry.get("cumulative_mass_by_mass")
        cm_freq = entry.get("cumulative_mass_by_freq")
        sum_abs = entry.get("sum_abs")
        freq = entry.get("freq")

        if cm_mass is not None:
            n = len(cm_mass)
            cum = np.array(cm_mass, dtype=float)
            # build x starting at 0% and y starting at 0.0 so curve begins at 0%
            x = (np.arange(1, n + 1) / float(n)) * 100.0
            x = np.concatenate([[0.0], x])
            y = np.concatenate([[0.0], cum])
            plt.plot(x, y, label=f"layer {layer_idx} (by mass)")
            plotted = True
            # accumulate for mean
            if mean_cum_mass is None:
                mean_cum_mass = cum.copy()
            else:
                mean_cum_mass += cum
            n_layers += 1

        if cm_freq is not None:
            n = len(cm_freq)
            cum = np.array(cm_freq, dtype=float)
            x = (np.arange(1, n + 1) / float(n)) * 100.0
            x = np.concatenate([[0.0], x])
            y = np.concatenate([[0.0], cum])
            plt.plot(x, y, label=f"layer {layer_idx} (by freq)", linestyle="--")
            plotted = True

        if sum_abs is not None and freq is not None:
            # compute cumulative mass sorted by mass and by freq
            sum_abs_arr = np.array(sum_abs, dtype=float)
            order_by_mass = np.argsort(-sum_abs_arr)
            sorted_mass = sum_abs_arr[order_by_mass]
            cum_mass = np.cumsum(sorted_mass) / (sorted_mass.sum() + 1e-12)
            n = len(cum_mass)
            x = (np.arange(1, n + 1) / float(n)) * 100.0
            x = np.concatenate([[0.0], x])
            y = np.concatenate([[0.0], cum_mass])
            plt.plot(x, y, label=f"layer {layer_idx} (mass from sum_abs)")
            plotted = True

    # Plot mean cumulative mass if we have data
    if mean_cum_mass is not None and n_layers > 0:
        mean_cum = mean_cum_mass / n_layers
        n = len(mean_cum)
        x = (np.arange(1, n + 1) / float(n)) * 100.0
        x = np.concatenate([[0.0], x])
        y = np.concatenate([[0.0], mean_cum])
        plt.plot(x, y, label=f"mean cumulative mass ({n_layers} layers)", linewidth=2, color="black", linestyle="-")
        plotted = True

    if not plotted:
        plt.close()
        return

    plt.xlabel("Top X% neurons")
    plt.ylabel("Cumulative activation mass (fraction)")
    plt.title("Activation mass coverage across layers")
    plt.legend(loc="lower right")
    plt.tight_layout()
    for fmt in fmt_list:
        plt.savefig(os.path.join(out_dir, f"activation_mass_coverage.{fmt}"))
    plt.close()


def generate_html_report(image_paths: List[str], output_html: str):
    """Generate HTML report with embedded raster images (PNG/JPG/WebP)"""
    safe_mkdir(os.path.dirname(output_html) if os.path.dirname(output_html) else ".")
    
    html_parts = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<meta charset='utf-8'>",
        "<title>PowerInfer Benchmark Results</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; margin: 20px; }",
        "h1 { color: #333; }",
        "h3 { color: #666; margin-top: 20px; }",
        "img { border: 1px solid #ccc; margin: 10px 0; }",
        "a { color: #0066cc; }",
        "</style>",
        "</head>",
        "<body>",
        "<h1>PowerInfer Benchmark Results</h1>",
    ]
    
    embeddable_exts = {"png", "jpg", "jpeg", "webp"}
    for p in image_paths:
        ext = os.path.splitext(p)[1].lstrip(".").lower()
        if ext in embeddable_exts:
            with open(p, "rb") as f:
                b = f.read()
            b64 = base64.b64encode(b).decode("ascii")
            html_parts.append(f"<h3>{os.path.basename(p)}</h3>")
            html_parts.append(f"<img src='data:image/{ext};base64,{b64}' style='max-width:100%;height:auto;'/>")
        else:
            # For PDF or other formats, just link
            html_parts.append(f"<p><a href='{os.path.basename(p)}'>{os.path.basename(p)}</a></p>")
    
    html_parts.extend([
        "</body>",
        "</html>"
    ])
    
    with open(output_html, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))
    
    print(f"HTML report written to: {output_html}")


def write_summary_csv(rows: List[Dict[str, Any]], out_dir: str, csv_name: str = "summary_table.csv"):
    # write a simple CSV summarizing key metrics
    import csv
    path = os.path.join(out_dir, csv_name)
    fieldnames = ["mask_mode", "hot_frac", "ppl_original", "ppl_masked", "ppl_delta", "kl_mean", "logit_mse_mean", "top1_agreement", "n_tokens", "count_batches"]
    # include topk keys if present in any row
    topk_keys = set()
    for r in rows:
        tk = r.get("topk_overlap")
        if isinstance(tk, dict):
            for k in tk.keys():
                topk_keys.add(k)
    topk_keys = sorted(list(topk_keys), key=lambda x: int(x) if str(x).isdigit() else 0)
    fieldnames += [f"top{k}_overlap" for k in topk_keys]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            row = {k: r.get(k) for k in fieldnames if k in r}
            # fill topk fields
            tk = r.get("topk_overlap") or {}
            for k in topk_keys:
                val = tk.get(str(k)) if str(k) in tk else tk.get(int(k)) if isinstance(k, int) else tk.get(k)
                row[f"top{k}_overlap"] = val if is_finite_number(val) else None
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Plot PowerInfer benchmark results",
        epilog="Examples:\n"
               "  headless: python powerinfer_plot_results_v2.py --eval_json results.json\n"
               "  with HTML: python powerinfer_plot_results_v2.py --eval_json results.json --html_report report.html\n"
               "  display: python powerinfer_plot_results_v2.py --eval_json results.json --html_report report.html --show"
    )
    parser.add_argument("--eval_json", type=str, required=True, help="Full-model eval JSON (output of powerinfer_full_model_mask_eval)")
    parser.add_argument("--profile_json", type=str, required=False, default=None, help="Optional profiler JSON (activation_profile_with_indices.json)")
    parser.add_argument("--output_dir", type=str, default="plots")
    parser.add_argument("--formats", type=str, default="png", help="Comma-separated formats, e.g. png,pdf")
    parser.add_argument("--topk_list", type=str, default=None, help="Comma-separated top-k values to plot (overrides values found in eval JSON)")
    parser.add_argument("--html_report", type=str, default=None, help="Generate HTML report with embedded images (e.g., report.html)")
    parser.add_argument("--show", action="store_true", help="Open HTML report after generation (if --html_report provided), or show first image")
    args = parser.parse_args()

    fmt_list = [x.strip() for x in args.formats.split(",") if x.strip()]
    safe_mkdir(args.output_dir)

    eval_json = load_eval_json(args.eval_json)
    profile_json = None
    if args.profile_json:
        profile_json = load_profile_json(args.profile_json)

    rows = gather_table(eval_json)
    if not rows:
        raise RuntimeError("No metric rows found in eval JSON. Check file format.")

    # determine topk_list
    if args.topk_list:
        topk_list = [int(x) for x in args.topk_list.split(",") if x.strip()]
    else:
        # try to infer from rows
        found = set()
        for r in rows:
            tk = r.get("topk_overlap")
            if isinstance(tk, dict):
                for k in tk.keys():
                    try:
                        found.add(int(k))
                    except Exception:
                        pass
        topk_list = sorted(list(found))

    # write summary CSV
    write_summary_csv(rows, args.output_dir, csv_name="summary_table_from_eval.csv")

    # plots
    plot_ppl_delta(rows, args.output_dir, fmt_list)
    plot_ppl_masked(rows, args.output_dir, fmt_list)
    plot_kl(rows, args.output_dir, fmt_list)
    plot_top1(rows, args.output_dir, fmt_list)
    if topk_list:
        plot_topk_overlap(rows, args.output_dir, fmt_list, topk_list)

    # optional profile mass coverage
    if profile_json is not None:
        plot_mass_coverage_from_profile(profile_json, args.output_dir, fmt_list)

    print(f"Plots written to: {args.output_dir}")

    # HTML report generation
    if args.html_report:
        # Collect all generated image files
        image_files = []
        for fmt in fmt_list:
            for f in os.listdir(args.output_dir):
                if f.endswith(fmt):
                    image_files.append(os.path.join(args.output_dir, f))
        
        # Filter to embeddable formats for HTML
        embeddable_exts = {".png", ".jpg", ".jpeg", ".webp"}
        html_images = [p for p in image_files if os.path.splitext(p)[1].lower() in embeddable_exts]
        
        if html_images:
            generate_html_report(html_images, args.html_report)
            if args.show:
                print(f"Opening HTML report: {args.html_report}")
                import webbrowser
                webbrowser.open(f"file://{os.path.abspath(args.html_report)}")
        else:
            print(f"Warning: No embeddable images (PNG/JPG/WebP) found in {args.output_dir}")

    # Simple show mode for legacy usage (show first image if no HTML report)
    elif args.show:
        # List files and open first image using matplotlib (non-blocking)
        files = sorted([os.path.join(args.output_dir, f) for f in os.listdir(args.output_dir) 
                       if any(f.endswith("." + ext) for ext in fmt_list)])
        if files:
            print(f"Opening first image: {files[0]}")
            img = plt.imread(files[0])
            plt.figure(figsize=(8, 6))
            plt.imshow(img)
            plt.axis("off")
            plt.title(os.path.basename(files[0]))
            plt.show()
        else:
            print("No images found to display.")


if __name__ == "__main__":
    main()