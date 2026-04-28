#!/usr/bin/env python3
# powerinfer_plot_results_v2.py
#
# Plot results from powerinfer_full_model_mask_eval and optional profiler JSON.
# Produces:
# - PPL delta vs hot_frac (per mask mode)
# - PPL masked vs retained fraction (hot_frac)
# - KL vs hot_frac
# - Top-1 agreement vs hot_frac
# - Top-k overlap vs hot_frac (for each requested k)
# - (optional) Activation mass coverage curves if profiler contains cumulative_mass_by_mass or sum_abs
# - (optional) HTML report
#
# Usage example:
# python powerinfer_plot_results_v2.py \
#   --eval_json full_model_mask_eval_v3.json \
#   --profile_json activation_profile_with_indices.json \
#   --output_dir plots \
#   --formats "png,pdf" \
#   --html_report plots/report.html
#
# To display output, append: --show

import argparse
import base64
import csv
import json
import os
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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
    return x is not None and isinstance(x, (int, float)) and np.isfinite(x)


def extract_hot_fracs_from_eval(eval_json: Dict[str, Any]) -> List[float]:
    meta = eval_json.get("profile_metadata") or eval_json.get("profile_meta") or {}
    if "hot_fracs" in meta and isinstance(meta["hot_fracs"], list):
        return [float(x) for x in meta["hot_fracs"] if is_finite_number(float(x))]

    for _k, v in eval_json.items():
        if isinstance(v, dict):
            try:
                fracs = sorted([float(x) for x in v.keys()])
                if fracs:
                    return fracs
            except Exception:
                continue
    return []


def _looks_like_metrics_dict(d: Dict[str, Any]) -> bool:
    metric_keys = {"ppl_masked", "kl_mean", "top1_agreement", "topk_overlap", "ppl_delta", "logit_mse_mean"}
    return any(k in d for k in metric_keys)


def gather_table(eval_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for mode, mode_data in eval_json.items():
        if not isinstance(mode_data, dict):
            continue

        candidate_items = []
        for hf_key, metrics in mode_data.items():
            if not isinstance(metrics, dict):
                continue
            if not _looks_like_metrics_dict(metrics):
                continue
            candidate_items.append((hf_key, metrics))

        if not candidate_items:
            continue

        for hf_key, metrics in candidate_items:
            try:
                hf = float(hf_key)
            except Exception:
                try:
                    hf = float(metrics.get("hot_frac", hf_key))
                except Exception:
                    continue

            if not is_finite_number(hf):
                continue

            rows.append(
                {
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
            )

    return sorted(rows, key=lambda r: (r["mask_mode"], r["hot_frac"]))


def _series(rows: List[Dict[str, Any]], mode: str, y_key: str):
    xs = []
    ys = []
    for r in rows:
        if r["mask_mode"] != mode:
            continue
        x = r.get("hot_frac")
        y = r.get(y_key)
        if is_finite_number(x) and is_finite_number(y):
            xs.append(x)
            ys.append(y)
    return xs, ys


def plot_ppl_delta(rows: List[Dict[str, Any]], out_dir: str, fmt_list: List[str], keep_open: bool = False):
    plt.figure(figsize=(8, 5))
    modes = sorted(set(r["mask_mode"] for r in rows))
    for mode in modes:
        xs, ys = _series(rows, mode, "ppl_delta")
        if xs:
            plt.plot(xs, ys, marker="o", label=mode)
    plt.xscale("log")
    plt.xlabel("hot_frac")
    plt.ylabel("PPL delta (masked - original)")
    plt.title("Perplexity Delta vs Hot Fraction")
    plt.legend()
    plt.tight_layout()
    for fmt in fmt_list:
        plt.savefig(os.path.join(out_dir, f"ppl_delta.{fmt}"))
    if not keep_open:
        plt.close()


def plot_ppl_masked(rows: List[Dict[str, Any]], out_dir: str, fmt_list: List[str], keep_open: bool = False):
    plt.figure(figsize=(8, 5))
    modes = sorted(set(r["mask_mode"] for r in rows))
    for mode in modes:
        xs, ys = _series(rows, mode, "ppl_masked")
        if xs:
            plt.plot(xs, ys, marker="o", label=mode)
    plt.xscale("log")
    plt.xlabel("Retained fraction (hot_frac)")
    plt.ylabel("PPL (masked model)")
    plt.title("PPL masked vs retained fraction")
    plt.legend()
    plt.tight_layout()
    for fmt in fmt_list:
        plt.savefig(os.path.join(out_dir, f"ppl_vs_retained_fraction.{fmt}"))
    if not keep_open:
        plt.close()


def plot_kl(rows: List[Dict[str, Any]], out_dir: str, fmt_list: List[str], keep_open: bool = False):
    plt.figure(figsize=(8, 5))
    modes = sorted(set(r["mask_mode"] for r in rows))
    for mode in modes:
        xs, ys = _series(rows, mode, "kl_mean")
        if xs:
            plt.plot(xs, ys, marker="o", label=mode)
    plt.xscale("log")
    plt.xlabel("hot_frac")
    plt.ylabel("KL (original || masked)")
    plt.title("KL Divergence vs Hot Fraction")
    plt.legend()
    plt.tight_layout()
    for fmt in fmt_list:
        plt.savefig(os.path.join(out_dir, f"kl_vs_hot_frac.{fmt}"))
    if not keep_open:
        plt.close()


def plot_logit_mse(rows: List[Dict[str, Any]], out_dir: str, fmt_list: List[str], keep_open: bool = False):
    plt.figure(figsize=(8, 5))
    modes = sorted(set(r["mask_mode"] for r in rows))
    plotted = False
    for mode in modes:
        xs, ys = _series(rows, mode, "logit_mse_mean")
        if xs:
            plotted = True
            plt.plot(xs, ys, marker="o", label=mode)
    if not plotted:
        if not keep_open:
            plt.close()
        return
    plt.xscale("log")
    plt.xlabel("hot_frac")
    plt.ylabel("Logit MSE mean")
    plt.title("Logit MSE vs Hot Fraction")
    plt.legend()
    plt.tight_layout()
    for fmt in fmt_list:
        plt.savefig(os.path.join(out_dir, f"logit_mse_vs_hot_frac.{fmt}"))
    if not keep_open:
        plt.close()


def plot_top1(rows: List[Dict[str, Any]], out_dir: str, fmt_list: List[str], keep_open: bool = False):
    plt.figure(figsize=(8, 5))
    modes = sorted(set(r["mask_mode"] for r in rows))
    for mode in modes:
        xs, ys = _series(rows, mode, "top1_agreement")
        if xs:
            plt.plot(xs, ys, marker="o", label=mode)
    plt.xscale("log")
    plt.xlabel("hot_frac")
    plt.ylabel("Top-1 agreement")
    plt.title("Top-1 Agreement vs Hot Fraction")
    plt.legend()
    plt.tight_layout()
    for fmt in fmt_list:
        plt.savefig(os.path.join(out_dir, f"top1_vs_hot_frac.{fmt}"))
    if not keep_open:
        plt.close()


def plot_topk_overlap(rows: List[Dict[str, Any]], out_dir: str, fmt_list: List[str], topk_list: List[int], keep_open: bool = False):
    for k in topk_list:
        plt.figure(figsize=(8, 5))
        modes = sorted(set(r["mask_mode"] for r in rows))
        any_plot = False
        for mode in modes:
            xs, ys = [], []
            for r in rows:
                if r["mask_mode"] != mode or not is_finite_number(r.get("hot_frac")):
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
            if not keep_open:
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
        if not keep_open:
            plt.close()


def _coverage_xy(cum: np.ndarray):
    x = (np.arange(1, len(cum) + 1) / len(cum)) * 100.0
    x = np.concatenate([[0.0], x])
    y = np.concatenate([[0.0], cum])
    return x, y


def plot_mass_coverage_from_profile(
    profile_json: Dict[str, Any], out_dir: str, fmt_list: List[str], keep_open: bool = False
):
    layers: Dict[int, Dict[str, Any]] = {}
    for k, v in profile_json.items():
        if isinstance(k, str) and k.isdigit() and isinstance(v, dict):
            layers[int(k)] = v
        elif isinstance(k, int) and isinstance(v, dict):
            layers[int(k)] = v

    if not layers:
        return

    layer_indices = sorted(layers.keys())[: min(3, len(layers))]
    plt.figure(figsize=(8, 5))
    plotted = False

    mean_cum_mass = None
    n_layers = 0
    for layer_idx in layer_indices:
        entry = layers[layer_idx]
        cm_mass = entry.get("cumulative_mass_by_mass")
        cm_freq = entry.get("cumulative_mass_by_freq")
        sum_abs = entry.get("sum_abs")

        if cm_mass is not None and len(cm_mass) > 0:
            cum = np.array(cm_mass, dtype=float)
            x, y = _coverage_xy(cum)
            plt.plot(x, y, label=f"layer {layer_idx} (by mass)")
            plotted = True
            mean_cum_mass = cum.copy() if mean_cum_mass is None else (mean_cum_mass + cum)
            n_layers += 1

        if cm_freq is not None and len(cm_freq) > 0:
            cum = np.array(cm_freq, dtype=float)
            x, y = _coverage_xy(cum)
            plt.plot(x, y, label=f"layer {layer_idx} (by freq)", linestyle="--")
            plotted = True

        if sum_abs is not None and len(sum_abs) > 0:
            sum_abs_arr = np.array(sum_abs, dtype=float)
            order_by_mass = np.argsort(-sum_abs_arr)
            sorted_mass = sum_abs_arr[order_by_mass]
            cum_mass = np.cumsum(sorted_mass) / (sorted_mass.sum() + 1e-12)
            x, y = _coverage_xy(cum_mass)
            plt.plot(x, y, label=f"layer {layer_idx} (mass from sum_abs)")
            plotted = True

    if mean_cum_mass is not None and n_layers > 0:
        mean_cum = mean_cum_mass / n_layers
        x, y = _coverage_xy(mean_cum)
        plt.plot(x, y, label=f"mean cumulative mass ({n_layers} layers)", linewidth=2, color="black")
        plotted = True

    if not plotted:
        if not keep_open:
            plt.close()
        return

    plt.xlabel("Top X% neurons")
    plt.ylabel("Cumulative activation mass (fraction)")
    plt.title("Activation mass coverage across layers")
    plt.ylim(0, 1.02)
    plt.legend(loc="lower right")
    plt.tight_layout()
    for fmt in fmt_list:
        plt.savefig(os.path.join(out_dir, f"activation_mass_coverage.{fmt}"))
    if not keep_open:
        plt.close()


def generate_html_report(image_paths: List[str], output_html: str):
    safe_mkdir(os.path.dirname(output_html) if os.path.dirname(output_html) else ".")
    html_parts = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>PowerInfer Benchmark Results</title></head><body>",
        "<h1>PowerInfer Benchmark Results</h1>",
    ]

    embeddable_exts = {"png", "jpg", "jpeg", "webp"}
    for p in image_paths:
        ext = os.path.splitext(p)[1].lstrip(".").lower()
        if ext not in embeddable_exts:
            report_dir = os.path.dirname(output_html) or "."
            link_path = os.path.relpath(p, start=report_dir)
            html_parts.append(f"<p><a href='{link_path}'>{os.path.basename(p)}</a></p>")
            continue

        with open(p, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        html_parts.append(f"<h3>{os.path.basename(p)}</h3>")
        html_parts.append(f"<img style='max-width:100%;height:auto' src='data:image/{ext};base64,{b64}'/>")

    html_parts.append("</body></html>")
    with open(output_html, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))
    print(f"HTML report written to: {output_html}")


def write_summary_csv(rows: List[Dict[str, Any]], out_dir: str, csv_name: str = "summary_table.csv"):
    path = os.path.join(out_dir, csv_name)
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
    ]

    topk_keys = set()
    for r in rows:
        tk = r.get("topk_overlap")
        if isinstance(tk, dict):
            topk_keys.update(tk.keys())

    topk_keys = sorted(list(topk_keys), key=lambda x: int(x) if str(x).isdigit() else 0)
    fieldnames += [f"top{k}_overlap" for k in topk_keys]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            row = {k: r.get(k) for k in fieldnames if k in r}
            tk = r.get("topk_overlap") or {}
            for k in topk_keys:
                val = tk.get(str(k)) if str(k) in tk else tk.get(k)
                row[f"top{k}_overlap"] = val if is_finite_number(val) else None
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Plot PowerInfer benchmark results",
        epilog=(
            "Examples:\n"
            "  headless: python powerinfer_plot_results_v2.py --eval_json results.json\n"
            "  with HTML: python powerinfer_plot_results_v2.py --eval_json results.json --html_report report.html\n"
            "  display: python powerinfer_plot_results_v2.py --eval_json results.json --html_report report.html --show"
        ),
    )
    parser.add_argument("--eval_json", type=str, required=True)
    parser.add_argument("--profile_json", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="plots")
    parser.add_argument("--formats", type=str, default="png")
    parser.add_argument("--topk_list", type=str, default=None)
    parser.add_argument("--html_report", type=str, default=None)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    fmt_list = [x.strip().lower() for x in args.formats.split(",") if x.strip()]
    safe_mkdir(args.output_dir)

    eval_json = load_eval_json(args.eval_json)
    profile_json = load_profile_json(args.profile_json) if args.profile_json else None

    rows = gather_table(eval_json)
    if not rows:
        raise RuntimeError("No metric rows found in eval JSON.")

    topk_list = parse_list(args.topk_list, int)
    if not topk_list:
        ks = set()
        for r in rows:
            tk = r.get("topk_overlap")
            if isinstance(tk, dict):
                for k in tk.keys():
                    try:
                        ks.add(int(k))
                    except Exception:
                        pass
        topk_list = sorted(ks)

    write_summary_csv(rows, args.output_dir)

    keep_figures_open = args.show and not args.html_report

    plot_ppl_delta(rows, args.output_dir, fmt_list, keep_open=keep_figures_open)
    plot_ppl_masked(rows, args.output_dir, fmt_list, keep_open=keep_figures_open)
    plot_kl(rows, args.output_dir, fmt_list, keep_open=keep_figures_open)
    plot_logit_mse(rows, args.output_dir, fmt_list, keep_open=keep_figures_open)
    plot_top1(rows, args.output_dir, fmt_list, keep_open=keep_figures_open)
    if topk_list:
        plot_topk_overlap(rows, args.output_dir, fmt_list, topk_list, keep_open=keep_figures_open)
    if profile_json:
        plot_mass_coverage_from_profile(profile_json, args.output_dir, fmt_list, keep_open=keep_figures_open)

    produced = sorted(
        [
            os.path.join(args.output_dir, n)
            for n in os.listdir(args.output_dir)
            if os.path.splitext(n)[1].lstrip(".").lower() in set(fmt_list + ["csv"])
        ]
    )

    if args.html_report:
        generate_html_report([p for p in produced if p.endswith(tuple(fmt_list))], args.html_report)

    print("\n=== Summary ===")
    print(f"metric rows: {len(rows)}")
    print(f"mask_modes: {sorted(set(r['mask_mode'] for r in rows))}")
    print(f"hot_fracs: {sorted(set(r['hot_frac'] for r in rows))}")
    print("generated files:")
    for p in produced:
        print(f"  - {p}")
    if args.html_report:
        print(f"html report: {args.html_report}")

    if args.show:
        if args.html_report:
            import webbrowser

            webbrowser.open(f"file://{os.path.abspath(args.html_report)}")
        else:
            plt.show(block=True)


if __name__ == "__main__":
    main()
