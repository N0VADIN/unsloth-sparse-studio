# PowerInfer Pipeline Runbook

This runbook documents the recommended 3-step workflow for the PowerInfer dense pipeline.

## Step 1: Activation profiling

```bash
python powerinfer_activation_profile_v2_3_save_indices.py \
  --model_path ./models/TinyLlama-1.1B-Chat-v1.0 \
  --layers "all" \
  --num_sequences 128 \
  --seq_len 128 \
  --batch_size 8 \
  --hot_fracs "0.01,0.02,0.05,0.1" \
  --do_error_analysis True \
  --sample_error_tokens 4096 \
  --dtype float16 \
  --save_indices True \
  --output_json activation_profile_with_indices.json
```

## Step 2: Full-model mask evaluation

First dry-run:

```bash
python powerinfer_full_model_mask_eval_v2_fix_dryrun.py \
  --model_path ./models/TinyLlama-1.1B-Chat-v1.0 \
  --profile_json activation_profile_with_indices.json \
  --dry_run
```

Quick smoke test:

```bash
python powerinfer_full_model_mask_eval_v2_fix_dryrun.py \
  --model_path ./models/TinyLlama-1.1B-Chat-v1.0 \
  --profile_json activation_profile_with_indices.json \
  --quick
```

Full evaluation:

```bash
python powerinfer_full_model_mask_eval_v2_fix_dryrun.py \
  --model_path ./models/TinyLlama-1.1B-Chat-v1.0 \
  --profile_json activation_profile_with_indices.json \
  --prompts_file prompts.txt \
  --seq_len 512 \
  --eval_batch_size 1 \
  --hot_fracs "0.05,0.1,0.2" \
  --mask_modes "global_by_freq,global_by_mass,oracle_dynamic_topk_abs" \
  --topk_list "1,5,10" \
  --dtype float16 \
  --metric_chunk_size 128 \
  --output_json full_model_mask_eval.json \
  --output_csv full_model_mask_eval.csv
```

## Step 3: Plot results

```bash
python powerinfer_plot_results_v2.py \
  --eval_json full_model_mask_eval.json \
  --profile_json activation_profile_with_indices.json \
  --output_dir plots \
  --formats "png,pdf" \
  --html_report plots/report.html
```

## Mask mode quick explanation

- `global_by_freq`: keeps neurons ranked by activation frequency.
- `global_by_mass`: keeps neurons ranked by cumulative activation magnitude.
- `oracle_dynamic_topk_abs`: per-token dynamic top-k mask, useful as an upper bound.

## Practical guidance

- Use `--dry_run` before expensive runs.
- Avoid `oracle_original_topk_abs` on large models unless enough GPU RAM is available.
- For large models, start with `--global_only`, `eval_batch_size=1`, and `metric_chunk_size=64` or `128`.

## Canonical script filenames

- `powerinfer_activation_profile_v2_3_save_indices.py`
- `powerinfer_full_model_mask_eval_v2_fix_dryrun.py`
- `powerinfer_plot_results_v2.py`
