# Hanoi Full Comparison Runbook

## Purpose

This runbook is for the full comparison workflow on the current branch.

It covers:

- selecting the best existing Hanoi transporter checkpoint
- exporting the full reranker dataset
- training the current `reranker-transporter` policy
- running three full benchmark families for comparison
  - base `transporter`
  - older `vlm-transporter`
  - current `reranker-transporter`
- generating one combined comparison report

This runbook does not retrain the transporter. It reuses the existing checkpoints under `./checkpoints`.

## Recommended Comparison Modes

There are two useful ways to compare the older VLM architecture:

1. Real VLM comparison
   - use `hf-chat` or local `smolvlm`
   - measures the actual model-dependent older approach
2. Architecture-only comparison
   - use `--backend=heuristic`
   - removes VLM model quality as a confounder and isolates the older pipeline itself

For the latest architecture, the comparison target is `reranker-transporter`. It does not require a live VLM worker.

## 1. Activate the Ravens Environment

```bash
source /home/praharsh/perception/perception_env/bin/activate
cd /home/praharsh/perception/ravens
export PYTHONPATH=${PWD}
```

## 2. Optional: Regenerate Hanoi Datasets

Skip this if `./towers-of-hanoi-train` and `./towers-of-hanoi-test` already exist and are complete.

```bash
python ravens/demos.py --assets_root=./ravens/environments/assets --task=towers-of-hanoi --mode=train --n=100
python ravens/demos.py --assets_root=./ravens/environments/assets --task=towers-of-hanoi --mode=test --n=100
```

## 3. Optional: Train or Reuse Transporter Checkpoints

Skip this if the Hanoi transporter checkpoints already exist under `./checkpoints`.

```bash
python ravens/train.py --task=towers-of-hanoi --agent=transporter --n_demos=10 --n_steps=5000
python ravens/train.py --task=towers-of-hanoi --agent=transporter --n_demos=20 --n_steps=1000
python ravens/train.py --task=towers-of-hanoi --agent=transporter --n_demos=100 --n_steps=1000
```

## 4. Select the Best Existing Transporter Checkpoint

This step chooses the checkpoint that will be used by the reranker export and training pipeline.

```bash
python ravens/hanoi_select_checkpoint.py \
  --root_dir=. \
  --data_dir=. \
  --assets_root=./ravens/environments/assets \
  --task=towers-of-hanoi \
  --split=test \
  --episodes=100 \
  --output_path=./hanoi-reranker-full/checkpoint-selection.json
```

Expected artifact:

- `./hanoi-reranker-full/checkpoint-selection.json`

Important:

- If the selected checkpoint changes from the current smoke result, use that selected checkpoint for all downstream reranker steps.
- The base and older comparison commands below still use explicit `--n_demos`, `--run_index`, and `--n_steps`, so update those values if your selected checkpoint is not `towers-of-hanoi-transporter-10-0` step `4000`.

## 5. Export the Full Reranker Dataset

```bash
python ravens/hanoi_export_reranker_dataset.py \
  --root_dir=. \
  --data_dir=. \
  --assets_root=./ravens/environments/assets \
  --task=towers-of-hanoi \
  --selection_manifest=./hanoi-reranker-full/checkpoint-selection.json \
  --output_dir=./hanoi-reranker-full/export \
  --split=both \
  --episodes=100
```

Expected artifacts:

- `./hanoi-reranker-full/export/train.npz`
- `./hanoi-reranker-full/export/test.npz`
- `./hanoi-reranker-full/export/train_audit.jsonl`
- `./hanoi-reranker-full/export/test_audit.jsonl`
- `./hanoi-reranker-full/export/metadata.json`

## 6. Train the Full Reranker

```bash
python ravens/hanoi_train_reranker.py \
  --train_path=./hanoi-reranker-full/export/train.npz \
  --test_path=./hanoi-reranker-full/export/test.npz \
  --output_dir=./hanoi-reranker-full/model \
  --epochs=50 \
  --batch_size=64 \
  --learning_rate=1e-3 \
  --patience=5 \
  --validation_fraction=0.2
```

Expected artifacts:

- `./hanoi-reranker-full/model/model.weights.h5`
- `./hanoi-reranker-full/model/metadata.json`

## 7. Optional: Prepare the Older VLM Worker

### Option A: Hugging Face hosted VLM

If you want the older architecture compared with a real hosted open-source VLM:

```bash
export HF_TOKEN="<HF_TOKEN>"
export HANOI_VLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
```

Benchmark worker command:

```bash
/home/praharsh/perception/perception_env/bin/python ravens/hanoi_vlm_worker.py \
  --backend=hf-chat \
  --model_id=${HANOI_VLM_MODEL}
```

### Option B: Local SmolVLM

```bash
source /home/praharsh/perception/smolvlm_env/bin/activate
pip install -r /home/praharsh/perception/ravens/requirements-smolvlm.txt
```

Benchmark worker command:

```bash
/home/praharsh/perception/smolvlm_env/bin/python ravens/hanoi_vlm_worker.py \
  --backend=smolvlm \
  --device=cpu \
  --model_id=HuggingFaceTB/SmolVLM-256M-Instruct
```

### Option C: Heuristic Worker

Use this when you want to compare the older architecture itself without live VLM variance.

```bash
/home/praharsh/perception/perception_env/bin/python ravens/hanoi_vlm_worker.py \
  --backend=heuristic
```

## 8. Run the Full Comparison Benchmarks

Create one directory for the comparison suite:

```bash
mkdir -p ./hanoi-results-full-compare
```

### 8A. Base Transporter

Replace the `10 / 0 / 4000` triplet if your selected transporter differs.

```bash
python ravens/hanoi_benchmark.py \
  --assets_root=./ravens/environments/assets \
  --data_dir=. \
  --root_dir=. \
  --task=towers-of-hanoi \
  --agent=transporter \
  --n_demos=10 \
  --n_steps=4000 \
  --episodes=100 \
  --output_dir=./hanoi-results-full-compare
```

### 8B. Older VLM-Transporter

Recommended flags:

- `--candidate_mode=legal`
- `--decision_policy=guarded`
- `--history_images=2`

#### Real hosted VLM example

```bash
python ravens/hanoi_benchmark.py \
  --assets_root=./ravens/environments/assets \
  --data_dir=. \
  --root_dir=. \
  --task=towers-of-hanoi \
  --agent=vlm-transporter \
  --candidate_mode=legal \
  --decision_policy=guarded \
  --history_images=2 \
  --n_demos=10 \
  --n_steps=4000 \
  --episodes=100 \
  --vlm_timeout_s=300 \
  --vlm_command="/home/praharsh/perception/perception_env/bin/python ravens/hanoi_vlm_worker.py --backend=hf-chat --model_id=${HANOI_VLM_MODEL}" \
  --output_dir=./hanoi-results-full-compare
```

#### Local SmolVLM example

```bash
python ravens/hanoi_benchmark.py \
  --assets_root=./ravens/environments/assets \
  --data_dir=. \
  --root_dir=. \
  --task=towers-of-hanoi \
  --agent=vlm-transporter \
  --candidate_mode=legal \
  --decision_policy=guarded \
  --history_images=2 \
  --n_demos=10 \
  --n_steps=4000 \
  --episodes=100 \
  --vlm_timeout_s=300 \
  --vlm_command="/home/praharsh/perception/smolvlm_env/bin/python ravens/hanoi_vlm_worker.py --backend=smolvlm --device=cpu --model_id=HuggingFaceTB/SmolVLM-256M-Instruct" \
  --output_dir=./hanoi-results-full-compare
```

#### Heuristic architecture-only example

```bash
python ravens/hanoi_benchmark.py \
  --assets_root=./ravens/environments/assets \
  --data_dir=. \
  --root_dir=. \
  --task=towers-of-hanoi \
  --agent=vlm-transporter \
  --candidate_mode=legal \
  --decision_policy=guarded \
  --history_images=2 \
  --n_demos=10 \
  --n_steps=4000 \
  --episodes=100 \
  --vlm_command="/home/praharsh/perception/perception_env/bin/python ravens/hanoi_vlm_worker.py --backend=heuristic" \
  --output_dir=./hanoi-results-full-compare
```

### 8C. Current Reranker-Transporter

```bash
python ravens/hanoi_benchmark.py \
  --assets_root=./ravens/environments/assets \
  --data_dir=. \
  --root_dir=. \
  --task=towers-of-hanoi \
  --agent=reranker-transporter \
  --candidate_mode=legal \
  --episodes=100 \
  --reranker_dir=./hanoi-reranker-full/model \
  --output_dir=./hanoi-results-full-compare
```

## 9. Generate the Combined Comparison Report

```bash
python ravens/hanoi_report.py \
  --input_dir=./hanoi-results-full-compare \
  --output_dir=./hanoi-results-full-compare/report
```

Expected artifacts:

- `summary.csv`
- `summary.json`
- `success_vs_steps.png`
- `legal_move_rate.png`
- `oracle_move_agreement.png`
- `steps_to_success.png`
- `latency.png`
- `legal_candidate_coverage.png`
- `planner_override_rate.png`
- `executed_oracle_move_agreement.png`
- `repeat_state_rate.png`
- `reranker_latency.png`

## 10. How to Read the Final Comparison

For the current comparison, focus on:

- `success_rate`
  - full-episode solve rate
- `oracle_move_agreement`
  - whether the selector is choosing the correct next symbolic move
- `executed_oracle_move_agreement`
  - whether the move that actually ran matched the oracle after any guard/override
- `repeat_state_rate`
  - whether the policy is drifting into loops or dead states
- `mean_total_latency_s`
  - wall-clock decision cost per step
- `planner_override_rate`
  - how often the older guarded VLM path needed the planner to rescue a bad selection
- `mean_reranker_latency_s`
  - extra learned-selector overhead for the latest architecture

Interpretation:

- If `transporter` has low `success_rate`, the raw action head is not enough on its own.
- If `vlm-transporter` solves but needs overrides or has high latency, the older architecture is workable but expensive and model-dependent.
- If `reranker-transporter` matches or beats the old path with lower selector latency and no VLM worker, it is the better default for this branch.

## 11. Current Smoke Validation Reference

The latest smoke validation artifacts from this branch are:

- current reranker:
  - `/tmp/codex-hanoi-reranker/benchmark-smoke/towers-of-hanoi-reranker-transporter-legal-10-0-4000.json`
- older architecture comparable slice:
  - `/tmp/codex-hanoi-compare-old/towers-of-hanoi-vlm-transporter-legal-guarded-h2-10-0-4000.json`
- base transporter comparable slice:
  - `/tmp/codex-hanoi-compare-base/towers-of-hanoi-transporter-10-0-4000.json`
- combined comparison report:
  - `/tmp/codex-hanoi-compare-all/report/summary.json`

These are reference artifacts only. The commands above are the proper full-run path.
