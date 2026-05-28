# Hanoi Two-Stage Training Report

## Executive Summary

- Training can improve full-episode Hanoi completion on this branch, but the recommended path on the current CPU-only machine is not joint SmolVLM plus transporter fine-tuning.
- The main bottleneck in the current implementation is legal-move selection, not legal-move generation. The transporter already provides a useful scored candidate set; the failure mode is choosing the correct symbolic move consistently over all seven steps.
- This implementation adds a CPU-feasible two-stage alternative:
  - stage 1 selects the best existing Hanoi transporter checkpoint by oracle agreement on legal candidates
  - stage 2 trains a small supervised MLP reranker on symbolic Hanoi state plus candidate features
- The existing `vlm-transporter` path remains in the branch as a baseline. SmolVLM fine-tuning is documented as deferred future work, not implemented here.

## Current Bottleneck Evidence From This Branch

- Earlier branch work already fixed the physical execution bug in the legal Hanoi path. The benchmark now executes the chosen legal move with oracle-style object-relative geometry instead of raw pixel-to-world pick/place poses.
- After that fix, the limiting factor is move choice quality:
  - the transporter can usually produce the oracle move inside the legal candidate set
  - small VLM backends still fail to select the correct move reliably across all seven steps
  - guarded symbolic execution can solve the full episode without retraining, which further isolates the bottleneck to selection policy quality
- This branch therefore treats the transporter as a candidate generator and adds a lightweight learned reranker on top of legal symbolic moves.

## Why True Joint SmolVLM + Transporter Training Is Not Recommended Here

- Joint fine-tuning would require a larger training budget, additional multimodal supervision infrastructure, and GPU memory that is not available on the current machine.
- The current branch does not yet have a clean end-to-end differentiable interface between image-conditioned VLM reasoning and the transporter action head. The practical control surface in this codebase is a discrete legal-candidate list, not raw pixel actions.
- Even if fine-tuning were attempted, the dataset size in the local Hanoi splits is small for stable multimodal adaptation. That makes overfitting likely unless the setup is expanded substantially.
- The branch already contains enough structure to train a CPU-feasible selector on top of legal candidates with much lower engineering and runtime cost.

## CPU-Feasible Two-Stage Architecture

### Stage 1: Transporter Checkpoint Selection

- New script: `ravens/hanoi_select_checkpoint.py`
- New shared implementation: `ravens/hanoi_reranker.py`
- The selector scans `checkpoints/towers-of-hanoi-transporter-*`, loads each available step, and evaluates it on held-out Hanoi states restored from the saved dataset split.
- Metric:
  - top-1 legal-candidate oracle agreement
- Secondary diagnostics:
  - legal candidate coverage
  - mean candidate count
  - mean transporter latency
- The chosen checkpoint is persisted in a JSON manifest and reused by dataset export, reranker training, and benchmark inference.
- A minimum quality gate is enforced:
  - `oracle_move_agreement >= 0.75`
- If no checkpoint clears that threshold, the reranker pipeline should stop and the report should conclude that transporter retraining is required on a later GPU-capable pass.

### Stage 2: Supervised Legal-Move Reranker

- New script: `ravens/hanoi_export_reranker_dataset.py`
- New script: `ravens/hanoi_train_reranker.py`
- The exporter restores each saved Hanoi state from the existing train/test splits, computes the optimal symbolic move for the real Ravens target peg, and emits one row per legal candidate at each state.
- The reranker is a small TensorFlow/Keras MLP:
  - input feature vector
  - `Dense(64, relu)`
  - `Dense(64, relu)`
  - `Dense(1)`
- Loss and optimization:
  - binary cross-entropy from logits
  - Adam
  - default learning rate `1e-3`
- Early stopping:
  - deterministic held-out validation groups from the training export
  - patience `5`
- Inference:
  - score each legal candidate independently
  - execute the argmax logit

## Feature Schema and Training Objective

- Feature implementation lives in `ravens/hanoi_reranker.py`.
- The default feature vector has 21 dimensions:
  - 9 one-hot state features for disk placement across the three pegs
  - 3 one-hot features for candidate disk rank
  - 3 one-hot features for candidate source peg
  - 3 one-hot features for candidate target peg
  - 1 transporter score feature
  - 1 candidate rank-fraction feature
  - 1 normalized step-index feature
- Labels:
  - `1` for the oracle legal move at that state
  - `0` for all other legal candidates
- Training target:
  - rank the oracle move above all other legal candidates in each state group
- Exported audit metadata includes:
  - episode index
  - random seed
  - step index
  - candidate index
  - oracle candidate index
  - candidate description
  - peg-state text
  - transporter score

## Implemented Code Changes

### New Files

- `ravens/hanoi_reranker.py`
  - checkpoint discovery and evaluation
  - feature construction
  - dataset export helpers
  - MLP model build/train/load
  - runtime reranker policy wrapper
- `ravens/hanoi_select_checkpoint.py`
  - offline transporter checkpoint sweep CLI
- `ravens/hanoi_export_reranker_dataset.py`
  - candidate-level dataset export CLI
- `ravens/hanoi_train_reranker.py`
  - CPU reranker training CLI

### Updated Files

- `ravens/hanoi_benchmark.py`
  - adds `--agent=reranker-transporter`
  - adds `--reranker_dir`
  - loads the selected transporter from reranker metadata
  - records reranker latency and reranker logits in per-step diagnostics
  - includes reranker metadata in result JSON
  - reports reranker runs with the selected transporter step in the output filename and payload
- `ravens/hanoi_utils.py`
  - adds `mean_reranker_latency_s` to episode-summary aggregation
  - adds symbolic Hanoi shortest-path solving toward the real task goal peg
  - restores saved object poses from dataset info snapshots during export/evaluation
  - switches peg-stack ordering from raw `z` to disk-rank order for symbolic robustness
  - hardens local transporter patch selection when a candidate center falls outside the crop
- `ravens/hanoi_report.py`
  - aggregates reranker runs distinctly by `reranker_dir`
  - includes `reranker_dir` in summary payloads
  - adds a reranker latency plot path for reranker-only runs
- `ravens/hanoi_reranker.py`
  - adds group-based train/validation splitting
  - keeps the test export strictly for final evaluation
  - removes the incompatible logits-plus-AUC training metric path

## New CLI, Interfaces, and Artifact Layout

### Checkpoint Selection

```bash
PYTHONPATH=/home/praharsh/perception/ravens \
/home/praharsh/perception/perception_env/bin/python ravens/hanoi_select_checkpoint.py \
  --root_dir=. \
  --data_dir=. \
  --assets_root=./ravens/environments/assets \
  --task=towers-of-hanoi \
  --split=test \
  --episodes=100 \
  --output_path=/tmp/codex-hanoi-reranker/checkpoint-selection.json
```

Output:

- selection manifest with `selected_checkpoint`
- ranked `all_checkpoints`

### Dataset Export

```bash
PYTHONPATH=/home/praharsh/perception/ravens \
/home/praharsh/perception/perception_env/bin/python ravens/hanoi_export_reranker_dataset.py \
  --root_dir=. \
  --data_dir=. \
  --assets_root=./ravens/environments/assets \
  --task=towers-of-hanoi \
  --selection_manifest=/tmp/codex-hanoi-reranker/checkpoint-selection.json \
  --output_dir=/tmp/codex-hanoi-reranker/export \
  --split=both \
  --episodes=100
```

Outputs:

- `train.npz`
- `test.npz`
- `train_audit.jsonl`
- `test_audit.jsonl`
- `metadata.json`

### Reranker Training

```bash
PYTHONPATH=/home/praharsh/perception/ravens \
/home/praharsh/perception/perception_env/bin/python ravens/hanoi_train_reranker.py \
  --train_path=/tmp/codex-hanoi-reranker/export/train.npz \
  --test_path=/tmp/codex-hanoi-reranker/export/test.npz \
  --output_dir=/tmp/codex-hanoi-reranker/model \
  --epochs=50 \
  --batch_size=64 \
  --learning_rate=1e-3 \
  --patience=5 \
  --validation_fraction=0.2
```

Outputs:

- `model.weights.h5`
- `metadata.json`

### Benchmark Inference

```bash
PYTHONPATH=/home/praharsh/perception/ravens \
/home/praharsh/perception/perception_env/bin/python ravens/hanoi_benchmark.py \
  --assets_root=./ravens/environments/assets \
  --data_dir=. \
  --root_dir=. \
  --task=towers-of-hanoi \
  --agent=reranker-transporter \
  --candidate_mode=legal \
  --episodes=100 \
  --reranker_dir=/tmp/codex-hanoi-reranker/model \
  --output_dir=/tmp/codex-hanoi-reranker/benchmark
```

New benchmark interface:

- `--agent=reranker-transporter`
- `--reranker_dir`

## Validation Protocol and Acceptance Criteria

- Checkpoint sweep sanity:
  - select a single checkpoint from the available Hanoi transporter runs
  - verify it clears the `0.75` oracle-agreement threshold
- Dataset export sanity:
  - each exported state must contain exactly one positive oracle candidate label
  - each state must contain at least two legal candidates
- Reranker offline evaluation:
  - report train, validation, and held-out test metrics
  - candidate-level top-1 oracle agreement is the main metric
- End-to-end benchmark:
  - run `reranker-transporter` on held-out test episodes in `candidate_mode=legal`
  - track success rate, oracle agreement, legal move rate, repeat-state rate, and reranker latency
- Regression:
  - confirm the existing `vlm-transporter` path still executes after the reranker integration

## Results

### Important Scope Note

- The implementation is complete, but the validation in this turn was intentionally scaled down for CPU time.
- The code paths support the 100-episode commands shown above.
- The executed validation used:
  - existing 3-episode checkpoint selection manifest
  - 10 exported train episodes
  - 10 exported test episodes
  - 5 reranker benchmark test episodes
  - 1 `vlm-transporter` heuristic regression episode

### Commands Executed

Checkpoint selection already available in `/tmp/codex-hanoi-reranker/checkpoint-selection.json`:

```bash
PYTHONPATH=/home/praharsh/perception/ravens \
/home/praharsh/perception/perception_env/bin/python ravens/hanoi_select_checkpoint.py \
  --root_dir=. \
  --data_dir=. \
  --assets_root=./ravens/environments/assets \
  --task=towers-of-hanoi \
  --split=test \
  --episodes=3 \
  --output_path=/tmp/codex-hanoi-reranker/checkpoint-selection.json
```

Smoke export:

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=/home/praharsh/perception/ravens \
/home/praharsh/perception/perception_env/bin/python ravens/hanoi_export_reranker_dataset.py \
  --root_dir=. \
  --data_dir=. \
  --assets_root=./ravens/environments/assets \
  --task=towers-of-hanoi \
  --selection_manifest=/tmp/codex-hanoi-reranker/checkpoint-selection.json \
  --output_dir=/tmp/codex-hanoi-reranker/export-smoke \
  --split=both \
  --episodes=10
```

Smoke reranker training:

```bash
PYTHONPATH=/home/praharsh/perception/ravens \
/home/praharsh/perception/perception_env/bin/python ravens/hanoi_train_reranker.py \
  --train_path=/tmp/codex-hanoi-reranker/export-smoke/train.npz \
  --test_path=/tmp/codex-hanoi-reranker/export-smoke/test.npz \
  --output_dir=/tmp/codex-hanoi-reranker/model-smoke \
  --epochs=50 \
  --batch_size=64 \
  --learning_rate=1e-3 \
  --patience=5 \
  --validation_fraction=0.2
```

Smoke reranker benchmark:

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=/home/praharsh/perception/ravens \
/home/praharsh/perception/perception_env/bin/python ravens/hanoi_benchmark.py \
  --assets_root=./ravens/environments/assets \
  --data_dir=. \
  --root_dir=. \
  --task=towers-of-hanoi \
  --agent=reranker-transporter \
  --candidate_mode=legal \
  --episodes=5 \
  --reranker_dir=/tmp/codex-hanoi-reranker/model-smoke \
  --output_dir=/tmp/codex-hanoi-reranker/benchmark-smoke
```

Regression run for the existing VLM path:

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=/home/praharsh/perception/ravens \
/home/praharsh/perception/perception_env/bin/python ravens/hanoi_benchmark.py \
  --assets_root=./ravens/environments/assets \
  --data_dir=. \
  --root_dir=. \
  --task=towers-of-hanoi \
  --agent=vlm-transporter \
  --candidate_mode=legal \
  --n_demos=10 \
  --n_steps=1000 \
  --episodes=1 \
  --vlm_command="/home/praharsh/perception/perception_env/bin/python ravens/hanoi_vlm_worker.py --backend=heuristic" \
  --output_dir=/tmp/codex-hanoi-reranker/vlm-regression
```

Report aggregation:

```bash
PYTHONPATH=/home/praharsh/perception/ravens \
/home/praharsh/perception/perception_env/bin/python ravens/hanoi_report.py \
  --input_dir=/tmp/codex-hanoi-reranker/benchmark-smoke \
  --output_dir=/tmp/codex-hanoi-reranker/benchmark-smoke-report
```

### Observed Metrics

- Selected transporter checkpoint:
  - `towers-of-hanoi-transporter-10-0` step `4000`
  - manifest oracle agreement `1.0` on the existing 3-episode selection run
  - legal candidate coverage `1.0`
- Smoke export dataset:
  - train: `74` states, `212` candidate rows, `74` positive labels
  - test: `70` states, `200` candidate rows, `70` positive labels
  - mean candidates per state: `2.8649` train, `2.8571` test
- Smoke reranker training:
  - feature dimension: `21`
  - hidden sizes: `64 -> 64`
  - epochs trained: `50`
  - train candidate top-1 oracle agreement: `1.0`
  - validation candidate top-1 oracle agreement: `1.0`
  - test candidate top-1 oracle agreement: `1.0`
- Smoke reranker benchmark:
  - artifact: `/tmp/codex-hanoi-reranker/benchmark-smoke/towers-of-hanoi-reranker-transporter-legal-10-0-4000.json`
  - success rate: `1.0` over `5` test episodes
  - mean reward: `1.0`
  - oracle move agreement: `1.0`
  - executed oracle move agreement: `1.0`
  - legal move rate: `1.0`
  - legal candidate coverage: `1.0`
  - repeat state rate: `0.0`
  - mean steps: `7.0`
  - mean transporter latency: `2.8963s`
  - mean reranker latency: `0.000587s`
- VLM regression run:
  - artifact: `/tmp/codex-hanoi-reranker/vlm-regression/towers-of-hanoi-vlm-transporter-legal-guarded-h2-10-0-1000.json`
  - success rate: `1.0` over `1` episode
  - oracle move agreement: `0.7143`
  - planner override rate: `0.2857`
  - executed oracle move agreement: `1.0`
  - legal candidate coverage: `1.0`
  - confirms the existing `vlm-transporter` guarded path still runs after the reranker changes

### Debugging Notes That Changed the Final Design

- The first reranker export attempt failed because replaying stored continuous dataset actions drifted the environment off the intended symbolic state.
- The second export attempt failed because some saved Hanoi states have enough physical noise that raw `z` order does not identify the top disk correctly.
- The final export/evaluation design therefore uses:
  - restored saved object poses for each dataset step
  - symbolic stack order based on disk rank rather than raw `z`
  - symbolic shortest-path solving toward the real Ravens target peg
  - explicit candidate matching by `(disk_id, target_peg)` instead of trusting `candidate.matches_oracle` in restored states
- The training loop originally compiled with an `AUC` metric on logits and failed on this TensorFlow build. That metric path was removed because grouped top-1 oracle agreement is the decision metric that matters here.

### Acceptance Status

- Checkpoint selection implementation: complete
- Dataset export implementation: complete
- Reranker training implementation: complete
- `reranker-transporter` benchmark integration: complete
- Standalone report: complete
- Full 100-episode CPU validation: not run in this turn

## Deferred Appendix: What SmolVLM Fine-Tuning Would Require

- A GPU-capable machine with enough memory for multimodal supervised fine-tuning or parameter-efficient adapters
- A curated multimodal dataset that pairs:
  - current observation
  - optional short history
  - legal candidate board
  - oracle candidate choice
  - consistent textual reasoning target if chain-of-thought style supervision is desired
- A clean training harness that can:
  - serialize the exact multimodal prompt format used at inference
  - evaluate candidate-choice accuracy across full episodes
  - compare adapter-tuned VLM policies against the symbolic reranker baseline
- A decision about whether the VLM should:
  - directly choose from legal candidates
  - produce a symbolic move description that is post-mapped to a legal candidate
  - or serve as a feature extractor for a separate selector head
- In the current branch state, the lightweight reranker is the lower-risk path and the right first training baseline before investing in VLM fine-tuning.
