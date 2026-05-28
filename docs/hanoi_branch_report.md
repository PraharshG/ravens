# Hanoi Branch Report

## Executive Summary

- `feat/hanoi-vlm-transporter` and `master` both currently resolve to commit `4752ffab487ebd387781ad7890aa540ae2780970`.
- The effective comparison is therefore the current working-tree delta on this branch, not a commit-ahead/behind diff.
- The branch adds a Towers of Hanoi benchmark/reporting stack, a persistent SmolVLM worker, helper utilities for symbolic/legal move extraction, and several baseline plumbing changes in training, testing, and agent code.
- A follow-up implementation on this branch adds a hosted Hugging Face VLM path, bounded scene-history context, and a guarded symbolic execution policy that guarantees full-episode completion without retraining.
- Generated artifacts were intentionally excluded from the detailed implementation diff: dataset episode blobs, `.pkl` outputs, images, rollouts, checkpoints, logs, and report figures.
- The SmolVLM/Hanoi execution bug was real and reproducible: after the first two successful moves, the legal-candidate path stalled because it executed Hanoi moves with direct pixel-to-world pick/place poses instead of object-relative pick/place geometry.
- The post-approval fix removed that stall, fixed SmolVLM preprocessing/parsing failures, and made the real worker return parseable selections again.
- Residual limitation: on the smoke seed and current 1000-step transporter checkpoint, `HuggingFaceTB/SmolVLM-256M-Instruct` still chooses candidate `0` throughout the episode, so the run progresses but does not fully solve the smoke episode.

## Comparison Method and Scope

- Baseline: `master` at `4752ffab487ebd387781ad7890aa540ae2780970`.
- Compared material:
  - tracked source/config/doc edits already in the repo
  - new source files under `ravens/`
  - new documentation under `docs/`
  - new `requirements-smolvlm.txt`
- Excluded from the detailed narrative:
  - `checkpoints/`
  - `logs/`
  - `hanoi-results*/`
  - rollout videos and frames
  - dataset episode payloads under `towers-of-hanoi-*` and similar task folders
  - generated `.png`, `.mp4`, and top-level `.pkl` byproducts
- Existing byproducts such as `hanoi-results*`, `checkpoints/`, and `logs/` are treated as branch output, not implementation.

## Changes vs `master` by Subsystem

### Agent and Training/Test Plumbing

- `ravens/agents/__init__.py`
  - adds a `transporter_3d` alias that points at `Transporter6dAgent`
- `ravens/agents/gt_state.py`
  - disables eager visualizer creation
  - fixes quaternion/matrix handling during augmentation
  - reduces observation-stat sampling cost for small datasets
- `ravens/agents/transporter.py`
  - adds default `root_dir='.'`
  - preserves compatibility with older call sites that pass only a rotation count
- `ravens/agents/transporter_6dof.py`
  - large rewrite from the older hybrid implementation to a cleaner attention + planar transport + 6-DoF regression stack
  - adds explicit `save()` / `load()` behavior and simplified train/validate/act flow
- `ravens/models/transport_6dof.py`
  - adds save/load support for the transport head plus the `z`, `roll`, and `pitch` regressors
- `ravens/tasks/task.py`
  - enables full SE(3) pose matching for tasks that explicitly set `self.sixdof = True`
- `ravens/demos.py`
  - resumes dataset collection instead of blindly recreating already-complete splits
- `ravens/train.py`
  - changes the default task from `hanoi` to `towers-of-hanoi`
- `ravens/test.py`
  - changes the default task to `towers-of-hanoi`
  - makes legacy top-level `.pkl` output optional via `--save_results_pickle`
- `ravens/plot.py`
  - only formatting-level change

### New Hanoi Benchmarking Stack

- `ravens/hanoi_benchmark.py`
  - adds a benchmark runner for `transporter`, `vlm-transporter`, `gt_state`, and `oracle`
  - supports legal symbolic candidate extraction, VLM reranking, request image generation, rollout capture, JSON result export, and GT-state checkpointing
- `ravens/hanoi_utils.py`
  - adds Hanoi symbolic state helpers, candidate enumeration/scoring, request-board rendering, VLM worker client, rollout recording, and summary aggregation
- `ravens/hanoi_vlm_worker.py`
  - adds a persistent line-delimited JSON subprocess worker
  - supports `heuristic` and `smolvlm` backends
- `ravens/hanoi_report.py`
  - aggregates benchmark JSON into CSV/JSON summaries and metric plots
- `ravens/plot_new.py`
  - standalone plotting script for block insertion accuracy

### New Docs and Dependencies

- `requirements-smolvlm.txt`
  - adds a separate PyTorch/Transformers environment for SmolVLM
- `docs/hanoi_vlm.md`
  - original Hanoi benchmark/runbook
- `docs/hanoi_vlm_improved.md`
  - improved runbook centered on legal symbolic Hanoi candidates
- `docs/run_tasks.md`
  - operator-oriented command summary for the new task stack
- `FAILURE_ANALYSIS.md`
  - pre-existing local notes about worker failure modes and benchmark robustness gaps

## Hanoi / SmolVLM Failure Analysis

### Root Cause 1: Legal Candidate Execution Geometry

- The legal Hanoi pipeline was scoring symbolic legal moves correctly, but it executed those moves by feeding the candidate `pick_pixel` and `place_pixel` directly through `pix_to_xyz`.
- That shortcut is not equivalent to the Ravens oracle:
  - the disk pick point must be a valid suction point on the visible ring surface
  - the place pose must preserve the live pick-to-object transform, not simply place the gripper at the peg center
- Result:
  - the first two moves often worked
  - the third symbolic move was still marked as legal/oracle-matching in diagnostics
  - the environment reward stopped increasing because the physical action did not place the disk into the goal pose expected by the task

### Root Cause 2: SmolVLM Backend Fragility

- The real SmolVLM worker initially failed before generation with:
  - ``ValueError: `resolution_max_side` cannot be larger than `max_image_size```
- Cause:
  - the worker pre-resized images but still let the processor trigger its incompatible internal resize path

### Root Cause 3: Prompt-Echo / Parser Mismatch

- Even after the preprocessing fix, the 256M model often emitted malformed or partial JSON copied from the prompt structure.
- The original parser was biased toward the first candidate-like token it found and did not robustly handle:
  - prompt echo
  - quoted numeric fields
  - partial JSON blocks
  - full-sequence decoding where the prompt itself was included in the decoded text

## Fixes Implemented After Approval

### SmolVLM Worker

- Updated `ravens/hanoi_vlm_worker.py` to:
  - decode generated suffix tokens when the backend returns prompt + completion in one sequence
  - skip the processor resize path with `do_resize=False` after the worker has already normalized the image
  - prefer the last valid JSON block or last valid candidate reference when parsing
  - accept quoted numeric `candidate_index` values
  - preserve raw model output in the worker response for debugging/reporting
  - simplify the VLM-facing prompt so the tiny model is less likely to regurgitate exemplar structure

### Legal Hanoi Execution Path

- Updated `ravens/hanoi_utils.py` to add a dedicated Hanoi execution helper that:
  - derives a valid suction pick point from the selected disk’s live segmentation mask via distance transform
  - uses the disk’s live object pose from PyBullet
  - targets the chosen peg pose in world coordinates
  - reconstructs the place pose using the same pick-to-object transform logic as the Ravens oracle
- Updated `ravens/hanoi_benchmark.py` so `vlm-transporter` in `candidate_mode=legal` uses that helper instead of direct pixel-to-world action synthesis.

### Benchmark Robustness

- Updated `ravens/hanoi_utils.py` `VLMWorkerClient` to:
  - poll worker stdout with `select.select(...)`
  - honor request timeouts without blocking forever on `readline()`
  - terminate/kill cleanly on shutdown when needed
- Updated `ravens/hanoi_benchmark.py` to:
  - catch worker `RuntimeError` / `TimeoutError` at step time
  - close the dead worker and allow later lazy restart
  - record worker errors in step diagnostics
  - fall back to the highest-scoring legal move for that step instead of aborting the entire episode

### VLM Request Presentation

- Removed score text from the request-board legend shown to the VLM path.
- Left transporter scores intact internally for candidate scoring, fallback ordering, and non-VLM diagnostics.

## Validation Results and Before / After Behavior

### Exact Commands Run

Parser regression check:

```bash
PYTHONPATH=/home/praharsh/perception/ravens \
/home/praharsh/perception/perception_env/bin/python - <<'PY'
from ravens.hanoi_vlm_worker import extract_json_robust
text = '''Available numbered moves:
[0] move a
[1] move b
[2] move c
Assistant: I choose [2] because it clears the smaller disk.'''
print(extract_json_robust(text, 3))
PY
```

Execution geometry check:

```bash
PYTHONPATH=/home/praharsh/perception/ravens \
/home/praharsh/perception/perception_env/bin/python -u - <<'PY'
import numpy as np
from ravens import tasks
from ravens.dataset import Dataset
from ravens.environments.environment import Environment
from ravens.hanoi_utils import build_hanoi_execution_action, enumerate_legal_hanoi_moves, get_oracle_move

assets_root='./ravens/environments/assets'
task_name='towers-of-hanoi'
ds = Dataset(f'./{task_name}-test')
_, seed = ds.load(0, images=False)
env = Environment(assets_root, disp=False, shared_memory=False, hz=480)
task = tasks.names[task_name]()
task.mode='test'
env.seed(seed)
np.random.seed(seed)
env.set_task(task)
obs = env.reset()
for step in range(task.max_steps):
  oracle = get_oracle_move(env)
  candidates = enumerate_legal_hanoi_moves(env)
  selected = next(c for c in candidates
                  if c.disk_id == oracle['disk_id']
                  and c.target_peg == oracle['target_peg'])
  action = build_hanoi_execution_action(env, selected)
  obs, reward, done, _ = env.step(action)
  print(f'step={step + 1} reward={reward:.6f} done={done}', flush=True)
  if done:
    break
env.close()
PY
```

Heuristic smoke benchmark:

```bash
PYTHONPATH=/home/praharsh/perception/ravens \
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
  --output_dir=/tmp/codex-hanoi-heuristic-smoke
```

Real SmolVLM smoke benchmark:

```bash
PYTHONPATH=/home/praharsh/perception/ravens \
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
  --vlm_timeout_s=300 \
  --vlm_command="/home/praharsh/perception/smolvlm_env/bin/python ravens/hanoi_vlm_worker.py --backend=smolvlm --device=cpu --model_id=HuggingFaceTB/SmolVLM-256M-Instruct" \
  --output_dir=/tmp/codex-hanoi-real-smoke-v3
```

### Observed Results

- Parser regression:
  - returned `{'candidate_index': 2, 'valid': True, ...}`
  - confirms the parser no longer grabs the first prompt-listed `[0]`
- Execution geometry:
  - reached reward `1.0` in 7 moves on smoke seed `1`
  - confirms the physical-action bug was the direct cause of the old two-step stall
- Heuristic smoke:
  - `mean_reward = 0.42857142857142855`
  - `legal_move_rate = 1.0`
  - `vlm_fallback_rate = 0.0`
  - progressed through the full 14-step episode instead of stalling after step 2
- Real SmolVLM smoke, intermediate behavior:
  - preprocessor bug run: `vlm_fallback_rate = 1.0`, `vlm_parse_success_rate = 0.0`
  - parser-imperfect run: `vlm_fallback_rate = 1.0`, `vlm_parse_success_rate = 0.0`
- Real SmolVLM smoke, final run (`/tmp/codex-hanoi-real-smoke-v3/...json`):
  - `mean_reward = 0.42857142857142855`
  - `success_rate = 0.0`
  - `legal_move_rate = 1.0`
  - `vlm_fallback_rate = 0.0`
  - `vlm_parse_success_rate = 1.0`
  - `vlm_invalid_response_rate = 0.0`
  - the worker now returns parseable SmolVLM responses rather than crashing or falling back
  - residual limitation: the 256M model still chose candidate `0` throughout this smoke episode, so the run remained legal and stable but did not solve the full task

### Before / After Summary

- Before:
  - the legal Hanoi path often stopped making reward progress after the first two successful moves
  - the real SmolVLM worker either failed preprocessing or produced malformed outputs that forced fallback
- After:
  - the legal Hanoi path executes physically valid object-relative actions and can complete the full 7-step oracle sequence
  - the benchmark survives worker failures/timeouts instead of aborting the episode
  - the real SmolVLM worker now reaches `parse_success_rate = 1.0` on the smoke run
  - remaining issue is model-choice quality on this checkpoint, not the earlier execution/preprocessing/parser bugs

## Residual Limitations

- The implementation fixes the mechanical stall and the worker/parse failures.
- It does **not** make `HuggingFaceTB/SmolVLM-256M-Instruct` solve the smoke episode on the current `towers-of-hanoi-transporter-10-0` 1000-step checkpoint.
- No full retraining was run during this implementation pass.

## Full-Episode Completion Follow-Up

This section covers the later follow-up work requested after the initial stall/parsing fix. The goal was to analyze stronger free/open VLM options, evaluate short visual history, and make the branch capable of finishing a full Towers of Hanoi episode without retraining.

The follow-up validation below supersedes the earlier single-image SmolVLM smoke numbers for the current branch state because the worker prompt, benchmark request shape, and execution policy now differ.

### Analysis of the Proposed Changes

- Using a different VLM is the highest-leverage change.
  - After the original mechanical/action-execution fix, the remaining failure mode was model-choice quality, not action synthesis.
  - A stronger open-source VLM is therefore more promising than further prompt tuning on `HuggingFaceTB/SmolVLM-256M-Instruct`.
- Passing the last 2-3 images as context can be reasonable, but only if those images are raw scene states.
  - Reusing old numbered candidate boards would introduce stale numbering and stale arrows.
  - The implemented version therefore keeps the current numbered board as the primary decision image and passes only prior unnumbered top-down scene snapshots as history.
- “Keeping the VLM in the same thread” was not the best primary design.
  - The worker process was already persistent across the episode.
  - The missing piece was explicit bounded episode context, not hidden conversational state.
  - The implemented design keeps requests self-contained and reproducible by rebuilding the prompt from bounded textual state plus optional recent images.
- The most reliable alternative is a symbolic guardrail.
  - Towers of Hanoi is fully symbolic and deterministic.
  - A planner-guarded execution policy can guarantee full completion even when the VLM is weak, slow, or malformed.

### Follow-Up Changes Implemented

- `ravens/hanoi_vlm_worker.py`
  - added `hf-chat` as a hosted backend using Hugging Face routed chat completions
  - added environment-driven model resolution via `HANOI_VLM_MODEL`
  - added `HF_TOKEN`-based hosted authentication
  - added bounded multi-image support:
    - first image = current numbered candidate board
    - later images = prior raw scene snapshots
  - kept the response contract unchanged (`candidate_index`, `valid`, `parse_success`, `rationale`, `backend`, raw debug text)
- `ravens/hanoi_benchmark.py`
  - added `--decision_policy=guarded|vlm-only`
  - added `--history_images`
  - saved raw scene-history images separately from numbered request boards
  - tracked the VLM-selected move separately from the actually executed move
  - added planner override, repeat-state, history-image, and backend diagnostics
  - made result filenames include candidate mode / decision policy / history window to avoid JSON overwrite across comparable runs
- `ravens/hanoi_utils.py`
  - added `render_scene_image(...)` for unnumbered history images
  - added `find_symbolic_move(...)` for matching oracle/planner moves to legal candidates
  - added summary metrics for planner overrides, executed oracle agreement, repeat-state rate, and mean history images used
- `ravens/hanoi_report.py`
  - added plots and summary fields for the new guarded-policy metrics
  - updated run grouping so different VLM policies/history windows do not collapse into one curve

### Hugging Face Setup Path

Recommended hosted open-source models for this branch:

- `Qwen/Qwen2.5-VL-7B-Instruct`
- `Qwen/Qwen3.6-35B-A3B`

Required setup:

1. Create a Hugging Face fine-grained token.
2. Grant the token the `Inference Providers` permission (`Make calls to Inference Providers`).
3. Export:

```bash
export HF_TOKEN=...
export HANOI_VLM_MODEL=Qwen/Qwen3.6-35B-A3B
```

4. Run the benchmark through the hosted backend:

```bash
PYTHONPATH=/home/praharsh/perception/ravens \
/home/praharsh/perception/perception_env/bin/python ravens/hanoi_benchmark.py \
  --assets_root=./ravens/environments/assets \
  --data_dir=. \
  --root_dir=. \
  --task=towers-of-hanoi \
  --agent=vlm-transporter \
  --candidate_mode=legal \
  --decision_policy=guarded \
  --history_images=2 \
  --n_demos=10 \
  --n_steps=1000 \
  --episodes=1 \
  --vlm_timeout_s=300 \
  --vlm_command="/home/praharsh/perception/perception_env/bin/python ravens/hanoi_vlm_worker.py --backend=hf-chat --model_id=Qwen/Qwen3.6-35B-A3B" \
  --output_dir=/tmp/codex-hanoi-hf-guarded
```

Notes:

- Hugging Face routed inference uses open-source hosted models, but the free credits are limited; it is not an unlimited free hosted service.
- Ollama remains the no-key local alternative, but it was not implemented in this pass because this machine is CPU-only and larger local vision models would be too slow to be the default.

### Additional Validation Commands Run

Hosted HF backend connectivity check:

```bash
HF_TOKEN=*** \
PYTHONPATH=/home/praharsh/perception/ravens \
/home/praharsh/perception/perception_env/bin/python - <<'PY'
import json
import os
import tempfile
from PIL import Image
from ravens.hanoi_vlm_worker import HFChatBackend

with tempfile.TemporaryDirectory() as tmpdir:
  current = os.path.join(tmpdir, 'current.png')
  history = os.path.join(tmpdir, 'history.png')
  Image.new('RGB', (16, 16), color='red').save(current)
  Image.new('RGB', (16, 16), color='blue').save(history)
  backend = HFChatBackend('Qwen/Qwen2.5-VL-7B-Instruct', 64)
  result = backend.select({
      'image_path': current,
      'current_request_image_path': current,
      'history_image_paths': [history],
      'peg_state': 'left peg: large green disk, medium yellow disk\nmiddle peg: small red disk\nright peg: empty',
      'move_history': [{'description': 'small red disk: left peg -> middle peg (legal)', 'reward': 0.0}],
      'prompt_examples': [],
      'candidates': [
          {'index': 0, 'description': 'small red disk: middle peg -> right peg (legal)'},
          {'index': 1, 'description': 'medium yellow disk: left peg -> right peg (legal)'},
      ],
  })
  print(json.dumps(result, indent=2))
PY
```

Guarded heuristic benchmark:

```bash
PYTHONPATH=/home/praharsh/perception/ravens \
/home/praharsh/perception/perception_env/bin/python ravens/hanoi_benchmark.py \
  --assets_root=./ravens/environments/assets \
  --data_dir=. \
  --root_dir=. \
  --task=towers-of-hanoi \
  --agent=vlm-transporter \
  --candidate_mode=legal \
  --decision_policy=guarded \
  --history_images=2 \
  --n_demos=10 \
  --n_steps=1000 \
  --episodes=1 \
  --vlm_command="/home/praharsh/perception/perception_env/bin/python ravens/hanoi_vlm_worker.py --backend=heuristic" \
  --output_dir=/tmp/codex-hanoi-heuristic-guarded
```

SmolVLM `vlm-only` without history:

```bash
PYTHONPATH=/home/praharsh/perception/ravens \
/home/praharsh/perception/perception_env/bin/python ravens/hanoi_benchmark.py \
  --assets_root=./ravens/environments/assets \
  --data_dir=. \
  --root_dir=. \
  --task=towers-of-hanoi \
  --agent=vlm-transporter \
  --candidate_mode=legal \
  --decision_policy=vlm-only \
  --history_images=0 \
  --n_demos=10 \
  --n_steps=1000 \
  --episodes=1 \
  --vlm_timeout_s=300 \
  --vlm_command="/home/praharsh/perception/smolvlm_env/bin/python ravens/hanoi_vlm_worker.py --backend=smolvlm --device=cpu --model_id=HuggingFaceTB/SmolVLM-256M-Instruct" \
  --output_dir=/tmp/codex-hanoi-smolvlm-h0
```

SmolVLM `vlm-only` with bounded raw-image history:

```bash
PYTHONPATH=/home/praharsh/perception/ravens \
/home/praharsh/perception/perception_env/bin/python ravens/hanoi_benchmark.py \
  --assets_root=./ravens/environments/assets \
  --data_dir=. \
  --root_dir=. \
  --task=towers-of-hanoi \
  --agent=vlm-transporter \
  --candidate_mode=legal \
  --decision_policy=vlm-only \
  --history_images=2 \
  --n_demos=10 \
  --n_steps=1000 \
  --episodes=1 \
  --vlm_timeout_s=300 \
  --vlm_command="/home/praharsh/perception/smolvlm_env/bin/python ravens/hanoi_vlm_worker.py --backend=smolvlm --device=cpu --model_id=HuggingFaceTB/SmolVLM-256M-Instruct" \
  --output_dir=/tmp/codex-hanoi-smolvlm-h2
```

Guarded SmolVLM benchmark:

```bash
PYTHONPATH=/home/praharsh/perception/ravens \
/home/praharsh/perception/perception_env/bin/python -u ravens/hanoi_benchmark.py \
  --assets_root=./ravens/environments/assets \
  --data_dir=. \
  --root_dir=. \
  --task=towers-of-hanoi \
  --agent=vlm-transporter \
  --candidate_mode=legal \
  --decision_policy=guarded \
  --history_images=2 \
  --n_demos=10 \
  --n_steps=1000 \
  --episodes=1 \
  --vlm_timeout_s=300 \
  --vlm_command="/home/praharsh/perception/smolvlm_env/bin/python ravens/hanoi_vlm_worker.py --backend=smolvlm --device=cpu --model_id=HuggingFaceTB/SmolVLM-256M-Instruct --max_new_tokens=48" \
  --output_dir=/tmp/codex-hanoi-smolvlm-guarded
```

### Follow-Up Validation Results

- Hosted HF backend, live router check:
  - request reached `router.huggingface.co`
  - failed with `HTTP 403`
  - exact failure: the provided token did not have sufficient permission to call Inference Providers
  - conclusion: the code path works, but the token/account configuration was incomplete
- Guarded heuristic benchmark:
  - `success_rate = 1.0`
  - `mean_reward = 1.0`
  - `mean_steps = 7`
  - `planner_override_rate = 0.2857`
  - `oracle_move_agreement = 0.7143`
  - `executed_oracle_move_agreement = 1.0`
  - `repeat_state_rate = 0.0`
- SmolVLM `vlm-only`, `history_images=0`:
  - `success_rate = 0.0`
  - `mean_reward = 0.4286`
  - `oracle_move_agreement = 0.2143`
  - `vlm_parse_success_rate = 0.4286`
  - `vlm_fallback_rate = 0.5714`
  - `repeat_state_rate = 0.7857`
  - `mean_total_latency_s = 20.05`
- SmolVLM `vlm-only`, `history_images=2`:
  - `success_rate = 0.0`
  - `mean_reward = 0.4286`
  - `oracle_move_agreement = 0.2143`
  - `vlm_parse_success_rate = 0.5000`
  - `vlm_fallback_rate = 0.5000`
  - `repeat_state_rate = 0.7857`
  - `mean_total_latency_s = 24.47`
  - interpretation: bounded raw-image history did not materially improve the local 256M model on this smoke seed; it only increased latency
- SmolVLM `guarded`, `history_images=2`:
  - `success_rate = 1.0`
  - `mean_reward = 1.0`
  - `mean_steps = 7`
  - `oracle_move_agreement = 0.7143`
  - `vlm_parse_success_rate = 0.5714`
  - `vlm_fallback_rate = 0.4286`
  - `planner_override_rate = 0.2857`
  - `executed_oracle_move_agreement = 1.0`
  - `repeat_state_rate = 0.0`
  - interpretation: the guarded policy solves the full episode even though local SmolVLM remains weak and partially malformed

### Recommendation After the Follow-Up

- Best immediate path to reliable full-episode completion:
  - use `--decision_policy=guarded`
  - keep `--candidate_mode=legal`
  - keep `--history_images=2`
  - switch to a stronger hosted open-source VLM once the Hugging Face token has Inference Providers permission
- Best hosted model candidates for this branch:
  - `Qwen/Qwen2.5-VL-7B-Instruct`
  - `Qwen/Qwen3.6-35B-A3B`
- What did **not** work as a standalone fix:
  - adding only the last 2 raw scene images did not materially improve the local `HuggingFaceTB/SmolVLM-256M-Instruct` policy
- Why hidden multi-turn state was not chosen:
  - the persistent worker already keeps the process alive
  - explicit bounded context is easier to reproduce, debug, and compare across runs
