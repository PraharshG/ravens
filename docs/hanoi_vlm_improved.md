# Improved Hanoi VLM-over-Transporter Workflow

This runbook uses:

- the Ravens TensorFlow environment for demos, benchmark, and reporting
- a separate Python environment for SmolVLM
- oracle-generated demonstrations for Transporter training
- legal symbolic Hanoi move candidates scored by the trained Transporter

## 1. Activate the Ravens Environment

```bash
source /home/praharsh/perception/perception_env/bin/activate
cd /home/praharsh/perception/ravens
export PYTHONPATH=${PWD}
```

## 2. Generate Oracle Demonstrations

Ravens already uses the task oracle for demonstrations. These are the correct Hanoi demos to train the Transporter baseline.

```bash
python ravens/demos.py --assets_root=./ravens/environments/assets --task=towers-of-hanoi --mode=train --n=100
python ravens/demos.py --assets_root=./ravens/environments/assets --task=towers-of-hanoi --mode=test --n=100
```

## 3. Optional: Train or Reuse Transporter Checkpoints

If you already have Hanoi checkpoints in `./checkpoints`, you can skip this section.

```bash
python ravens/train.py --task=towers-of-hanoi --agent=transporter --n_demos=10 --n_steps=1000
python ravens/train.py --task=towers-of-hanoi --agent=transporter --n_demos=20 --n_steps=1000
python ravens/train.py --task=towers-of-hanoi --agent=transporter --n_demos=100 --n_steps=1000
```

## 4. Create the Separate SmolVLM Environment

```bash
python3 -m venv /home/praharsh/perception/smolvlm_env
source /home/praharsh/perception/smolvlm_env/bin/activate
pip install -U pip
pip install -r /home/praharsh/perception/ravens/requirements-smolvlm.txt
```

## 5. Quick Sanity Check with the Heuristic Worker

This uses the improved legal-symbolic candidate pipeline without requiring Hugging Face downloads.

```bash
source /home/praharsh/perception/perception_env/bin/activate
cd /home/praharsh/perception/ravens
export PYTHONPATH=${PWD}

python ravens/hanoi_benchmark.py \
  --assets_root=./ravens/environments/assets \
  --data_dir=. \
  --root_dir=. \
  --task=towers-of-hanoi \
  --agent=vlm-transporter \
  --candidate_mode=legal \
  --n_demos=10 \
  --n_steps=1000 \
  --episodes=5 \
  --vlm_command="/home/praharsh/perception/perception_env/bin/python ravens/hanoi_vlm_worker.py --backend=heuristic" \
  --record_rollout \
  --record_episode_index=0 \
  --output_dir=./hanoi-results-legal-smoke
```

Expected artifacts:

- `./hanoi-results-legal-smoke/*.json`
- `./hanoi-results-legal-smoke/vlm-requests/*.png`
- `./hanoi-results-legal-smoke/rollouts/episode-000.mp4`

## 6. Run the Real SmolVLM Worker

```bash
source /home/praharsh/perception/smolvlm_env/bin/activate
cd /home/praharsh/perception/ravens
python ravens/hanoi_vlm_worker.py --backend=smolvlm --model_id=HuggingFaceTB/SmolVLM-256M-Instruct
```

You do not need to keep this in a separate terminal if you pass the same command through `--vlm_command` in the benchmark below.

## 7. Run the Main Benchmark

### Vanilla Transporter Baseline

```bash
source /home/praharsh/perception/perception_env/bin/activate
cd /home/praharsh/perception/ravens
export PYTHONPATH=${PWD}

python ravens/hanoi_benchmark.py \
  --assets_root=./ravens/environments/assets \
  --data_dir=. \
  --root_dir=. \
  --task=towers-of-hanoi \
  --agent=transporter \
  --n_demos=10 \
  --n_steps=1000 \
  --episodes=100 \
  --output_dir=./hanoi-results-legal
```

### Improved VLM-Transporter

```bash
source /home/praharsh/perception/perception_env/bin/activate
cd /home/praharsh/perception/ravens
export PYTHONPATH=${PWD}

python ravens/hanoi_benchmark.py \
  --assets_root=./ravens/environments/assets \
  --data_dir=. \
  --root_dir=. \
  --task=towers-of-hanoi \
  --agent=vlm-transporter \
  --candidate_mode=legal \
  --pick_radius=4 \
  --place_radius=6 \
  --num_prompt_exemplars=3 \
  --n_demos=10 \
  --n_steps=1000 \
  --episodes=100 \
  --vlm_command="/home/praharsh/perception/smolvlm_env/bin/python ravens/hanoi_vlm_worker.py --backend=smolvlm --model_id=HuggingFaceTB/SmolVLM-256M-Instruct" \
  --record_rollout \
  --record_episode_index=0 \
  --output_dir=./hanoi-results-legal
```

### Oracle Upper Bound

```bash
python ravens/hanoi_benchmark.py \
  --assets_root=./ravens/environments/assets \
  --data_dir=. \
  --root_dir=. \
  --task=towers-of-hanoi \
  --agent=oracle \
  --n_steps=0 \
  --episodes=1 \
  --record_rollout \
  --record_episode_index=0 \
  --output_dir=./hanoi-results-legal
```

## 8. Generate the Report

```bash
source /home/praharsh/perception/perception_env/bin/activate
cd /home/praharsh/perception/ravens
python ravens/hanoi_report.py --input_dir=./hanoi-results-legal --output_dir=./hanoi-results-legal/report
```

Expected report artifacts:

- `summary.csv`
- `summary.json`
- `success_vs_steps.png`
- `legal_move_rate.png`
- `oracle_move_agreement.png`
- `steps_to_success.png`
- `latency.png`
- `legal_candidate_coverage.png`
- `vlm_parse_success_rate.png`
- `vlm_invalid_response_rate.png`
- `vlm_fallback_rate.png`

## 9. How to Read the Results

- `success_rate`: fraction of episodes solved completely.
- `legal_move_rate`: fraction of executed moves that obey Hanoi rules.
- `oracle_move_agreement`: fraction of steps where the chosen move matches the oracle’s next move.
- `legal_candidate_coverage`: fraction of VLM steps where the oracle move was present in the candidate list.
  - In the improved legal-symbolic pipeline this should be near `1.0`.
- `vlm_parse_success_rate`: fraction of VLM calls that returned a valid parsed candidate index.
- `vlm_invalid_response_rate`: fraction of VLM calls marked invalid.
- `vlm_fallback_rate`: fraction of VLM steps that fell back to the highest-scoring legal move.

The most important sanity checks are:

1. `legal_candidate_coverage` should be close to `1.0`.
2. `vlm_parse_success_rate` should be high.
3. `vlm_fallback_rate` should be much lower than the earlier broken setup.
4. `vlm-transporter` should not regress badly against plain `transporter` on `legal_move_rate`.

## 10. Visualization

To inspect a rollout:

- open `./hanoi-results-legal/rollouts/episode-000.mp4`
- open `./hanoi-results-legal/vlm-requests/`

The rollout video shows:

- the standard PyBullet simulation on the left
- the numbered legal candidate board on the right
- the selected move, oracle reference, and VLM rationale in the overlay text

The request images in `vlm-requests/` are exactly what the VLM receives for each decision.

## 11. Legacy `.pkl` Note

`ravens/test.py` no longer writes top-level legacy result `.pkl` files by default.

If you explicitly need the old behavior:

```bash
python ravens/test.py --task=towers-of-hanoi --agent=transporter --n_demos=10 --n_steps=1000 --save_results_pickle
```

Do not delete the `.pkl` files inside dataset folders such as `towers-of-hanoi-train/` or `towers-of-hanoi-test/`; those are the actual Ravens dataset format.
