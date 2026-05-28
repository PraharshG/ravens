# Hanoi VLM Benchmark

## Current Environment

Use the Ravens TensorFlow environment for demos, transporter training, evaluation, and reporting:

```bash
source /home/praharsh/perception/perception_env/bin/activate
cd /home/praharsh/perception/ravens
export PYTHONPATH=${PWD}
```

## 1. Generate Hanoi Demos

```bash
python ravens/demos.py --assets_root=./ravens/environments/assets --task=towers-of-hanoi --mode=train --n=100
python ravens/demos.py --assets_root=./ravens/environments/assets --task=towers-of-hanoi --mode=test --n=100
```

## 2. Train the Vanilla Transporter

Run one command per demo count:

```bash
python ravens/train.py --task=towers-of-hanoi --agent=transporter --n_demos=10 --n_steps=1000
python ravens/train.py --task=towers-of-hanoi --agent=transporter --n_demos=20 --n_steps=1000
python ravens/train.py --task=towers-of-hanoi --agent=transporter --n_demos=100 --n_steps=1000
```

## 3. Create the Separate SmolVLM Environment

```bash
python3 -m venv /home/praharsh/perception/smolvlm_env
source /home/praharsh/perception/smolvlm_env/bin/activate
pip install -U pip
pip install -r /home/praharsh/perception/ravens/requirements-smolvlm.txt
```

## 4. Start the Persistent VLM Worker

Real SmolVLM worker:

```bash
source /home/praharsh/perception/smolvlm_env/bin/activate
cd /home/praharsh/perception/ravens
python ravens/hanoi_vlm_worker.py --backend=smolvlm --model_id=HuggingFaceTB/SmolVLM-256M-Instruct
```

Fast smoke-test worker:

```bash
source /home/praharsh/perception/perception_env/bin/activate
cd /home/praharsh/perception/ravens
python ravens/hanoi_vlm_worker.py --backend=heuristic
```

## 5. Evaluate Baselines

Vanilla transporter:

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
  --output_dir=./hanoi-results
```

VLM-guided transporter:

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
  --n_demos=10 \
  --n_steps=1000 \
  --episodes=100 \
  --top_pick_k=3 \
  --top_place_k=3 \
  --max_candidates=6 \
  --vlm_command="/home/praharsh/perception/smolvlm_env/bin/python ravens/hanoi_vlm_worker.py --backend=smolvlm --model_id=HuggingFaceTB/SmolVLM-256M-Instruct" \
  --output_dir=./hanoi-results
```

Ground-truth-state baseline:

```bash
source /home/praharsh/perception/perception_env/bin/activate
cd /home/praharsh/perception/ravens
export PYTHONPATH=${PWD}
python ravens/hanoi_benchmark.py \
  --assets_root=./ravens/environments/assets \
  --data_dir=. \
  --root_dir=. \
  --task=towers-of-hanoi \
  --agent=gt_state \
  --n_demos=10 \
  --n_steps=1000 \
  --train_gt_state \
  --gt_state_train_steps=5000 \
  --gt_state_interval=1000 \
  --episodes=100 \
  --output_dir=./hanoi-results
```

Oracle reference and rollout capture:

```bash
source /home/praharsh/perception/perception_env/bin/activate
cd /home/praharsh/perception/ravens
export PYTHONPATH=${PWD}
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
  --output_dir=./hanoi-results
```

## 6. Generate Comparison Graphs

```bash
source /home/praharsh/perception/perception_env/bin/activate
cd /home/praharsh/perception/ravens
python ravens/hanoi_report.py --input_dir=./hanoi-results --output_dir=./hanoi-results/report
```

This produces:

- `summary.csv`
- `summary.json`
- `success_vs_steps.png`
- `legal_move_rate.png`
- `oracle_move_agreement.png`
- `steps_to_success.png`
- `latency.png`
