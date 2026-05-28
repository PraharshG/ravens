# Ravens Hanoi Benchmark Pipeline (Transporter + VLM + GT State)

This document provides a sequential execution plan for training, benchmarking, and reporting.

---

## STEP 1: Train Transporter

source /home/praharsh/perception/perception_env/bin/activate
cd /home/praharsh/perception/ravens
export PYTHONPATH=${PWD}

python ravens/train.py --task=towers-of-hanoi --agent=transporter --n_demos=10 --n_steps=1000
python ravens/train.py --task=towers-of-hanoi --agent=transporter --n_demos=20 --n_steps=1000
python ravens/train.py --task=towers-of-hanoi --agent=transporter --n_demos=100 --n_steps=1000

---

## STEP 2: Setup SmolVLM

python3 -m venv /home/praharsh/perception/smolvlm_env
source /home/praharsh/perception/smolvlm_env/bin/activate
pip install -U pip
pip install -r /home/praharsh/perception/ravens/requirements-smolvlm.txt

---

## STEP 3: Benchmark

source /home/praharsh/perception/perception_env/bin/activate
cd /home/praharsh/perception/ravens
export PYTHONPATH=${PWD}

### Transporter
python ravens/hanoi_benchmark.py --assets_root=./ravens/environments/assets --data_dir=. --root_dir=. --task=towers-of-hanoi --agent=transporter --n_demos=10 --n_steps=1000 --episodes=100 --output_dir=./hanoi-results

### VLM Transporter
python ravens/hanoi_benchmark.py --assets_root=./ravens/environments/assets --data_dir=. --root_dir=. --task=towers-of-hanoi --agent=vlm-transporter --n_demos=10 --n_steps=1000 --episodes=100 --top_pick_k=3 --top_place_k=3 --max_candidates=6 --vlm_command="/home/praharsh/perception/smolvlm_env/bin/python ravens/hanoi_vlm_worker.py --backend=smolvlm --model_id=HuggingFaceTB/SmolVLM-256M-Instruct" --output_dir=./hanoi-results

### GT State
python ravens/hanoi_benchmark.py --assets_root=./ravens/environments/assets --data_dir=. --root_dir=. --task=towers-of-hanoi --agent=gt_state --n_demos=10 --n_steps=1000 --train_gt_state --gt_state_train_steps=1000 --gt_state_interval=1000 --episodes=100 --output_dir=./hanoi-results

---

## STEP 4: Report

python ravens/hanoi_report.py --input_dir=./hanoi-results --output_dir=./hanoi-results/report

---

## Outputs

- success_vs_steps.png
- legal_move_rate.png
- oracle_move_agreement.png
- steps_to_success.png
- latency.png
- summary.csv
- summary.json