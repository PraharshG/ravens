# coding=utf-8
"""Select the best existing Hanoi transporter checkpoint on oracle states."""

from __future__ import annotations

import argparse
import json
import os

from ravens.hanoi_reranker import discover_transporter_checkpoints
from ravens.hanoi_reranker import evaluate_transporter_checkpoint
from ravens.hanoi_reranker import select_best_checkpoint
from ravens.hanoi_utils import ensure_dir
from ravens.hanoi_utils import json_ready


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--root_dir', default='.')
  parser.add_argument('--data_dir', default='.')
  parser.add_argument('--assets_root', default='./ravens/environments/assets')
  parser.add_argument('--task', default='towers-of-hanoi')
  parser.add_argument('--split', default='test', choices=['train', 'test'])
  parser.add_argument('--episodes', type=int, default=5)
  parser.add_argument('--pick_radius', type=int, default=4)
  parser.add_argument('--place_radius', type=int, default=6)
  parser.add_argument('--output_path',
                      default='./hanoi-reranker-data/checkpoint-selection.json')
  return parser.parse_args()


def main():
  args = parse_args()
  checkpoints = discover_transporter_checkpoints(args.root_dir, args.task)
  if not checkpoints:
    raise ValueError(
        f'No transporter checkpoints found under {args.root_dir}/checkpoints '
        f'for task {args.task}.')

  results = []
  for checkpoint in checkpoints:
    print(
        f'Evaluating {checkpoint["name"]} step {checkpoint["step"]} on '
        f'{args.split} split ({args.episodes} episodes)...')
    result = evaluate_transporter_checkpoint(
        root_dir=args.root_dir,
        data_dir=args.data_dir,
        assets_root=args.assets_root,
        checkpoint=checkpoint,
        task_name=args.task,
        split=args.split,
        episodes=args.episodes,
        pick_radius=args.pick_radius,
        place_radius=args.place_radius)
    print(
        '  oracle_move_agreement='
        f'{result["oracle_move_agreement"]:.4f} '
        'coverage='
        f'{result["legal_candidate_coverage"]:.4f} '
        'latency='
        f'{result["mean_transporter_latency_s"]:.4f}s')
    results.append(result)

  selection = select_best_checkpoint(results)
  payload = {
      'task': args.task,
      'selection_split': args.split,
      'selection_episodes': int(args.episodes),
      'pick_radius': int(args.pick_radius),
      'place_radius': int(args.place_radius),
      **selection,
  }

  output_dir = ensure_dir(os.path.dirname(os.path.abspath(args.output_path)))
  del output_dir
  with open(args.output_path, 'w', encoding='utf-8') as stream:
    json.dump(json_ready(payload), stream, indent=2)

  selected = payload['selected_checkpoint']
  print('Selected checkpoint:')
  print(
      f'  {selected["name"]} step {selected["step"]} '
      f'oracle_move_agreement={selected["oracle_move_agreement"]:.4f} '
      f'meets_threshold={selected["meets_threshold"]}')
  print(f'Saved selection manifest to {os.path.abspath(args.output_path)}')


if __name__ == '__main__':
  main()
