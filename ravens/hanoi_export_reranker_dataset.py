# coding=utf-8
"""Export candidate-level Hanoi reranker supervision from oracle states."""

from __future__ import annotations

import argparse
import json
import os

from ravens.hanoi_reranker import DEFAULT_ORACLE_STEPS
from ravens.hanoi_reranker import export_reranker_split
from ravens.hanoi_utils import ensure_dir
from ravens.hanoi_utils import json_ready


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--root_dir', default='.')
  parser.add_argument('--data_dir', default='.')
  parser.add_argument('--assets_root', default='./ravens/environments/assets')
  parser.add_argument('--task', default='towers-of-hanoi')
  parser.add_argument('--selection_manifest', required=True)
  parser.add_argument('--output_dir', default='./hanoi-reranker-data/export')
  parser.add_argument('--split', default='both', choices=['train', 'test', 'both'])
  parser.add_argument('--episodes', type=int, default=10)
  parser.add_argument('--pick_radius', type=int, default=4)
  parser.add_argument('--place_radius', type=int, default=6)
  return parser.parse_args()


def main():
  args = parse_args()
  output_dir = ensure_dir(args.output_dir)
  splits = ['train', 'test'] if args.split == 'both' else [args.split]
  split_payloads = {}
  for split in splits:
    print(f'Exporting {split} split ({args.episodes} episodes)...')
    split_payloads[split] = export_reranker_split(
        root_dir=args.root_dir,
        data_dir=args.data_dir,
        assets_root=args.assets_root,
        selection_manifest_path=args.selection_manifest,
        output_dir=output_dir,
        task_name=args.task,
        split=split,
        episodes=args.episodes,
        pick_radius=args.pick_radius,
        place_radius=args.place_radius)
    summary = split_payloads[split]
    print(
        f'  rows={summary["rows_exported"]} '
        f'states={summary["states_exported"]} '
        f'positives={summary["positive_labels"]} '
        f'feature_dim={summary["feature_dim"]}')

  metadata = {
      'task': args.task,
      'selection_manifest': os.path.abspath(args.selection_manifest),
      'episodes_requested_per_split': int(args.episodes),
      'oracle_steps': DEFAULT_ORACLE_STEPS,
      'selected_transporter': next(
          iter(split_payloads.values()))['selected_transporter'],
      'feature_names': next(iter(split_payloads.values()))['feature_names'],
      'splits': split_payloads,
  }
  metadata_path = os.path.join(output_dir, 'metadata.json')
  with open(metadata_path, 'w', encoding='utf-8') as stream:
    json.dump(json_ready(metadata), stream, indent=2)
  print(f'Saved dataset metadata to {os.path.abspath(metadata_path)}')


if __name__ == '__main__':
  main()
