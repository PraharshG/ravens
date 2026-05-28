# coding=utf-8
"""Train the CPU-feasible Hanoi legal-move reranker."""

from __future__ import annotations

import argparse
import json
import os

from ravens.hanoi_reranker import train_reranker


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--train_path', required=True)
  parser.add_argument('--test_path', required=True)
  parser.add_argument('--output_dir', default='./checkpoints/hanoi-reranker/default')
  parser.add_argument('--epochs', type=int, default=50)
  parser.add_argument('--batch_size', type=int, default=64)
  parser.add_argument('--learning_rate', type=float, default=1e-3)
  parser.add_argument('--patience', type=int, default=5)
  parser.add_argument('--validation_fraction', type=float, default=0.2)
  return parser.parse_args()


def main():
  args = parse_args()
  result = train_reranker(
      train_dataset_path=args.train_path,
      test_dataset_path=args.test_path,
      output_dir=args.output_dir,
      epochs=args.epochs,
      batch_size=args.batch_size,
      learning_rate=args.learning_rate,
      patience=args.patience,
      validation_fraction=args.validation_fraction)
  print('Reranker training complete.')
  print(f'  output_dir: {os.path.abspath(result["output_dir"])}')
  print('  train_metrics:')
  print(json.dumps(result['train_metrics'], indent=2))
  print('  validation_metrics:')
  print(json.dumps(result['validation_metrics'], indent=2))
  print('  test_metrics:')
  print(json.dumps(result['test_metrics'], indent=2))


if __name__ == '__main__':
  main()
