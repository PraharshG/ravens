# coding=utf-8
"""Aggregate Hanoi benchmark JSON files into graphs and summary tables."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


METRICS = {
    'success_vs_steps.png': ('success_rate', 'success_rate_std',
                             'Success Rate', None),
    'legal_move_rate.png': ('legal_move_rate', 'legal_move_rate_std',
                            'Legal Move Rate', None),
    'oracle_move_agreement.png': ('oracle_move_agreement',
                                  'oracle_move_agreement_std',
                                  'Oracle Move Agreement', None),
    'steps_to_success.png': ('mean_steps', 'mean_steps_std',
                             'Mean Steps Per Episode', None),
    'latency.png': ('mean_total_latency_s', None,
                    'Mean Action Latency (s)', None),
    'legal_candidate_coverage.png': (
        'legal_candidate_coverage', 'legal_candidate_coverage_std',
        'Legal Candidate Coverage', {'vlm-transporter'}),
    'vlm_parse_success_rate.png': (
        'vlm_parse_success_rate', 'vlm_parse_success_rate_std',
        'VLM Parse Success Rate', {'vlm-transporter'}),
    'vlm_invalid_response_rate.png': (
        'vlm_invalid_response_rate', 'vlm_invalid_response_rate_std',
        'VLM Invalid Response Rate', {'vlm-transporter'}),
    'vlm_fallback_rate.png': (
        'vlm_fallback_rate', None,
        'VLM Fallback Rate', {'vlm-transporter'}),
    'planner_override_rate.png': (
        'planner_override_rate', 'planner_override_rate_std',
        'Planner Override Rate', {'vlm-transporter'}),
    'executed_oracle_move_agreement.png': (
        'executed_oracle_move_agreement',
        'executed_oracle_move_agreement_std',
        'Executed Oracle Agreement', {'vlm-transporter'}),
    'repeat_state_rate.png': (
        'repeat_state_rate', 'repeat_state_rate_std',
        'Repeat State Rate', None),
    'history_images_used.png': (
        'mean_history_images_used', None,
        'Mean History Images Used', {'vlm-transporter'}),
    'reranker_latency.png': (
        'mean_reranker_latency_s', None,
        'Mean Reranker Latency (s)', {'reranker-transporter'}),
}


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_dir', default='./hanoi-results')
  parser.add_argument('--output_dir', default='./hanoi-results/report')
  return parser.parse_args()


def load_results(input_dir):
  results = []
  for fname in sorted(os.listdir(input_dir)):
    if not fname.endswith('.json'):
      continue
    path = os.path.join(input_dir, fname)
    with open(path, 'r', encoding='utf-8') as stream:
      results.append(json.load(stream))
  return results


def aggregate_curves(results):
  grouped = defaultdict(list)
  for result in results:
    key = (
        result['agent'],
        result.get('candidate_mode', ''),
        result['n_demos'],
        result.get('decision_policy', ''),
        result.get('history_images', 0),
        result.get('vlm_backend', ''),
        result.get('reranker_dir', ''),
    )
    grouped[key].append(result)
  return grouped


def write_summary(results, output_dir):
  summary_path = os.path.join(output_dir, 'summary.csv')
  json_path = os.path.join(output_dir, 'summary.json')
  rows = []
  for result in results:
    row = {
        'agent': result['agent'],
        'candidate_mode': result.get('candidate_mode', ''),
        'decision_policy': result.get('decision_policy', ''),
        'history_images': result.get('history_images', 0),
        'vlm_backend': result.get('vlm_backend', ''),
        'reranker_dir': result.get('reranker_dir', ''),
        'n_demos': result['n_demos'],
        'checkpoint_step': result['checkpoint_step'],
        **result['summary'],
    }
    rows.append(row)

  with open(summary_path, 'w', newline='', encoding='utf-8') as stream:
    writer = csv.DictWriter(stream, fieldnames=sorted(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

  with open(json_path, 'w', encoding='utf-8') as stream:
    json.dump(rows, stream, indent=2)
  return summary_path, json_path


def plot_metric(grouped, output_dir, filename, metric_key, std_key, ylabel,
                allowed_agents=None):
  plt.figure(figsize=(10, 6))
  plotted = False
  for key, entries in sorted(grouped.items()):
    agent, candidate_mode, n_demos, decision_policy, history_images, vlm_backend, reranker_dir = key
    if allowed_agents is not None and agent not in allowed_agents:
      continue
    entries = sorted(entries, key=lambda item: item['checkpoint_step'])
    x = np.array([entry['checkpoint_step'] for entry in entries], dtype=np.float32)
    y = np.array([entry['summary'].get(metric_key, 0.0) for entry in entries],
                 dtype=np.float32)
    label_parts = [agent, f'{n_demos} demos']
    if candidate_mode:
      label_parts.append(candidate_mode)
    if decision_policy:
      label_parts.append(decision_policy)
    if history_images:
      label_parts.append(f'hist={history_images}')
    if vlm_backend:
      label_parts.append(vlm_backend)
    if reranker_dir:
      label_parts.append(os.path.basename(os.path.normpath(reranker_dir)))
    label = ' | '.join(label_parts)
    plt.plot(x, y, marker='o', linewidth=2, label=label)
    plotted = True
    if std_key is not None:
      std = np.array([entry['summary'].get(std_key, 0.0) for entry in entries],
                     dtype=np.float32)
      plt.fill_between(x, y - std, y + std, alpha=0.2)

  if not plotted:
    plt.close()
    return None

  plt.xlabel('Training Steps')
  plt.ylabel(ylabel)
  plt.title(f'Hanoi {ylabel} Comparison')
  plt.grid(True, linestyle='--', alpha=0.4)
  plt.legend()
  plt.tight_layout()
  output_path = os.path.join(output_dir, filename)
  plt.savefig(output_path)
  plt.close()
  return output_path


def main():
  args = parse_args()
  os.makedirs(args.output_dir, exist_ok=True)
  results = load_results(args.input_dir)
  if not results:
    raise ValueError(f'No benchmark JSON files found in {args.input_dir}')

  grouped = aggregate_curves(results)
  summary_csv, summary_json = write_summary(results, args.output_dir)
  created = [summary_csv, summary_json]
  for filename, (metric_key, std_key, ylabel, allowed_agents) in METRICS.items():
    output_path = plot_metric(grouped, args.output_dir, filename, metric_key,
                              std_key, ylabel, allowed_agents)
    if output_path is not None:
      created.append(output_path)

  print('Created report artifacts:')
  for path in created:
    print(path)


if __name__ == '__main__':
  main()
