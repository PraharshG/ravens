# coding=utf-8
"""CPU-feasible Hanoi checkpoint selection and reranker utilities."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
from ravens import agents
from ravens import tasks
from ravens.dataset import Dataset
from ravens.environments.environment import Environment
from ravens.hanoi_utils import ensure_dir
from ravens.hanoi_utils import find_symbolic_move
from ravens.hanoi_utils import format_peg_state
from ravens.hanoi_utils import get_hanoi_rank_map
from ravens.hanoi_utils import get_optimal_hanoi_move
from ravens.hanoi_utils import assign_disks_to_pegs
from ravens.hanoi_utils import json_ready
from ravens.hanoi_utils import restore_env_from_info
from ravens.hanoi_utils import score_legal_hanoi_moves


FEATURE_NAMES = (
    'disk0_on_left',
    'disk0_on_middle',
    'disk0_on_right',
    'disk1_on_left',
    'disk1_on_middle',
    'disk1_on_right',
    'disk2_on_left',
    'disk2_on_middle',
    'disk2_on_right',
    'candidate_disk0',
    'candidate_disk1',
    'candidate_disk2',
    'candidate_source_left',
    'candidate_source_middle',
    'candidate_source_right',
    'candidate_target_left',
    'candidate_target_middle',
    'candidate_target_right',
    'candidate_transport_score',
    'candidate_rank_fraction',
    'normalized_step_index',
)

RERANKER_HIDDEN_SIZES = (64, 64)
SELECTION_MIN_ORACLE_AGREEMENT = 0.75
DEFAULT_ORACLE_STEPS = 7


def _step_from_checkpoint_path(path: str) -> Optional[int]:
  match = re.search(r'-(\d+)\.h5$', path)
  if match is None:
    return None
  return int(match.group(1))


def _one_hot(index: Optional[int], size: int) -> np.ndarray:
  vector = np.zeros(size, dtype=np.float32)
  if index is not None and 0 <= int(index) < size:
    vector[int(index)] = 1.0
  return vector


def build_state_feature_vector(env) -> np.ndarray:
  """Encode the live symbolic Hanoi state as 3x3 one-hot peg occupancy."""
  rank_map = get_hanoi_rank_map(env)
  disk_to_peg = assign_disks_to_pegs(env)
  rank_to_peg = {
      rank_map[disk_id]: peg for disk_id, peg in disk_to_peg.items()
      if rank_map[disk_id] < 3
  }
  pieces = []
  for rank in range(3):
    pieces.append(_one_hot(rank_to_peg.get(rank), 3))
  return np.concatenate(pieces, axis=0)


def build_candidate_feature_vector(env,
                                   candidate,
                                   candidate_rank: int,
                                   candidate_count: int,
                                   step_index: int,
                                   oracle_steps: int = DEFAULT_ORACLE_STEPS
                                   ) -> np.ndarray:
  """Build the feature vector for one legal Hanoi candidate."""
  pieces = [build_state_feature_vector(env)]
  pieces.append(_one_hot(candidate.disk_rank, 3))
  pieces.append(_one_hot(candidate.source_peg, 3))
  pieces.append(_one_hot(candidate.target_peg, 3))
  rank_fraction = (
      float(candidate_rank) / float(max(candidate_count - 1, 1))
      if candidate_count > 1 else 0.0)
  normalized_step = (
      float(step_index) / float(max(oracle_steps - 1, 1))
      if oracle_steps > 1 else 0.0)
  scalars = np.asarray(
      [float(candidate.score), rank_fraction, normalized_step],
      dtype=np.float32)
  pieces.append(scalars)
  return np.concatenate(pieces, axis=0).astype(np.float32)


def build_candidate_feature_matrix(env,
                                   candidates,
                                   step_index: int,
                                   oracle_steps: int = DEFAULT_ORACLE_STEPS
                                   ) -> np.ndarray:
  """Build a feature matrix for all candidates at the current state."""
  rows = [
      build_candidate_feature_vector(
          env,
          candidate,
          candidate_rank=idx,
          candidate_count=len(candidates),
          step_index=step_index,
          oracle_steps=oracle_steps)
      for idx, candidate in enumerate(candidates)
  ]
  if not rows:
    return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
  return np.stack(rows, axis=0).astype(np.float32)


def discover_transporter_checkpoints(root_dir: str,
                                     task_name: str = 'towers-of-hanoi'
                                     ) -> List[Dict[str, Any]]:
  """Discover transporter checkpoints that have matching attention/transport."""
  checkpoints_dir = os.path.join(root_dir, 'checkpoints')
  if not os.path.isdir(checkpoints_dir):
    return []

  discovered = []
  prefix = f'{task_name}-transporter-'
  for dirname in sorted(os.listdir(checkpoints_dir)):
    if not dirname.startswith(prefix):
      continue
    model_dir = os.path.join(checkpoints_dir, dirname)
    if not os.path.isdir(model_dir):
      continue
    attention_steps = {}
    transport_steps = {}
    for fname in sorted(os.listdir(model_dir)):
      full_path = os.path.join(model_dir, fname)
      if fname.startswith('attention-ckpt-') and fname.endswith('.h5'):
        step = _step_from_checkpoint_path(full_path)
        if step is not None:
          attention_steps[step] = full_path
      if fname.startswith('transport-ckpt-') and fname.endswith('.h5'):
        step = _step_from_checkpoint_path(full_path)
        if step is not None:
          transport_steps[step] = full_path
    common_steps = sorted(set(attention_steps) & set(transport_steps))
    for step in common_steps:
      discovered.append({
          'name': dirname,
          'step': int(step),
          'attention_path': attention_steps[step],
          'transport_path': transport_steps[step],
      })
  return discovered


def evaluate_transporter_checkpoint(root_dir: str,
                                    data_dir: str,
                                    assets_root: str,
                                    checkpoint: Dict[str, Any],
                                    task_name: str = 'towers-of-hanoi',
                                    split: str = 'test',
                                    episodes: int = 5,
                                    pick_radius: int = 4,
                                    place_radius: int = 6
                                    ) -> Dict[str, Any]:
  """Measure legal top-1 oracle agreement for one transporter checkpoint."""
  dataset = Dataset(os.path.join(data_dir, f'{task_name}-{split}'))
  num_episodes = min(int(episodes), dataset.n_episodes)
  env = Environment(assets_root, disp=False, shared_memory=False, hz=480)
  task = tasks.names[task_name]()
  task.mode = split
  env.set_task(task)
  agent = agents.names['transporter'](checkpoint['name'], task_name, root_dir)
  agent.load(int(checkpoint['step']))

  total_states = 0
  oracle_matches = 0
  oracle_covered = 0
  transporter_latency = 0.0
  mean_candidate_count = 0.0

  try:
    for episode_index in range(num_episodes):
      episode, seed = dataset.load(episode_index, images=True)
      np.random.seed(seed)
      env.seed(seed)
      env.set_task(task)
      env.reset()

      for step_index, (obs, _, _, info_snapshot) in enumerate(episode[:-1]):
        if step_index > 0 and info_snapshot is not None:
          restore_env_from_info(env, info_snapshot)
        oracle_move = get_optimal_hanoi_move(env)
        if oracle_move is None:
          break
        _, candidates, latency, coverage = score_legal_hanoi_moves(
            agent,
            env,
            obs,
            pick_radius=pick_radius,
            place_radius=place_radius)
        total_states += 1
        transporter_latency += latency
        oracle_candidate = find_symbolic_move(
            candidates, oracle_move['disk_id'], oracle_move['target_peg'])
        oracle_matches += int(bool(
            candidates and oracle_candidate is not None
            and candidates[0].disk_id == oracle_candidate.disk_id
            and candidates[0].target_peg == oracle_candidate.target_peg))
        oracle_covered += int(bool(coverage and oracle_candidate is not None))
        mean_candidate_count += float(len(candidates))
        if oracle_candidate is None:
          raise ValueError(
              f'Oracle move {oracle_move} was not present in the legal '
              f'candidate set for episode {episode_index} step {step_index}.')
  finally:
    env.close()

  states = max(total_states, 1)
  return {
      'checkpoint': checkpoint,
      'split': split,
      'episodes_evaluated': num_episodes,
      'states_evaluated': total_states,
      'oracle_move_agreement': oracle_matches / states,
      'legal_candidate_coverage': oracle_covered / states,
      'mean_candidate_count': mean_candidate_count / states,
      'mean_transporter_latency_s': transporter_latency / states,
  }


def select_best_checkpoint(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
  """Choose the best checkpoint from a list of evaluation results."""
  if not results:
    raise ValueError('No checkpoint evaluation results were provided.')
  ranked = sorted(
      results,
      key=lambda item: (
          item['oracle_move_agreement'],
          item['legal_candidate_coverage'],
          item['checkpoint']['step'],
      ),
      reverse=True)
  selected = ranked[0]
  return {
      'selected_checkpoint': {
          **selected['checkpoint'],
          'oracle_move_agreement': selected['oracle_move_agreement'],
          'legal_candidate_coverage': selected['legal_candidate_coverage'],
          'mean_candidate_count': selected['mean_candidate_count'],
          'mean_transporter_latency_s': selected['mean_transporter_latency_s'],
          'episodes_evaluated': selected['episodes_evaluated'],
          'states_evaluated': selected['states_evaluated'],
          'meets_threshold': (
              selected['oracle_move_agreement'] >=
              SELECTION_MIN_ORACLE_AGREEMENT),
      },
      'all_checkpoints': ranked,
  }


def _load_selected_checkpoint(selection_manifest_path: str) -> Dict[str, Any]:
  with open(selection_manifest_path, 'r', encoding='utf-8') as stream:
    manifest = json.load(stream)
  selected = manifest.get('selected_checkpoint')
  if not selected:
    raise ValueError(
        f'No selected_checkpoint found in {selection_manifest_path}')
  if not selected.get('meets_threshold', False):
    raise ValueError(
        'The selected transporter checkpoint does not meet the minimum '
        f'oracle agreement threshold of {SELECTION_MIN_ORACLE_AGREEMENT:.2f}.')
  return manifest


def export_reranker_split(root_dir: str,
                          data_dir: str,
                          assets_root: str,
                          selection_manifest_path: str,
                          output_dir: str,
                          task_name: str = 'towers-of-hanoi',
                          split: str = 'train',
                          episodes: Optional[int] = None,
                          pick_radius: int = 4,
                          place_radius: int = 6
                          ) -> Dict[str, Any]:
  """Export candidate-level reranker supervision for one dataset split."""
  manifest = _load_selected_checkpoint(selection_manifest_path)
  selected = manifest['selected_checkpoint']
  dataset = Dataset(os.path.join(data_dir, f'{task_name}-{split}'))
  num_episodes = dataset.n_episodes if episodes is None else min(
      int(episodes), dataset.n_episodes)
  agent = agents.names['transporter'](selected['name'], task_name, root_dir)
  agent.load(int(selected['step']))
  env = Environment(assets_root, disp=False, shared_memory=False, hz=480)
  task = tasks.names[task_name]()
  task.mode = split
  env.set_task(task)

  features = []
  labels = []
  group_ids = []
  episode_ids = []
  seeds = []
  step_indices = []
  candidate_indices = []
  transporter_scores = []
  candidate_rank_fraction = []
  normalized_step_indices = []
  oracle_candidate_indices = []
  audit_rows = []
  transporter_latency = 0.0
  state_counter = 0
  row_counter = 0

  try:
    for episode_index in range(num_episodes):
      episode, seed = dataset.load(episode_index, images=True)
      np.random.seed(seed)
      env.seed(seed)
      env.set_task(task)
      env.reset()
      oracle_steps = max(len(episode), 1)

      for logical_step, (obs, _, _, info_snapshot) in enumerate(episode[:-1]):
        if logical_step > 0 and info_snapshot is not None:
          restore_env_from_info(env, info_snapshot)
        oracle_move = get_optimal_hanoi_move(env)
        if oracle_move is None:
          break

        _, candidates, latency, _ = score_legal_hanoi_moves(
            agent,
            env,
            obs,
            pick_radius=pick_radius,
            place_radius=place_radius)
        transporter_latency += latency
        if not candidates:
          raise ValueError(
              f'No legal candidates were scored for episode {episode_index} '
              f'step {logical_step}.')
        positive_indices = [
            idx for idx, candidate in enumerate(candidates)
            if candidate.disk_id == oracle_move['disk_id']
            and candidate.target_peg == oracle_move['target_peg']
        ]
        if len(positive_indices) != 1:
          raise ValueError(
              f'Expected exactly one oracle-matching candidate, found '
              f'{len(positive_indices)} for episode {episode_index} '
              f'step {logical_step}.')
        oracle_candidate_index = positive_indices[0]
        feature_matrix = build_candidate_feature_matrix(
            env,
            candidates,
            step_index=logical_step,
            oracle_steps=oracle_steps)
        peg_state = format_peg_state(env)
        group_id = state_counter
        state_counter += 1

        for candidate_index, candidate in enumerate(candidates):
          features.append(feature_matrix[candidate_index])
          labels.append(float(candidate_index == oracle_candidate_index))
          group_ids.append(group_id)
          episode_ids.append(int(episode_index))
          seeds.append(int(seed))
          step_indices.append(int(logical_step))
          candidate_indices.append(int(candidate_index))
          transporter_scores.append(float(candidate.score))
          candidate_rank_fraction.append(
              float(candidate_index) / float(max(len(candidates) - 1, 1))
              if len(candidates) > 1 else 0.0)
          normalized_step_indices.append(
              float(logical_step) / float(max(oracle_steps - 1, 1))
              if oracle_steps > 1 else 0.0)
          oracle_candidate_indices.append(int(oracle_candidate_index))
          audit_rows.append({
              'row_index': row_counter,
              'episode_index': int(episode_index),
              'seed': int(seed),
              'step_index': int(logical_step),
              'candidate_index': int(candidate_index),
              'oracle_candidate_index': int(oracle_candidate_index),
              'label': int(candidate_index == oracle_candidate_index),
              'candidate_description': candidate.description(),
              'peg_state': peg_state,
              'transport_score': float(candidate.score),
              'oracle_move': oracle_move['description'],
          })
          row_counter += 1
  finally:
    env.close()

  output_dir = ensure_dir(output_dir)
  npz_path = os.path.join(output_dir, f'{split}.npz')
  audit_path = os.path.join(output_dir, f'{split}_audit.jsonl')
  features_array = np.asarray(features, dtype=np.float32)
  labels_array = np.asarray(labels, dtype=np.float32)
  group_ids_array = np.asarray(group_ids, dtype=np.int32)
  step_indices_array = np.asarray(step_indices, dtype=np.int32)
  np.savez(
      npz_path,
      features=features_array,
      labels=labels_array,
      group_ids=group_ids_array,
      episode_ids=np.asarray(episode_ids, dtype=np.int32),
      seeds=np.asarray(seeds, dtype=np.int32),
      step_indices=step_indices_array,
      candidate_indices=np.asarray(candidate_indices, dtype=np.int32),
      transporter_scores=np.asarray(transporter_scores, dtype=np.float32),
      candidate_rank_fraction=np.asarray(
          candidate_rank_fraction, dtype=np.float32),
      normalized_step_indices=np.asarray(
          normalized_step_indices, dtype=np.float32),
      oracle_candidate_indices=np.asarray(
          oracle_candidate_indices, dtype=np.int32))

  with open(audit_path, 'w', encoding='utf-8') as stream:
    for row in audit_rows:
      stream.write(json.dumps(row) + '\n')

  return {
      'split': split,
      'path': npz_path,
      'audit_path': audit_path,
      'episodes_exported': num_episodes,
      'states_exported': int(len(np.unique(group_ids_array))) if group_ids else 0,
      'rows_exported': int(features_array.shape[0]),
      'positive_labels': int(np.sum(labels_array)),
      'feature_dim': int(features_array.shape[1]) if features_array.size else 0,
      'mean_candidates_per_state': (
          float(features_array.shape[0]) / float(max(len(np.unique(group_ids_array)), 1))
          if group_ids else 0.0),
      'mean_transporter_latency_s': (
          transporter_latency / float(max(len(np.unique(group_ids_array)), 1))
          if group_ids else 0.0),
      'selected_transporter': selected,
      'feature_names': list(FEATURE_NAMES),
  }


def load_reranker_dataset(npz_path: str) -> Dict[str, np.ndarray]:
  """Load an exported reranker dataset split."""
  payload = np.load(npz_path)
  return {key: payload[key] for key in payload.files}


def build_reranker_model(input_dim: int,
                         hidden_sizes: Sequence[int] = RERANKER_HIDDEN_SIZES,
                         learning_rate: float = 1e-3) -> tf.keras.Model:
  """Build the small CPU-feasible MLP reranker."""
  model = tf.keras.Sequential(name='hanoi_reranker')
  model.add(tf.keras.layers.Input(shape=(input_dim,)))
  for hidden_size in hidden_sizes:
    model.add(tf.keras.layers.Dense(hidden_size, activation='relu'))
  model.add(tf.keras.layers.Dense(1, activation=None))
  model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
      loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
      metrics=[
          tf.keras.metrics.BinaryAccuracy(
              name='binary_accuracy', threshold=0.0),
      ])
  model(np.zeros((1, input_dim), dtype=np.float32))
  return model


def group_top1_oracle_agreement(logits: np.ndarray,
                                labels: np.ndarray,
                                group_ids: np.ndarray) -> float:
  """Compute top-1 oracle agreement grouped by state/candidate set."""
  if logits.size == 0:
    return 0.0
  unique_groups = np.unique(group_ids)
  correct = 0
  for group_id in unique_groups:
    mask = group_ids == group_id
    group_logits = logits[mask]
    group_labels = labels[mask]
    selected_index = int(np.argmax(group_logits))
    correct += int(group_labels[selected_index] > 0.5)
  return float(correct) / float(max(len(unique_groups), 1))


def evaluate_model_on_dataset(model: tf.keras.Model,
                              dataset: Dict[str, np.ndarray]) -> Dict[str, Any]:
  """Evaluate loss and grouped top-1 oracle agreement on one split."""
  features = dataset['features'].astype(np.float32)
  labels = dataset['labels'].astype(np.float32)
  group_ids = dataset['group_ids'].astype(np.int32)
  metrics = model.evaluate(features, labels, verbose=0, return_dict=True)
  logits = model(features, training=False).numpy().reshape(-1)
  return {
      'loss': float(metrics['loss']),
      'binary_accuracy': float(metrics.get('binary_accuracy', 0.0)),
      'auc': float(metrics.get('auc', 0.0)),
      'candidate_top1_oracle_agreement': group_top1_oracle_agreement(
          logits, labels, group_ids),
      'num_rows': int(features.shape[0]),
      'num_states': int(len(np.unique(group_ids))),
  }


def split_dataset_by_group(dataset: Dict[str, np.ndarray],
                           validation_fraction: float = 0.2,
                           seed: int = 0
                           ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
  """Split a candidate dataset into train/validation subsets by state group."""
  group_ids = dataset['group_ids'].astype(np.int32)
  unique_groups = np.unique(group_ids)
  if unique_groups.size < 2:
    raise ValueError('Need at least two unique state groups for validation.')
  rng = np.random.default_rng(seed)
  shuffled_groups = np.array(unique_groups, copy=True)
  rng.shuffle(shuffled_groups)
  num_val_groups = int(np.ceil(float(unique_groups.size) * validation_fraction))
  num_val_groups = min(max(num_val_groups, 1), unique_groups.size - 1)
  validation_groups = set(int(group_id) for group_id in shuffled_groups[:num_val_groups])
  validation_mask = np.asarray(
      [int(group_id) in validation_groups for group_id in group_ids],
      dtype=bool)
  train_mask = ~validation_mask
  if not np.any(train_mask) or not np.any(validation_mask):
    raise ValueError('Group-based validation split produced an empty subset.')

  def subset(mask: np.ndarray) -> Dict[str, np.ndarray]:
    return {key: value[mask] for key, value in dataset.items()}

  return subset(train_mask), subset(validation_mask)


def train_reranker(train_dataset_path: str,
                   test_dataset_path: str,
                   output_dir: str,
                   epochs: int = 50,
                   batch_size: int = 64,
                   learning_rate: float = 1e-3,
                   patience: int = 5,
                   validation_fraction: float = 0.2) -> Dict[str, Any]:
  """Train the MLP reranker and save the resulting weights and metadata."""
  np.random.seed(0)
  tf.random.set_seed(0)
  train_data = load_reranker_dataset(train_dataset_path)
  test_data = load_reranker_dataset(test_dataset_path)
  fit_train_data, validation_data = split_dataset_by_group(
      train_data,
      validation_fraction=validation_fraction,
      seed=0)
  input_dim = int(train_data['features'].shape[1])
  model = build_reranker_model(
      input_dim=input_dim,
      hidden_sizes=RERANKER_HIDDEN_SIZES,
      learning_rate=learning_rate)
  early_stopping = tf.keras.callbacks.EarlyStopping(
      monitor='val_loss',
      patience=patience,
      restore_best_weights=True)
  history = model.fit(
      fit_train_data['features'],
      fit_train_data['labels'],
      validation_data=(
          validation_data['features'],
          validation_data['labels']),
      epochs=epochs,
      batch_size=batch_size,
      verbose=0,
      callbacks=[early_stopping],
      shuffle=True)

  train_metrics = evaluate_model_on_dataset(model, train_data)
  validation_metrics = evaluate_model_on_dataset(model, validation_data)
  test_metrics = evaluate_model_on_dataset(model, test_data)

  output_dir = ensure_dir(output_dir)
  weights_path = os.path.join(output_dir, 'model.weights.h5')
  metadata_path = os.path.join(output_dir, 'metadata.json')
  model.save_weights(weights_path)

  dataset_metadata_path = os.path.join(
      os.path.dirname(train_dataset_path), 'metadata.json')
  dataset_metadata = {}
  if os.path.exists(dataset_metadata_path):
    with open(dataset_metadata_path, 'r', encoding='utf-8') as stream:
      dataset_metadata = json.load(stream)

  metadata = {
      'input_dim': input_dim,
      'feature_names': list(FEATURE_NAMES),
      'hidden_sizes': list(RERANKER_HIDDEN_SIZES),
      'learning_rate': float(learning_rate),
      'epochs_requested': int(epochs),
      'epochs_trained': int(len(history.history.get('loss', []))),
      'batch_size': int(batch_size),
      'patience': int(patience),
      'validation_fraction': float(validation_fraction),
      'train_dataset_path': os.path.abspath(train_dataset_path),
      'test_dataset_path': os.path.abspath(test_dataset_path),
      'selected_transporter': dataset_metadata.get('selected_transporter'),
      'dataset_metadata': dataset_metadata,
      'validation_metrics': validation_metrics,
      'train_metrics': train_metrics,
      'test_metrics': test_metrics,
  }
  with open(metadata_path, 'w', encoding='utf-8') as stream:
    json.dump(json_ready(metadata), stream, indent=2)
  return {
      'output_dir': output_dir,
      'weights_path': weights_path,
      'metadata_path': metadata_path,
      'validation_metrics': validation_metrics,
      'train_metrics': train_metrics,
      'test_metrics': test_metrics,
  }


def load_reranker(output_dir: str) -> Tuple[tf.keras.Model, Dict[str, Any]]:
  """Load a trained reranker and its metadata."""
  metadata_path = os.path.join(output_dir, 'metadata.json')
  weights_path = os.path.join(output_dir, 'model.weights.h5')
  if not os.path.exists(metadata_path) or not os.path.exists(weights_path):
    raise FileNotFoundError(
        f'Missing reranker artifacts in {output_dir}. Expected metadata.json '
        'and model.weights.h5.')
  with open(metadata_path, 'r', encoding='utf-8') as stream:
    metadata = json.load(stream)
  model = build_reranker_model(
      input_dim=int(metadata['input_dim']),
      hidden_sizes=metadata.get('hidden_sizes', RERANKER_HIDDEN_SIZES),
      learning_rate=float(metadata.get('learning_rate', 1e-3)))
  model.load_weights(weights_path)
  return model, metadata


class HanoiRerankerPolicy:
  """Runtime policy wrapper that scores legal candidates with the MLP."""

  def __init__(self, model: tf.keras.Model, metadata: Dict[str, Any]):
    self.model = model
    self.metadata = metadata
    selected = metadata.get('selected_transporter') or metadata.get(
        'dataset_metadata', {}).get('selected_transporter')
    self.selected_transporter = selected or {}
    self.oracle_steps = int(
        metadata.get('dataset_metadata', {}).get('oracle_steps',
                                                 DEFAULT_ORACLE_STEPS))

  @classmethod
  def from_dir(cls, output_dir: str) -> 'HanoiRerankerPolicy':
    model, metadata = load_reranker(output_dir)
    return cls(model, metadata)

  def select_candidate(self,
                       env,
                       candidates,
                       step_index: int) -> Tuple[Any, np.ndarray, float]:
    """Return the highest-logit legal candidate and inference latency."""
    if not candidates:
      raise ValueError('No candidates available for reranker selection.')
    features = build_candidate_feature_matrix(
        env,
        candidates,
        step_index=step_index,
        oracle_steps=self.oracle_steps)
    started = time.perf_counter()
    logits = self.model(features, training=False).numpy().reshape(-1)
    latency = time.perf_counter() - started
    selected_index = int(np.argmax(logits))
    selected = candidates[selected_index]
    selected.rationale = (
        f'Reranker selected [{selected_index}] logit={logits[selected_index]:.4f}')
    return selected, logits, latency

  def selected_transporter_spec(self) -> Dict[str, Any]:
    return dict(self.selected_transporter)
