# coding=utf-8
"""Utilities for Towers of Hanoi benchmarking and visualization."""

from __future__ import annotations

import dataclasses
import json
import os
import select
import shlex
import subprocess
import time
from collections import deque
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pybullet as p
from ravens.dataset import CAMERA_CONFIG
from ravens.models.gt_state import MlpModel
from ravens.utils import utils


HANOI_BOUNDS = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])
HANOI_PIX_SIZE = 0.003125

PEG_LABELS = ('left', 'middle', 'right')
DISK_LABELS = (
    'small red disk',
    'medium yellow disk',
    'large green disk',
    'extra disk',
)
CANVAS_COLORS = (
    (70, 130, 180),
    (255, 140, 0),
    (34, 139, 34),
    (148, 0, 211),
    (220, 20, 60),
    (0, 206, 209),
)


@dataclasses.dataclass
class HanoiMove:
  """Discrete move annotation for a predicted action."""

  disk_id: Optional[int]
  disk_rank: Optional[int]
  disk_label: str
  source_peg: Optional[int]
  target_peg: Optional[int]
  legal: bool
  matches_oracle: bool
  score: float
  pick_pixel: Tuple[int, int]
  place_pixel: Tuple[int, int]
  pick_theta: float
  place_theta: float
  source_name: str
  fallback_used: bool = False
  rationale: str = ''

  def description(self) -> str:
    legality = 'legal' if self.legal else 'illegal'
    return (f'{self.disk_label}: {peg_name(self.source_peg)} -> '
            f'{peg_name(self.target_peg)} ({legality})')


def ensure_dir(path: str) -> str:
  os.makedirs(path, exist_ok=True)
  return path


def peg_name(index: Optional[int]) -> str:
  if index is None:
    return 'unknown peg'
  if 0 <= index < len(PEG_LABELS):
    return f'{PEG_LABELS[index]} peg'
  return f'peg {index}'


def disk_label(rank: Optional[int], disk_id: Optional[int] = None) -> str:
  if rank is not None and 0 <= rank < len(DISK_LABELS):
    return DISK_LABELS[rank]
  if disk_id is None:
    return 'unknown disk'
  return f'disk {disk_id}'


def json_ready(value: Any) -> Any:
  """Convert nested values into JSON-serializable types."""
  if dataclasses.is_dataclass(value):
    return json_ready(dataclasses.asdict(value))
  if isinstance(value, dict):
    return {str(k): json_ready(v) for k, v in value.items()}
  if isinstance(value, (list, tuple)):
    return [json_ready(v) for v in value]
  if isinstance(value, np.ndarray):
    return value.tolist()
  if isinstance(value, (np.integer,)):
    return int(value)
  if isinstance(value, (np.floating,)):
    return float(value)
  if isinstance(value, (np.bool_,)):
    return bool(value)
  return value


def legacy_action_from_dataset_action(action: Dict[str, Any]) -> Dict[str, Any]:
  """Wrap a modern dataset action in the legacy gt_state shape."""
  return {
      'camera_config': CAMERA_CONFIG,
      'primitive': 'pick_place',
      'params': {
          'pose0': action['pose0'],
          'pose1': action['pose1'],
      }
  }


def normalize_env_action(action: Dict[str, Any]) -> Dict[str, Any]:
  """Normalize current and legacy action formats to env.step inputs."""
  if 'params' in action:
    return action['params']
  return action


class LegacyDatasetAdapter:
  """Adapter exposing the older random_sample API for baseline agents."""

  def __init__(self, dataset):
    self.dataset = dataset

  def random_sample(self):
    goal_info = None
    for _ in range(16):
      (obs, action, _, info), goal = self.dataset.sample()
      goal_info = goal[3]
      if info is not None:
        return obs, legacy_action_from_dataset_action(action), info
    if goal_info is None:
      raise ValueError('Unable to sample a transition with non-empty info.')
    return obs, legacy_action_from_dataset_action(action), goal_info

  @property
  def n_episodes(self) -> int:
    return self.dataset.n_episodes


def get_hanoi_stand_pose(env) -> Tuple[Tuple[float, float, float],
                                       Tuple[float, float, float, float]]:
  stand_id = env.obj_ids['fixed'][0]
  return env.info[stand_id][:2]


def get_hanoi_stand_pose_from_info(
    env, info: Dict[int, Tuple[Any, Any, Any]]) -> Tuple[Any, Any]:
  """Return the Hanoi stand pose from an arbitrary info snapshot."""
  stand_id = env.obj_ids['fixed'][0]
  return info[stand_id][:2]


def get_hanoi_disk_ids(env) -> List[int]:
  return sorted(env.obj_ids['rigid'])


def get_hanoi_rank_map(env) -> Dict[int, int]:
  return {disk_id: rank for rank, disk_id in enumerate(get_hanoi_disk_ids(env))}


def get_hanoi_peg_positions(env) -> List[Tuple[float, float, float]]:
  stand_pose = get_hanoi_stand_pose(env)
  rod_pos = ((0, -0.12, 0.03), (0, 0, 0.03), (0, 0.12, 0.03))
  return [utils.apply(stand_pose, pos) for pos in rod_pos]


def get_hanoi_peg_positions_from_info(
    env, info: Dict[int, Tuple[Any, Any, Any]]) -> List[Tuple[float, float, float]]:
  """Return peg positions from an arbitrary info snapshot."""
  stand_pose = get_hanoi_stand_pose_from_info(env, info)
  rod_pos = ((0, -0.12, 0.03), (0, 0, 0.03), (0, 0.12, 0.03))
  return [utils.apply(stand_pose, pos) for pos in rod_pos]


def get_hanoi_peg_pixels(env,
                         bounds: np.ndarray,
                         pix_size: float) -> List[Tuple[int, int]]:
  return [utils.xyz_to_pix(pos, bounds, pix_size)
          for pos in get_hanoi_peg_positions(env)]


def get_disk_pixel_positions(env,
                             bounds: np.ndarray,
                             pix_size: float) -> Dict[int, Tuple[int, int]]:
  pixels = {}
  for disk_id in get_hanoi_disk_ids(env):
    position = env.info[disk_id][0]
    pixels[disk_id] = utils.xyz_to_pix(position, bounds, pix_size)
  return pixels


def assign_disks_to_pegs(env) -> Dict[int, int]:
  peg_positions = np.float32([pos[:2] for pos in get_hanoi_peg_positions(env)])
  mapping = {}
  for disk_id in get_hanoi_disk_ids(env):
    disk_xy = np.float32(env.info[disk_id][0][:2]).reshape(1, 2)
    distances = np.linalg.norm(peg_positions - disk_xy, axis=1)
    mapping[disk_id] = int(np.argmin(distances))
  return mapping


def assign_disks_to_pegs_from_info(
    env, info: Dict[int, Tuple[Any, Any, Any]]) -> Dict[int, int]:
  """Assign disks to pegs from a saved info snapshot."""
  peg_positions = np.float32(
      [pos[:2] for pos in get_hanoi_peg_positions_from_info(env, info)])
  mapping = {}
  for disk_id in get_hanoi_disk_ids(env):
    disk_xy = np.float32(info[disk_id][0][:2]).reshape(1, 2)
    distances = np.linalg.norm(peg_positions - disk_xy, axis=1)
    mapping[disk_id] = int(np.argmin(distances))
  return mapping


def restore_env_from_info(env, info: Dict[int, Tuple[Any, Any, Any]]) -> None:
  """Reset live object poses to a saved environment info snapshot."""
  for obj_id, (position, rotation, _) in info.items():
    p.resetBasePositionAndOrientation(obj_id, position, rotation)
    p.resetBaseVelocity(obj_id, (0, 0, 0), (0, 0, 0))


def get_peg_stacks(env) -> Dict[int, List[int]]:
  """Return disks on each peg ordered bottom-to-top."""
  rank_map = get_hanoi_rank_map(env)
  disk_to_peg = assign_disks_to_pegs(env)
  stacks = {0: [], 1: [], 2: []}
  for disk_id in get_hanoi_disk_ids(env):
    peg = disk_to_peg[disk_id]
    stacks[peg].append(disk_id)
  for peg in stacks:
    # Use disk rank, not raw z, to define symbolic stack order. Saved or live
    # physics states can drift slightly in z without changing the intended Hanoi
    # ordering, but disk size is stable.
    stacks[peg].sort(key=lambda disk_id: rank_map[disk_id], reverse=True)
  return stacks


def top_disks_by_peg(env) -> Dict[int, Optional[int]]:
  stacks = get_peg_stacks(env)
  return {peg: (stack[-1] if stack else None) for peg, stack in stacks.items()}


def format_peg_state(env) -> str:
  """Return a readable peg-state summary from the live environment."""
  rank_map = get_hanoi_rank_map(env)
  stacks = get_peg_stacks(env)
  lines = []
  for peg in range(3):
    labels = [disk_label(rank_map[disk_id], disk_id) for disk_id in stacks[peg]]
    if not labels:
      labels = ['empty']
    lines.append(f'{peg_name(peg)}: {", ".join(labels)}')
  return '\n'.join(lines)


def get_legal_moves(env) -> List[Tuple[int, int]]:
  rank_map = get_hanoi_rank_map(env)
  tops = top_disks_by_peg(env)
  legal = []
  for source_peg, disk_id in tops.items():
    if disk_id is None:
      continue
    disk_rank = rank_map[disk_id]
    for target_peg in range(3):
      if target_peg == source_peg:
        continue
      target_top = tops[target_peg]
      if target_top is None or rank_map[target_top] > disk_rank:
        legal.append((disk_id, target_peg))
  return legal


def get_oracle_move(env) -> Optional[Dict[str, Any]]:
  if not env.task.goals:
    return None
  objs, _, targs, _, _, _, _, _ = env.task.goals[0]
  disk_id = int(objs[0][0])
  rank_map = get_hanoi_rank_map(env)
  disk_to_peg = assign_disks_to_pegs(env)
  target_xy = np.float32(targs[0][0][:2]).reshape(1, 2)
  peg_positions = np.float32([pos[:2] for pos in get_hanoi_peg_positions(env)])
  target_peg = int(np.argmin(np.linalg.norm(peg_positions - target_xy, axis=1)))
  source_peg = disk_to_peg[disk_id]
  return {
      'disk_id': disk_id,
      'disk_rank': rank_map[disk_id],
      'disk_label': disk_label(rank_map[disk_id], disk_id),
      'source_peg': source_peg,
      'target_peg': target_peg,
      'description': (
          f'{disk_label(rank_map[disk_id], disk_id)}: '
          f'{peg_name(source_peg)} -> {peg_name(target_peg)}'
      ),
  }


def infer_oracle_move_from_info_transition(
    env,
    next_info: Dict[int, Tuple[Any, Any, Any]]) -> Optional[Dict[str, Any]]:
  """Infer the symbolic Hanoi move from the current state to a next info snapshot."""
  current_mapping = assign_disks_to_pegs(env)
  next_mapping = assign_disks_to_pegs_from_info(env, next_info)
  changed = [
      disk_id for disk_id in get_hanoi_disk_ids(env)
      if current_mapping[disk_id] != next_mapping[disk_id]
  ]
  if not changed:
    return None
  if len(changed) != 1:
    raise ValueError(
        f'Expected exactly one moving disk in transition, found {changed}.')
  disk_id = changed[0]
  rank_map = get_hanoi_rank_map(env)
  source_peg = current_mapping[disk_id]
  target_peg = next_mapping[disk_id]
  return {
      'disk_id': disk_id,
      'disk_rank': rank_map[disk_id],
      'disk_label': disk_label(rank_map[disk_id], disk_id),
      'source_peg': source_peg,
      'target_peg': target_peg,
      'description': (
          f'{disk_label(rank_map[disk_id], disk_id)}: '
          f'{peg_name(source_peg)} -> {peg_name(target_peg)}'
      ),
  }


def _symbolic_hanoi_state(env) -> Tuple[int, ...]:
  rank_map = get_hanoi_rank_map(env)
  disk_to_peg = assign_disks_to_pegs(env)
  state = [0] * len(rank_map)
  for disk_id, rank in rank_map.items():
    state[rank] = disk_to_peg[disk_id]
  return tuple(state)


def _legal_symbolic_hanoi_moves(
    state: Tuple[int, ...]) -> List[Tuple[int, int, int]]:
  tops = {0: None, 1: None, 2: None}
  for disk_rank, peg in enumerate(state):
    current_top = tops[peg]
    if current_top is None or disk_rank < current_top:
      tops[peg] = disk_rank
  legal = []
  for source_peg, disk_rank in tops.items():
    if disk_rank is None:
      continue
    for target_peg in range(3):
      if target_peg == source_peg:
        continue
      target_top = tops[target_peg]
      if target_top is None or target_top > disk_rank:
        legal.append((disk_rank, source_peg, target_peg))
  return legal


def _apply_symbolic_hanoi_move(
    state: Tuple[int, ...],
    move: Tuple[int, int, int]) -> Tuple[int, ...]:
  disk_rank, _, target_peg = move
  next_state = list(state)
  next_state[disk_rank] = target_peg
  return tuple(next_state)


def get_optimal_hanoi_move(env, goal_peg: int = 2) -> Optional[Dict[str, Any]]:
  """Solve the current symbolic Hanoi state and return the next optimal move."""
  state = _symbolic_hanoi_state(env)
  goal_state = tuple(goal_peg for _ in state)
  if state == goal_state:
    return None

  queue = deque([state])
  parents = {state: None}
  inbound_move = {}

  while queue:
    current = queue.popleft()
    if current == goal_state:
      break
    for move in _legal_symbolic_hanoi_moves(current):
      next_state = _apply_symbolic_hanoi_move(current, move)
      if next_state in parents:
        continue
      parents[next_state] = current
      inbound_move[next_state] = move
      queue.append(next_state)

  if goal_state not in parents:
    raise ValueError(f'No symbolic Hanoi path found from state {state}.')

  move = inbound_move[goal_state]
  parent = parents[goal_state]
  while parent is not None and parent != state:
    move = inbound_move[parent]
    parent = parents[parent]

  disk_rank, source_peg, target_peg = move
  rank_map = get_hanoi_rank_map(env)
  inverse_rank_map = {rank: disk_id for disk_id, rank in rank_map.items()}
  disk_id = inverse_rank_map[disk_rank]
  return {
      'disk_id': disk_id,
      'disk_rank': disk_rank,
      'disk_label': disk_label(disk_rank, disk_id),
      'source_peg': source_peg,
      'target_peg': target_peg,
      'description': (
          f'{disk_label(disk_rank, disk_id)}: '
          f'{peg_name(source_peg)} -> {peg_name(target_peg)}'
      ),
  }


def enumerate_legal_hanoi_moves(env,
                                bounds: np.ndarray = HANOI_BOUNDS,
                                pix_size: float = HANOI_PIX_SIZE,
                                source_name: str = 'symbolic') -> List[HanoiMove]:
  """Enumerate legal Hanoi moves from the live symbolic state."""
  rank_map = get_hanoi_rank_map(env)
  disk_to_peg = assign_disks_to_pegs(env)
  disk_pixels = get_disk_pixel_positions(env, bounds, pix_size)
  peg_pixels = get_hanoi_peg_pixels(env, bounds, pix_size)
  oracle_move = get_oracle_move(env)
  candidates = []
  for disk_id, target_peg in get_legal_moves(env):
    source_peg = disk_to_peg[disk_id]
    candidates.append(
        HanoiMove(
            disk_id=disk_id,
            disk_rank=rank_map[disk_id],
            disk_label=disk_label(rank_map[disk_id], disk_id),
            source_peg=source_peg,
            target_peg=target_peg,
            legal=True,
            matches_oracle=bool(
                oracle_move and disk_id == oracle_move['disk_id']
                and target_peg == oracle_move['target_peg']),
            score=0.0,
            pick_pixel=tuple(int(v) for v in disk_pixels[disk_id]),
            place_pixel=tuple(int(v) for v in peg_pixels[target_peg]),
            pick_theta=0.0,
            place_theta=0.0,
            source_name=source_name))
  return candidates


def _topk_indices(array: np.ndarray, k: int) -> List[Tuple[int, ...]]:
  flat = array.reshape(-1)
  k = max(1, min(k, flat.shape[0]))
  indices = np.argpartition(flat, -k)[-k:]
  indices = indices[np.argsort(flat[indices])[::-1]]
  return [tuple(int(i) for i in np.unravel_index(index, array.shape))
          for index in indices]


def _pick_place_to_action(agent,
                          img: np.ndarray,
                          pick_index: Tuple[int, int, int],
                          place_index: Tuple[int, int, int]) -> Dict[str, Any]:
  p0_pix = pick_index[:2]
  p0_theta = pick_index[2] * (2 * np.pi / agent.attention.n_rotations)
  p1_pix = place_index[:2]
  p1_theta = place_index[2] * (2 * np.pi / agent.transport.n_rotations)
  hmap = img[:, :, 3]
  p0_xyz = utils.pix_to_xyz(p0_pix, hmap, agent.bounds, agent.pix_size)
  p1_xyz = utils.pix_to_xyz(p1_pix, hmap, agent.bounds, agent.pix_size)
  return {
      'pose0': (np.asarray(p0_xyz), np.asarray(
          utils.eulerXYZ_to_quatXYZW((0, 0, -p0_theta)))),
      'pose1': (np.asarray(p1_xyz), np.asarray(
          utils.eulerXYZ_to_quatXYZW((0, 0, -p1_theta)))),
  }


def annotate_action(env,
                    action: Dict[str, Any],
                    bounds: np.ndarray,
                    pix_size: float,
                    score: float,
                    source_name: str,
                    fallback_used: bool = False,
                    rationale: str = '') -> HanoiMove:
  """Annotate an action with discrete Hanoi semantics."""
  env_action = normalize_env_action(action)
  pick_pos = env_action['pose0'][0]
  place_pos = env_action['pose1'][0]
  pick_rot = env_action['pose0'][1]
  place_rot = env_action['pose1'][1]

  pick_pixel = utils.xyz_to_pix(pick_pos, bounds, pix_size)
  place_pixel = utils.xyz_to_pix(place_pos, bounds, pix_size)
  pick_theta = -float(utils.quatXYZW_to_eulerXYZ(pick_rot)[2])
  place_theta = -float(utils.quatXYZW_to_eulerXYZ(place_rot)[2])

  rank_map = get_hanoi_rank_map(env)
  disk_pixels = get_disk_pixel_positions(env, bounds, pix_size)
  disk_to_peg = assign_disks_to_pegs(env)
  tops = top_disks_by_peg(env)
  peg_pixels = np.float32(get_hanoi_peg_pixels(env, bounds, pix_size))

  disk_id = None
  disk_rank = None
  if disk_pixels:
    pick_uv = np.float32(pick_pixel).reshape(1, 2)
    disk_id = min(
        disk_pixels,
        key=lambda k: np.linalg.norm(
            np.float32(disk_pixels[k]).reshape(1, 2) - pick_uv))
    disk_rank = rank_map[disk_id]

  source_peg = disk_to_peg.get(disk_id) if disk_id is not None else None
  target_peg = int(np.argmin(
      np.linalg.norm(peg_pixels - np.float32(place_pixel).reshape(1, 2), axis=1)))

  legal = False
  if disk_id is not None and source_peg is not None:
    legal = tops[source_peg] == disk_id
    target_top = tops[target_peg]
    if source_peg == target_peg:
      legal = False
    elif legal and target_top is not None:
      legal = rank_map[target_top] > disk_rank

  oracle_move = get_oracle_move(env)
  matches_oracle = bool(
      oracle_move and disk_id == oracle_move['disk_id']
      and target_peg == oracle_move['target_peg'])

  return HanoiMove(
      disk_id=disk_id,
      disk_rank=disk_rank,
      disk_label=disk_label(disk_rank, disk_id),
      source_peg=source_peg,
      target_peg=target_peg,
      legal=legal,
      matches_oracle=matches_oracle,
      score=float(score),
      pick_pixel=(int(pick_pixel[0]), int(pick_pixel[1])),
      place_pixel=(int(place_pixel[0]), int(place_pixel[1])),
      pick_theta=float(pick_theta),
      place_theta=float(place_theta),
      source_name=source_name,
      fallback_used=fallback_used,
      rationale=rationale)


def _window_slice(center: Tuple[int, int],
                  radius: int,
                  shape: Sequence[int]) -> Tuple[slice, slice]:
  y, x = center
  y0 = max(0, y - radius)
  y1 = min(shape[0], y + radius + 1)
  x0 = max(0, x - radius)
  x1 = min(shape[1], x + radius + 1)
  return slice(y0, y1), slice(x0, x1)


def _best_pose_from_confidence(confidence_map: np.ndarray,
                               center_pixel: Tuple[int, int],
                               radius: int,
                               n_rotations: int) -> Tuple[Tuple[int, int], float, float]:
  max_y = confidence_map.shape[0] - 1
  max_x = confidence_map.shape[1] - 1
  clamped_center = (
      int(np.clip(center_pixel[0], 0, max_y)),
      int(np.clip(center_pixel[1], 0, max_x)))
  ys, xs = _window_slice(clamped_center, radius, confidence_map.shape[:2])
  patch = confidence_map[ys, xs, :]
  if patch.size == 0:
    argmax = np.unravel_index(np.argmax(confidence_map), confidence_map.shape)
    pixel = (int(argmax[0]), int(argmax[1]))
    theta = float(argmax[2]) * (2 * np.pi / n_rotations)
    score = float(confidence_map[argmax])
    return pixel, theta, score
  argmax = np.unravel_index(np.argmax(patch), patch.shape)
  pixel = (ys.start + int(argmax[0]), xs.start + int(argmax[1]))
  theta = float(argmax[2]) * (2 * np.pi / n_rotations)
  score = float(patch[argmax])
  return pixel, theta, score


def score_legal_hanoi_moves(agent,
                            env,
                            obs: Dict[str, Any],
                            pick_radius: int = 4,
                            place_radius: int = 6
                            ) -> Tuple[np.ndarray, List[HanoiMove], float, float]:
  """Enumerate legal moves and score them with transporter heatmaps."""
  started = time.perf_counter()
  img = agent.get_image(obs)
  pick_conf = agent.attention.forward(img)
  symbolic_candidates = enumerate_legal_hanoi_moves(
      env, agent.bounds, agent.pix_size, source_name='symbolic-legal')
  scored: List[HanoiMove] = []
  for candidate in symbolic_candidates:
    pick_pixel, pick_theta, pick_score = _best_pose_from_confidence(
        pick_conf, candidate.pick_pixel, pick_radius, agent.attention.n_rotations)
    place_conf = agent.transport.forward(img, pick_pixel)
    place_pixel, place_theta, place_score = _best_pose_from_confidence(
        place_conf, candidate.place_pixel, place_radius,
        agent.transport.n_rotations)
    scored.append(
        dataclasses.replace(
            candidate,
            pick_pixel=pick_pixel,
            place_pixel=place_pixel,
            pick_theta=pick_theta,
            place_theta=place_theta,
            score=float(pick_score * place_score),
            source_name='transporter-legal'))

  scored.sort(key=lambda move: move.score, reverse=True)
  latency = time.perf_counter() - started
  oracle_covered = float(any(move.matches_oracle for move in scored))
  return img, scored, latency, oracle_covered


def _project_pose_to_planar_yaw(pose):
  position, rotation = pose
  euler = utils.quatXYZW_to_eulerXYZ(rotation)
  yaw_only = utils.eulerXYZ_to_quatXYZW((0, 0, euler[2]))
  return position, yaw_only


def _pick_pose_from_disk_mask(env,
                              disk_id: int
                              ) -> Tuple[Tuple[np.ndarray, np.ndarray],
                                         Tuple[int, int]]:
  _, hmap, obj_mask = env.task.get_true_image(env)
  pick_mask = np.uint8(obj_mask == disk_id)
  if np.any(pick_mask):
    distance = cv2.distanceTransform(pick_mask, cv2.DIST_L2, 5)
    pick_pixel = tuple(int(v) for v in np.unravel_index(
        np.argmax(distance), distance.shape))
    if distance[pick_pixel] > 0:
      pick_xyz = utils.pix_to_xyz(
          pick_pixel, hmap, env.task.bounds, env.task.pix_size)
      return (
          np.asarray(pick_xyz, dtype=np.float32),
          np.asarray((0, 0, 0, 1), dtype=np.float32)), pick_pixel

  pick_position = np.asarray(env.info[disk_id][0], dtype=np.float32)
  pick_pixel = utils.xyz_to_pix(
      pick_position, env.task.bounds, env.task.pix_size)
  return (
      pick_position,
      np.asarray((0, 0, 0, 1), dtype=np.float32)), pick_pixel


def build_hanoi_execution_action(env,
                                 candidate: HanoiMove) -> Dict[str, Any]:
  """Build a physically valid Hanoi action for a symbolic/legal move."""
  if candidate.disk_id is None or candidate.target_peg is None:
    raise ValueError('A legal Hanoi candidate requires disk_id and target_peg.')

  pick_pose, _ = _pick_pose_from_disk_mask(env, candidate.disk_id)
  object_pose = p.getBasePositionAndOrientation(candidate.disk_id)
  if not env.task.sixdof:
    object_pose = _project_pose_to_planar_yaw(object_pose)

  target_pose = (
      np.asarray(get_hanoi_peg_positions(env)[candidate.target_peg],
                 dtype=np.float32),
      np.asarray((0, 0, 0, 1), dtype=np.float32))
  world_to_pick = utils.invert(pick_pose)
  obj_to_pick = utils.multiply(world_to_pick, object_pose)
  pick_to_obj = utils.invert(obj_to_pick)
  place_pose = utils.multiply(target_pose, pick_to_obj)

  return {
      'pose0': (
          np.asarray(pick_pose[0], dtype=np.float32),
          np.asarray(pick_pose[1], dtype=np.float32)),
      'pose1': (
          np.asarray(place_pose[0], dtype=np.float32),
          np.asarray(place_pose[1], dtype=np.float32)),
  }


def extract_transporter_candidates(agent,
                                   env,
                                   obs: Dict[str, Any],
                                   top_pick_k: int = 3,
                                   top_place_k: int = 3,
                                   max_candidates: int = 6
                                   ) -> Tuple[np.ndarray, List[HanoiMove], float]:
  """Extract and annotate raw top-k transporter candidates."""
  started = time.perf_counter()
  img = agent.get_image(obs)
  pick_conf = agent.attention.forward(img)
  candidates: List[HanoiMove] = []

  for pick_index in _topk_indices(pick_conf, top_pick_k):
    p0_pix = pick_index[:2]
    place_conf = agent.transport.forward(img, p0_pix)
    pick_score = float(pick_conf[pick_index])
    for place_index in _topk_indices(place_conf, top_place_k):
      action = _pick_place_to_action(agent, img, pick_index, place_index)
      total_score = pick_score * float(place_conf[place_index])
      move = annotate_action(
          env=env,
          action=action,
          bounds=agent.bounds,
          pix_size=agent.pix_size,
          score=total_score,
          source_name='transporter-raw')
      candidates.append(move)

  unique: List[HanoiMove] = []
  seen = set()
  for candidate in sorted(candidates, key=lambda c: c.score, reverse=True):
    key = (candidate.disk_id, candidate.target_peg)
    if key in seen:
      continue
    seen.add(key)
    unique.append(candidate)
    if len(unique) >= max_candidates:
      break

  latency = time.perf_counter() - started
  return img, unique, latency


def choose_fallback_candidate(candidates: Sequence[HanoiMove]) -> HanoiMove:
  """Choose a fallback candidate prioritizing legality and score."""
  if not candidates:
    raise ValueError('No candidates available for fallback selection.')
  scored = [(candidate.legal, candidate.score, idx, candidate)
            for idx, candidate in enumerate(candidates)]
  scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
  return scored[0][3]


def find_symbolic_move(candidates: Sequence[HanoiMove],
                       disk_id: Optional[int],
                       target_peg: Optional[int]) -> Optional[HanoiMove]:
  """Find the candidate matching a symbolic Hanoi move."""
  for candidate in candidates:
    if candidate.disk_id == disk_id and candidate.target_peg == target_peg:
      return candidate
  return None


def collect_oracle_prompt_examples(assets_root: str,
                                   data_dir: str,
                                   task_name: str,
                                   limit: int = 3) -> List[Dict[str, Any]]:
  """Collect compact oracle-derived exemplars from train seeds."""
  if limit <= 0:
    return []

  from ravens import tasks  # pylint: disable=g-import-not-at-top
  from ravens.dataset import Dataset  # pylint: disable=g-import-not-at-top
  from ravens.environments.environment import Environment  # pylint: disable=g-import-not-at-top

  dataset = Dataset(os.path.join(data_dir, f'{task_name}-train'))
  if dataset.n_episodes == 0:
    return []

  env = Environment(assets_root, disp=False, shared_memory=False, hz=480)
  task = tasks.names[task_name]()
  task.mode = 'train'
  examples = []

  try:
    for episode_index in range(dataset.n_episodes):
      _, seed = dataset.load(episode_index, images=False)
      np.random.seed(seed)
      env.seed(seed)
      env.set_task(task)
      obs = env.reset()
      info = None
      oracle_agent = env.task.oracle(env)

      for _ in range(task.max_steps):
        legal_candidates = enumerate_legal_hanoi_moves(env)
        action = oracle_agent.act(obs, info)
        selected = annotate_action(
            env,
            action,
            HANOI_BOUNDS,
            HANOI_PIX_SIZE,
            score=1.0,
            source_name='oracle')
        examples.append({
            'peg_state': format_peg_state(env),
            'candidate_descriptions': [
                candidate.description() for candidate in legal_candidates
            ],
            'selected_description': selected.description(),
        })
        if len(examples) >= limit:
          return examples
        obs, _, done, info = env.step(action)
        if done:
          break
  finally:
    env.close()

  return examples


def save_gt_state_checkpoint(agent, checkpoint_dir: str, step: int) -> None:
  """Persist gt_state weights and metadata without touching legacy code."""
  if agent.model is None:
    raise ValueError('gt_state model has not been initialized.')
  ensure_dir(checkpoint_dir)
  dummy = np.zeros((1, agent.max_obs_vector_length), dtype=np.float32)
  agent.model(dummy)
  weights_path = os.path.join(checkpoint_dir, f'model-ckpt-{step}.weights.h5')
  metadata_path = os.path.join(checkpoint_dir, f'model-ckpt-{step}.metadata.npz')
  agent.model.save_weights(weights_path)
  np.savez(
      metadata_path,
      max_obs_vector_length=np.int32(agent.max_obs_vector_length),
      mean=np.asarray(agent.model.obs_train_mean, dtype=np.float32),
      std=np.asarray(agent.model.obs_train_std, dtype=np.float32),
      six_dof=np.bool_(agent.six_dof))


def load_gt_state_checkpoint(agent, checkpoint_dir: str, step: int) -> None:
  """Load gt_state weights and metadata produced by save_gt_state_checkpoint."""
  metadata_path = os.path.join(checkpoint_dir, f'model-ckpt-{step}.metadata.npz')
  weights_path = os.path.join(checkpoint_dir, f'model-ckpt-{step}.weights.h5')
  if not os.path.exists(metadata_path) or not os.path.exists(weights_path):
    raise FileNotFoundError(
        f'Missing gt_state checkpoint files for step {step} in {checkpoint_dir}')

  metadata = np.load(metadata_path)
  agent.max_obs_vector_length = int(metadata['max_obs_vector_length'])
  agent.six_dof = bool(metadata['six_dof'])
  act_dim = 9 if agent.six_dof else 6
  agent.model = MlpModel(
      agent.batch_size,
      agent.max_obs_vector_length,
      act_dim,
      'relu',
      agent.use_mdn,
      dropout=0.1)
  agent.model.set_normalization_parameters({
      'mean': metadata['mean'].astype(np.float32),
      'std': metadata['std'].astype(np.float32),
  })
  dummy = np.zeros((1, agent.max_obs_vector_length), dtype=np.float32)
  agent.model(dummy)
  agent.model.load_weights(weights_path)
  agent.total_iter = step


class VLMWorkerClient:
  """Line-delimited JSON client for the external VLM worker."""

  def __init__(self,
               command: str,
               cwd: Optional[str] = None,
               timeout_s: float = 120.0):
    self.command = command
    self.cwd = cwd
    self.timeout_s = timeout_s
    self.process: Optional[subprocess.Popen] = None

  def start(self) -> None:
    if self.process is not None:
      return
    args = shlex.split(self.command)
    self.process = subprocess.Popen(
        args,
        cwd=self.cwd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1)

  def request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    if self.process is None:
      self.start()
    assert self.process is not None
    if self.process.stdin is None or self.process.stdout is None:
      raise RuntimeError('VLM worker pipes are not available.')
    self.process.stdin.write(json.dumps(json_ready(payload)) + '\n')
    self.process.stdin.flush()

    deadline = time.time() + self.timeout_s
    while time.time() < deadline:
      if self.process.poll() is not None:
        stderr = ''
        if self.process.stderr is not None:
          stderr = self.process.stderr.read()
        raise RuntimeError(
            f'VLM worker exited with code {self.process.returncode}: {stderr}')
      wait_s = max(0.0, min(0.05, deadline - time.time()))
      ready, _, _ = select.select([self.process.stdout], [], [], wait_s)
      if not ready:
        continue
      line = self.process.stdout.readline()
      if line:
        return json.loads(line)
    raise TimeoutError(f'VLM worker timed out after {self.timeout_s} seconds.')

  def close(self) -> None:
    if self.process is None:
      return
    try:
      if self.process.stdin is not None:
        self.process.stdin.write(json.dumps({'type': 'shutdown'}) + '\n')
        self.process.stdin.flush()
    except BrokenPipeError:
      pass
    if self.process.poll() is None:
      self.process.terminate()
      try:
        self.process.wait(timeout=5)
      except subprocess.TimeoutExpired:
        self.process.kill()
        self.process.wait(timeout=5)
    self.process = None


def _scale_pixel(point: Tuple[int, int], image_size: int) -> Tuple[int, int]:
  return int(point[1] * image_size / 320), int(point[0] * image_size / 320)


def render_scene_image(obs: Dict[str, Any], size: int = 640) -> np.ndarray:
  """Render an unannotated top-down scene image from the observation."""
  cmap, _ = utils.get_fused_heightmap(obs, CAMERA_CONFIG, HANOI_BOUNDS,
                                      HANOI_PIX_SIZE)
  board = cv2.cvtColor(cmap, cv2.COLOR_RGB2BGR)
  return cv2.resize(board, (size, size), interpolation=cv2.INTER_NEAREST)


def render_candidate_board(obs: Dict[str, Any],
                           env,
                           candidates: Sequence[HanoiMove],
                           selected: Optional[HanoiMove] = None,
                           footer_lines: Sequence[str] = (),
                           show_scores: bool = True) -> np.ndarray:
  """Render a numbered top-down candidate board for the VLM and rollouts."""
  board_size = 640
  legend_width = 470
  cmap, _ = utils.get_fused_heightmap(obs, CAMERA_CONFIG, HANOI_BOUNDS,
                                      HANOI_PIX_SIZE)
  board = cv2.cvtColor(cmap, cv2.COLOR_RGB2BGR)
  board = cv2.resize(board, (board_size, board_size),
                     interpolation=cv2.INTER_NEAREST)

  for peg_index, peg_pixel in enumerate(get_hanoi_peg_pixels(env, HANOI_BOUNDS,
                                                             HANOI_PIX_SIZE)):
    scaled = _scale_pixel(peg_pixel, board_size)
    cv2.circle(board, scaled, 8, (255, 255, 255), -1)
    cv2.putText(board, PEG_LABELS[peg_index], (scaled[0] - 30, scaled[1] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
                cv2.LINE_AA)

  rank_map = get_hanoi_rank_map(env)
  for disk_id, pixel in get_disk_pixel_positions(env, HANOI_BOUNDS,
                                                 HANOI_PIX_SIZE).items():
    scaled = _scale_pixel(pixel, board_size)
    label = DISK_LABELS[rank_map[disk_id]].replace(' disk', '')
    cv2.putText(board, label, (scaled[0] + 8, scaled[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.putText(board, label, (scaled[0] + 8, scaled[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1,
                cv2.LINE_AA)

  for idx, candidate in enumerate(candidates):
    color = CANVAS_COLORS[idx % len(CANVAS_COLORS)]
    pick = _scale_pixel(candidate.pick_pixel, board_size)
    place = _scale_pixel(candidate.place_pixel, board_size)
    thickness = 4 if (
        selected is not None and candidate.disk_id == selected.disk_id
        and candidate.target_peg == selected.target_peg) else 2
    cv2.arrowedLine(board, pick, place, color, thickness, tipLength=0.15)
    cv2.circle(board, pick, 10, color, thickness)
    cv2.circle(board, place, 10, color, thickness)
    mid = ((pick[0] + place[0]) // 2, (pick[1] + place[1]) // 2)
    cv2.putText(board, str(idx), (mid[0] + 6, mid[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(board, str(idx), (mid[0] + 6, mid[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)

  legend = np.full((board_size, legend_width, 3), 245, dtype=np.uint8)
  cv2.putText(legend, 'Legal Hanoi Moves', (16, 28), cv2.FONT_HERSHEY_SIMPLEX,
              0.7, (30, 30, 30), 2, cv2.LINE_AA)
  cv2.putText(legend, 'All numbered moves are legal options.', (16, 54),
              cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1, cv2.LINE_AA)

  y = 84
  for idx, candidate in enumerate(candidates):
    color = CANVAS_COLORS[idx % len(CANVAS_COLORS)]
    text = f'[{idx}] {candidate.description()}'
    if show_scores:
      text = f'{text} score={candidate.score:.4f}'
    if selected is not None and candidate.disk_id == selected.disk_id \
        and candidate.target_peg == selected.target_peg:
      cv2.rectangle(legend, (8, y - 18), (legend_width - 10, y + 10),
                    (225, 255, 225), -1)
    cv2.putText(legend, text[:62], (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                color, 2, cv2.LINE_AA)
    y += 28
    if y > 360:
      break

  footer_y = 420
  for line in footer_lines[:8]:
    cv2.putText(legend, line[:64], (16, footer_y), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (40, 40, 40), 1, cv2.LINE_AA)
    footer_y += 22

  return np.concatenate((board, legend), axis=1)


class RolloutRecorder:
  """Simple MP4 recorder with simulation and numbered candidate board."""

  def __init__(self,
               output_dir: str,
               episode_index: int,
               fps: int = 6,
               enabled: bool = True):
    self.enabled = enabled
    self.output_dir = ensure_dir(output_dir)
    self.video_path = os.path.join(
        self.output_dir, f'episode-{episode_index:03d}.mp4')
    self.frame_dir = ensure_dir(
        os.path.join(self.output_dir, f'episode-{episode_index:03d}-frames'))
    self.writer = None
    self.fps = fps
    self.frame_count = 0

  def write(self,
            env,
            obs: Dict[str, Any],
            candidates: Sequence[HanoiMove],
            selected: Optional[HanoiMove],
            footer_lines: Sequence[str]) -> Optional[str]:
    if not self.enabled:
      return None
    sim_frame = cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR)
    sim_frame = cv2.resize(sim_frame, (853, 640))
    board = render_candidate_board(obs, env, candidates, selected, footer_lines)
    board = cv2.resize(board, (1110, 640))
    canvas = np.concatenate((sim_frame, board), axis=1)

    if self.writer is None:
      fourcc = cv2.VideoWriter_fourcc(*'mp4v')
      self.writer = cv2.VideoWriter(
          self.video_path, fourcc, self.fps, (canvas.shape[1], canvas.shape[0]))
    self.writer.write(canvas)

    frame_path = os.path.join(self.frame_dir, f'frame-{self.frame_count:04d}.png')
    cv2.imwrite(frame_path, canvas)
    self.frame_count += 1
    return frame_path

  def close(self) -> None:
    if self.writer is not None:
      self.writer.release()
      self.writer = None


def summarize_episodes(episodes: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
  if not episodes:
    return {}

  def mean_of(key: str) -> float:
    return float(np.mean([episode[key] for episode in episodes]))

  def std_of(key: str) -> float:
    return float(np.std([episode[key] for episode in episodes]))

  summary = {
      'num_episodes': len(episodes),
      'success_rate': mean_of('success'),
      'success_rate_std': std_of('success'),
      'mean_reward': mean_of('total_reward'),
      'mean_reward_std': std_of('total_reward'),
      'legal_move_rate': mean_of('legal_move_rate'),
      'legal_move_rate_std': std_of('legal_move_rate'),
      'oracle_move_agreement': mean_of('oracle_move_agreement'),
      'oracle_move_agreement_std': std_of('oracle_move_agreement'),
      'mean_steps': mean_of('steps_executed'),
      'mean_steps_std': std_of('steps_executed'),
      'mean_transporter_latency_s': mean_of('transporter_latency_s'),
      'mean_reranker_latency_s': mean_of('reranker_latency_s'),
      'mean_vlm_latency_s': mean_of('vlm_latency_s'),
      'mean_total_latency_s': mean_of('total_latency_s'),
      'vlm_fallback_rate': mean_of('vlm_fallback_rate'),
      'vlm_parse_success_rate': mean_of('vlm_parse_success_rate'),
      'vlm_parse_success_rate_std': std_of('vlm_parse_success_rate'),
      'vlm_invalid_response_rate': mean_of('vlm_invalid_response_rate'),
      'vlm_invalid_response_rate_std': std_of('vlm_invalid_response_rate'),
      'legal_candidate_coverage': mean_of('legal_candidate_coverage'),
      'legal_candidate_coverage_std': std_of('legal_candidate_coverage'),
      'planner_override_rate': mean_of('planner_override_rate'),
      'planner_override_rate_std': std_of('planner_override_rate'),
      'executed_oracle_move_agreement': mean_of('executed_oracle_move_agreement'),
      'executed_oracle_move_agreement_std': std_of(
          'executed_oracle_move_agreement'),
      'repeat_state_rate': mean_of('repeat_state_rate'),
      'repeat_state_rate_std': std_of('repeat_state_rate'),
      'mean_history_images_used': mean_of('mean_history_images_used'),
  }
  return summary
