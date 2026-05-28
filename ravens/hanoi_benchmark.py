# coding=utf-8
"""Benchmark runner for Towers of Hanoi baselines and VLM reranking."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from ravens import agents
from ravens import tasks
from ravens.dataset import Dataset
from ravens.environments.environment import Environment
from ravens.hanoi_reranker import HanoiRerankerPolicy
from ravens.hanoi_utils import HANOI_BOUNDS
from ravens.hanoi_utils import HANOI_PIX_SIZE
from ravens.hanoi_utils import LegacyDatasetAdapter
from ravens.hanoi_utils import RolloutRecorder
from ravens.hanoi_utils import VLMWorkerClient
from ravens.hanoi_utils import annotate_action
from ravens.hanoi_utils import build_hanoi_execution_action
from ravens.hanoi_utils import choose_fallback_candidate
from ravens.hanoi_utils import collect_oracle_prompt_examples
from ravens.hanoi_utils import ensure_dir
from ravens.hanoi_utils import extract_transporter_candidates
from ravens.hanoi_utils import find_symbolic_move
from ravens.hanoi_utils import format_peg_state
from ravens.hanoi_utils import get_oracle_move
from ravens.hanoi_utils import json_ready
from ravens.hanoi_utils import load_gt_state_checkpoint
from ravens.hanoi_utils import normalize_env_action
from ravens.hanoi_utils import render_candidate_board
from ravens.hanoi_utils import render_scene_image
from ravens.hanoi_utils import save_gt_state_checkpoint
from ravens.hanoi_utils import score_legal_hanoi_moves
from ravens.hanoi_utils import summarize_episodes
from ravens.utils import utils
import tensorflow as tf


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--root_dir', default='.')
  parser.add_argument('--data_dir', default='.')
  parser.add_argument('--assets_root', default='./ravens/environments/assets')
  parser.add_argument('--task', default='towers-of-hanoi')
  parser.add_argument(
      '--agent',
      default='transporter',
      choices=[
          'transporter',
          'vlm-transporter',
          'reranker-transporter',
          'gt_state',
          'oracle',
      ])
  parser.add_argument('--n_demos', type=int, default=10)
  parser.add_argument('--n_steps', type=int, default=1000)
  parser.add_argument('--train_gt_state', action='store_true')
  parser.add_argument('--gt_state_train_steps', type=int, default=5000)
  parser.add_argument('--gt_state_interval', type=int, default=1000)
  parser.add_argument('--episodes', type=int, default=100)
  parser.add_argument('--run_index', type=int, default=0)
  parser.add_argument('--top_pick_k', type=int, default=3)
  parser.add_argument('--top_place_k', type=int, default=3)
  parser.add_argument('--max_candidates', type=int, default=6)
  parser.add_argument('--candidate_mode', default='legal',
                      choices=['legal', 'raw'])
  parser.add_argument('--pick_radius', type=int, default=4)
  parser.add_argument('--place_radius', type=int, default=6)
  parser.add_argument('--num_prompt_exemplars', type=int, default=3)
  parser.add_argument('--include_oracle_in_prompt', action='store_true')
  parser.add_argument('--output_dir', default='./hanoi-results')
  parser.add_argument('--vlm_command', default='')
  parser.add_argument('--vlm_timeout_s', type=float, default=120.0)
  parser.add_argument('--reranker_dir', default='')
  parser.add_argument('--decision_policy', default='guarded',
                      choices=['guarded', 'vlm-only'])
  parser.add_argument('--history_images', type=int, default=2)
  parser.add_argument('--record_rollout', action='store_true')
  parser.add_argument('--record_episode_index', type=int, default=0)
  parser.add_argument('--disp', action='store_true')
  parser.add_argument('--shared_memory', action='store_true')
  parser.add_argument('--gpu', type=int, default=0)
  return parser.parse_args()


def configure_tensorflow(gpu_index: int):
  cfg = tf.config.experimental
  gpus = cfg.list_physical_devices('GPU')
  if not gpus:
    print('No GPUs detected. Running with CPU.')
    return
  cfg.set_visible_devices(gpus[gpu_index], 'GPU')


def build_model_name(args) -> str:
  agent_name = args.agent
  if args.agent in ('vlm-transporter', 'reranker-transporter'):
    agent_name = 'transporter'
  return f'{args.task}-{agent_name}-{args.n_demos}-{args.run_index}'


def _selected_transporter_run_spec(
    selected_transporter: Dict[str, Any],
    default_n_demos: int,
    default_run_index: int,
    default_step: int) -> Tuple[int, int, int]:
  name = str(selected_transporter.get('name', ''))
  parts = name.split('-')
  if len(parts) >= 2:
    try:
      return int(parts[-2]), int(parts[-1]), int(selected_transporter['step'])
    except (KeyError, TypeError, ValueError):
      pass
  return default_n_demos, default_run_index, default_step


def build_result_name(args,
                      selected_transporter: Optional[Dict[str, Any]] = None) -> str:
  parts = [args.task, args.agent]
  if args.agent in ('vlm-transporter', 'reranker-transporter'):
    parts.append(args.candidate_mode)
  if args.agent == 'vlm-transporter':
    parts.extend([
        args.decision_policy,
        f'h{max(args.history_images, 0)}',
    ])
  n_demos = args.n_demos
  run_index = args.run_index
  checkpoint_step = args.n_steps
  if args.agent == 'reranker-transporter' and selected_transporter:
    n_demos, run_index, checkpoint_step = _selected_transporter_run_spec(
        selected_transporter, args.n_demos, args.run_index, args.n_steps)
  parts.extend([str(n_demos), str(run_index), str(checkpoint_step)])
  return '-'.join(parts)


def action_from_move(agent, obs, move):
  img = agent.get_image(obs)
  hmap = img[:, :, 3]
  p0_xyz = utils.pix_to_xyz(move.pick_pixel, hmap, agent.bounds, agent.pix_size)
  p1_xyz = utils.pix_to_xyz(move.place_pixel, hmap, agent.bounds, agent.pix_size)
  return {
      'pose0': (
          np.asarray(p0_xyz),
          np.asarray(utils.eulerXYZ_to_quatXYZW((0, 0, -move.pick_theta)))),
      'pose1': (
          np.asarray(p1_xyz),
          np.asarray(utils.eulerXYZ_to_quatXYZW((0, 0, -move.place_theta)))),
  }


def choose_vlm_candidate(client: Optional[VLMWorkerClient],
                         request_image_path: str,
                         history_image_paths,
                         candidates,
                         peg_state: str,
                         move_history,
                         prompt_examples,
                         oracle_move: Optional[Dict[str, Any]] = None,
                         include_oracle_in_prompt: bool = False) -> Dict[str, Any]:
  if client is None:
    raise ValueError('A VLM worker command is required for vlm-transporter.')

  payload = {
      'type': 'select_candidate',
      'task': 'towers-of-hanoi',
      'goal': 'Choose the single best next Towers of Hanoi move.',
      'image_path': request_image_path,
      'current_request_image_path': request_image_path,
      'history_image_paths': list(history_image_paths),
      'peg_state': peg_state,
      'move_history': move_history[-3:],
      'prompt_examples': prompt_examples,
      'candidates': [
          {
              'index': idx,
              'description': candidate.description(),
              'disk_label': candidate.disk_label,
              'source_peg': candidate.source_peg,
              'target_peg': candidate.target_peg,
              'legal': candidate.legal,
              'score': candidate.score,
          } for idx, candidate in enumerate(candidates)
      ]
  }
  if include_oracle_in_prompt and oracle_move is not None:
    payload['oracle_move'] = oracle_move

  started = time.perf_counter()
  response = client.request(payload)
  response['latency_s'] = time.perf_counter() - started
  return response


def train_or_load_gt_state(args, result_name: str):
  checkpoint_dir = os.path.join(args.root_dir, 'checkpoints', result_name)
  train_dataset = LegacyDatasetAdapter(
      Dataset(os.path.join(args.data_dir, f'{args.task}-train')))
  test_dataset = LegacyDatasetAdapter(
      Dataset(os.path.join(args.data_dir, f'{args.task}-test')))
  log_dir = os.path.join(args.root_dir, 'logs', 'gt_state', args.task,
                         datetime.datetime.now().strftime('%Y%m%d-%H%M%S'),
                         'train')
  writer = tf.summary.create_file_writer(log_dir)
  agent = agents.names['gt_state'](result_name, args.task)

  if args.train_gt_state:
    checkpoints = list(range(args.gt_state_interval,
                             args.gt_state_train_steps + 1,
                             args.gt_state_interval))
    trained_steps = 0
    for step in checkpoints:
      agent.train(
          train_dataset,
          step - trained_steps,
          writer,
          test_dataset)
      save_gt_state_checkpoint(agent, checkpoint_dir, step)
      trained_steps = step
    if args.n_steps not in checkpoints:
      save_gt_state_checkpoint(agent, checkpoint_dir, args.n_steps)
    return agent

  load_gt_state_checkpoint(agent, checkpoint_dir, args.n_steps)
  return agent


def select_action_for_step(args,
                           episode_index: int,
                           step_index: int,
                           env,
                           obs,
                           info,
                           agent,
                           reranker_policy: Optional[HanoiRerankerPolicy],
                           worker: Optional[VLMWorkerClient],
                           request_dir: str,
                           history_dir: str,
                           scene_history_paths,
                           move_history,
                           prompt_examples):
  """Return action, candidates, selected move, and step diagnostics."""
  del info
  oracle_move = get_oracle_move(env)
  diagnostics = {
      'transporter_latency_s': 0.0,
      'vlm_latency_s': 0.0,
      'fallback_used': False,
      'candidate_coverage': 0.0,
      'parse_success': 0.0,
      'invalid_response': 0.0,
      'worker_error': '',
      'vlm_response': None,
      'request_image_path': '',
      'current_scene_image_path': '',
      'history_image_paths': [],
      'history_images_used': 0,
      'planner_override': False,
      'oracle_agreement': 0.0,
      'executed_oracle_agreement': 0.0,
      'vlm_backend': '',
      'vlm_selected_move': None,
      'planner_move': None,
      'reranker_latency_s': 0.0,
      'reranker_scores': None,
  }

  if args.agent == 'transporter':
    started = time.perf_counter()
    action = agent.act(obs)
    diagnostics['transporter_latency_s'] = time.perf_counter() - started
    move = annotate_action(
        env,
        action,
        HANOI_BOUNDS,
        HANOI_PIX_SIZE,
        score=1.0,
        source_name='transporter')
    return action, [move], move, diagnostics

  if args.agent == 'vlm-transporter':
    if args.candidate_mode == 'legal':
      _, candidates, latency, coverage = score_legal_hanoi_moves(
          agent,
          env,
          obs,
          pick_radius=args.pick_radius,
          place_radius=args.place_radius)
      diagnostics['candidate_coverage'] = coverage
      diagnostics['transporter_latency_s'] = latency
    else:
      _, candidates, latency = extract_transporter_candidates(
          agent,
          env,
          obs,
          top_pick_k=args.top_pick_k,
          top_place_k=args.top_place_k,
          max_candidates=args.max_candidates)
      diagnostics['candidate_coverage'] = float(
          any(candidate.matches_oracle for candidate in candidates))
      diagnostics['transporter_latency_s'] = latency

    current_scene_image_path = os.path.join(
        history_dir, f'scene-ep{episode_index:03d}-step{step_index:02d}.png')
    cv2.imwrite(current_scene_image_path, render_scene_image(obs))
    history_image_paths = (
        list(scene_history_paths[-max(args.history_images, 0):])
        if args.history_images > 0 else [])
    diagnostics['current_scene_image_path'] = current_scene_image_path
    diagnostics['history_image_paths'] = history_image_paths
    diagnostics['history_images_used'] = len(history_image_paths)

    request_image_path = os.path.join(
        request_dir, f'request-ep{episode_index:03d}-step{step_index:02d}.png')
    board = render_candidate_board(
        obs,
        env,
        candidates,
        selected=None,
        footer_lines=[
            'Goal: move all disks to the right peg.',
            f'Current symbolic state has {len(candidates)} legal moves.',
        ],
        show_scores=False)
    cv2.imwrite(request_image_path, board)
    diagnostics['request_image_path'] = request_image_path

    vlm_started = time.perf_counter()
    try:
      response = choose_vlm_candidate(
          worker,
          request_image_path,
          history_image_paths,
          candidates,
          peg_state=format_peg_state(env),
          move_history=move_history,
          prompt_examples=prompt_examples,
          oracle_move=oracle_move,
          include_oracle_in_prompt=args.include_oracle_in_prompt)
    except (RuntimeError, TimeoutError) as exc:
      diagnostics['worker_error'] = f'{type(exc).__name__}: {exc}'
      if worker is not None:
        worker.close()
      response = {
          'candidate_index': -1,
          'valid': False,
          'parse_success': False,
          'backend': 'worker-fallback',
          'error_type': type(exc).__name__,
          'error_msg': str(exc)[:400],
          'rationale': f'Fallback after worker failure: {type(exc).__name__}: {exc}',
      }
    diagnostics['vlm_latency_s'] = float(
        response.get('latency_s', time.perf_counter() - vlm_started))
    diagnostics['parse_success'] = float(bool(response.get('parse_success', False)))
    diagnostics['invalid_response'] = float(not bool(response.get('valid', True)))
    diagnostics['vlm_response'] = response
    diagnostics['vlm_backend'] = str(response.get('backend', ''))

    candidate_index = int(response.get('candidate_index', -1))
    if not (0 <= candidate_index < len(candidates)) or not bool(
        response.get('valid', True)):
      vlm_selected = choose_fallback_candidate(candidates)
      vlm_selected.fallback_used = True
      vlm_selected.rationale = str(
          response.get('rationale', 'Fallback to highest-scoring legal move.'))
      diagnostics['fallback_used'] = True
    else:
      vlm_selected = candidates[candidate_index]
      vlm_selected.rationale = str(response.get('rationale', ''))

    diagnostics['vlm_selected_move'] = vlm_selected
    diagnostics['oracle_agreement'] = float(vlm_selected.matches_oracle)
    selected = vlm_selected
    if args.decision_policy == 'guarded' and args.candidate_mode == 'legal' \
        and oracle_move is not None:
      planner_move = find_symbolic_move(
          candidates, oracle_move['disk_id'], oracle_move['target_peg'])
      diagnostics['planner_move'] = planner_move
      if planner_move is None:
        selected.rationale = (
            'Guarded policy could not recover the oracle move from the legal '
            f'candidate set. {selected.rationale}')
      elif not vlm_selected.matches_oracle:
        selected = planner_move
        selected.rationale = (
            f'Planner override after VLM chose {vlm_selected.description()}. '
            f'{vlm_selected.rationale}')
        diagnostics['planner_override'] = True

    diagnostics['executed_oracle_agreement'] = float(selected.matches_oracle)

    if args.candidate_mode == 'legal':
      action = build_hanoi_execution_action(env, selected)
    else:
      action = action_from_move(agent, obs, selected)
    return action, candidates, selected, diagnostics

  if args.agent == 'reranker-transporter':
    if args.candidate_mode != 'legal':
      raise ValueError(
          'reranker-transporter currently requires --candidate_mode=legal.')
    if reranker_policy is None:
      raise ValueError(
          '--reranker_dir is required when --agent=reranker-transporter.')
    _, candidates, latency, coverage = score_legal_hanoi_moves(
        agent,
        env,
        obs,
        pick_radius=args.pick_radius,
        place_radius=args.place_radius)
    diagnostics['candidate_coverage'] = coverage
    diagnostics['transporter_latency_s'] = latency
    selected, logits, reranker_latency = reranker_policy.select_candidate(
        env, candidates, step_index)
    diagnostics['reranker_latency_s'] = reranker_latency
    diagnostics['reranker_scores'] = [float(v) for v in logits]
    diagnostics['oracle_agreement'] = float(selected.matches_oracle)
    diagnostics['executed_oracle_agreement'] = float(selected.matches_oracle)
    action = build_hanoi_execution_action(env, selected)
    return action, candidates, selected, diagnostics

  if args.agent == 'gt_state':
    legacy_action = agent.act(obs, env.info)
    action = normalize_env_action(legacy_action)
    move = annotate_action(
        env,
        action,
        HANOI_BOUNDS,
        HANOI_PIX_SIZE,
        score=1.0,
        source_name='gt_state')
    return action, [move], move, diagnostics

  if args.agent == 'oracle':
    oracle_agent = env.task.oracle(env)
    action = oracle_agent.act(obs, env.info)
    move = annotate_action(
        env,
        action,
        HANOI_BOUNDS,
        HANOI_PIX_SIZE,
        score=1.0,
        source_name='oracle')
    return action, [move], move, diagnostics

  raise ValueError(f'Unsupported agent: {args.agent}')


def evaluate(args):
  configure_tensorflow(args.gpu)
  ensure_dir(args.output_dir)
  request_dir = ensure_dir(os.path.join(args.output_dir, 'vlm-requests'))
  history_dir = ensure_dir(os.path.join(args.output_dir, 'vlm-scene-history'))

  if args.agent == 'vlm-transporter' and args.decision_policy == 'guarded' \
      and args.candidate_mode != 'legal':
    raise ValueError(
        '--decision_policy=guarded currently requires --candidate_mode=legal.')
  if args.agent == 'reranker-transporter' and args.candidate_mode != 'legal':
    raise ValueError(
        '--agent=reranker-transporter currently requires '
        '--candidate_mode=legal.')

  worker = None
  reranker_policy = None
  selected_transporter = {}
  prompt_examples = []
  if args.agent == 'vlm-transporter':
    if not args.vlm_command.strip():
      raise ValueError(
          '--vlm_command is required when --agent=vlm-transporter.')
    worker = VLMWorkerClient(
        args.vlm_command, cwd=args.root_dir, timeout_s=args.vlm_timeout_s)
    worker.start()
    prompt_examples = collect_oracle_prompt_examples(
        args.assets_root,
        args.data_dir,
        args.task,
        limit=args.num_prompt_exemplars)
    print(f'Loaded {len(prompt_examples)} oracle prompt exemplars from train seeds.')
  elif args.agent == 'reranker-transporter':
    if not args.reranker_dir.strip():
      raise ValueError(
          '--reranker_dir is required when --agent=reranker-transporter.')
    reranker_policy = HanoiRerankerPolicy.from_dir(args.reranker_dir)
    selected_transporter = reranker_policy.selected_transporter_spec()
    if not selected_transporter:
      raise ValueError(
          f'No selected transporter metadata found in {args.reranker_dir}.')
    print(
        'Loaded reranker policy from '
        f'{args.reranker_dir} using transporter '
        f'{selected_transporter["name"]} step {selected_transporter["step"]}.')

  env = Environment(
      args.assets_root,
      disp=args.disp,
      shared_memory=args.shared_memory,
      hz=480)
  task = tasks.names[args.task]()
  task.mode = 'test'
  env.set_task(task)

  ds = Dataset(os.path.join(args.data_dir, f'{args.task}-test'))
  num_episodes = min(args.episodes, ds.n_episodes)
  result_name = build_result_name(args, selected_transporter)
  model_name = build_model_name(args)

  if args.agent in ('transporter', 'vlm-transporter'):
    agent = agents.names['transporter'](model_name, args.task, args.root_dir)
    if args.n_steps > 0:
      agent.load(args.n_steps)
  elif args.agent == 'reranker-transporter':
    agent = agents.names['transporter'](
        selected_transporter['name'], args.task, args.root_dir)
    agent.load(int(selected_transporter['step']))
  elif args.agent == 'gt_state':
    agent = train_or_load_gt_state(args, model_name)
  else:
    agent = None

  results: List[Dict[str, Any]] = []
  recorder = None

  try:
    for episode_index in range(num_episodes):
      print(f'Episode {episode_index + 1}/{num_episodes}')
      _, seed = ds.load(episode_index, images=False)
      np.random.seed(seed)
      env.seed(seed)
      env.set_task(task)
      obs = env.reset()
      info = env.info
      total_reward = 0.0
      step_logs = []
      move_history = []
      scene_history_paths = []
      seen_states = {format_peg_state(env)}
      legal_moves = 0
      oracle_matches = 0
      executed_oracle_matches = 0
      fallback_count = 0
      parse_success_count = 0
      invalid_response_count = 0
      legal_candidate_coverage_count = 0
      candidate_scoring_steps = 0
      planner_override_count = 0
      repeat_state_count = 0
      history_images_used_total = 0
      transporter_latency = 0.0
      reranker_latency = 0.0
      vlm_latency = 0.0
      total_latency = 0.0
      vlm_steps = 0
      vlm_backend = ''

      if args.record_rollout and episode_index == args.record_episode_index:
        recorder = RolloutRecorder(
            os.path.join(args.output_dir, 'rollouts'),
            episode_index,
            enabled=True)

      for step_index in range(task.max_steps):
        step_started = time.perf_counter()
        action, candidates, selected, diagnostics = select_action_for_step(
            args,
            episode_index,
            step_index,
            env,
            obs,
            info,
            agent,
            reranker_policy,
            worker,
            request_dir,
            history_dir,
            scene_history_paths,
            move_history,
            prompt_examples)

        transporter_latency += diagnostics['transporter_latency_s']
        reranker_latency += diagnostics['reranker_latency_s']
        vlm_latency += diagnostics['vlm_latency_s']
        if args.agent == 'vlm-transporter':
          vlm_steps += 1
          fallback_count += int(diagnostics['fallback_used'])
          parse_success_count += int(diagnostics['parse_success'])
          invalid_response_count += int(diagnostics['invalid_response'])
          planner_override_count += int(diagnostics['planner_override'])
          history_images_used_total += int(diagnostics['history_images_used'])
          if diagnostics['vlm_backend'] and not vlm_backend:
            vlm_backend = diagnostics['vlm_backend']
        if args.agent in ('vlm-transporter', 'reranker-transporter'):
          candidate_scoring_steps += 1
          legal_candidate_coverage_count += int(diagnostics['candidate_coverage'])

        oracle_move = get_oracle_move(env)
        footer_lines = [
            f'Agent: {args.agent}',
            f'Step: {step_index + 1}',
            f'Selected: {selected.description()}',
            f'Oracle: {oracle_move["description"] if oracle_move else "n/a"}',
            f'Peg state: {format_peg_state(env).replace(chr(10), " | ")}',
        ]
        if diagnostics['planner_override'] and diagnostics['vlm_selected_move'] \
            is not None:
          footer_lines.append(
              f'VLM chose: {diagnostics["vlm_selected_move"].description()}')
        if selected.rationale:
          footer_lines.append(f'Reason: {selected.rationale}')
        if recorder is not None:
          recorder.write(env, obs, candidates, selected, footer_lines)

        obs, reward, done, info = env.step(action)
        total_reward += reward
        step_latency = time.perf_counter() - step_started
        total_latency += step_latency
        new_peg_state = format_peg_state(env)
        repeat_state = new_peg_state in seen_states
        seen_states.add(new_peg_state)
        repeat_state_count += int(repeat_state)

        legal_moves += int(selected.legal)
        oracle_matches += int(diagnostics['oracle_agreement'])
        executed_oracle_matches += int(diagnostics['executed_oracle_agreement'])
        move_history.append({
            'description': selected.description(),
            'legal': selected.legal,
            'reward': reward,
        })
        if diagnostics['current_scene_image_path']:
          scene_history_paths.append(diagnostics['current_scene_image_path'])
          if args.history_images > 0:
            scene_history_paths = scene_history_paths[-args.history_images:]
          else:
            scene_history_paths = []

        step_logs.append({
            'step_index': step_index + 1,
            'reward': reward,
            'done': done,
            'selected_move': selected,
            'vlm_selected_move': diagnostics['vlm_selected_move'],
            'planner_move': diagnostics['planner_move'],
            'oracle_move': oracle_move,
            'candidate_count': len(candidates),
            'candidate_coverage': diagnostics['candidate_coverage'],
            'request_image_path': diagnostics['request_image_path'],
            'current_scene_image_path': diagnostics['current_scene_image_path'],
            'history_image_paths': diagnostics['history_image_paths'],
            'history_images_used': diagnostics['history_images_used'],
            'transporter_latency_s': diagnostics['transporter_latency_s'],
            'reranker_latency_s': diagnostics['reranker_latency_s'],
            'vlm_latency_s': diagnostics['vlm_latency_s'],
            'total_latency_s': step_latency,
            'vlm_parse_success': diagnostics['parse_success'],
            'vlm_invalid_response': diagnostics['invalid_response'],
            'vlm_worker_error': diagnostics['worker_error'],
            'planner_override': diagnostics['planner_override'],
            'oracle_agreement': diagnostics['oracle_agreement'],
            'executed_oracle_agreement': diagnostics['executed_oracle_agreement'],
            'repeat_state': repeat_state,
            'vlm_backend': diagnostics['vlm_backend'],
            'reranker_scores': diagnostics['reranker_scores'],
            'vlm_response': diagnostics['vlm_response'],
        })
        print(
            f'  reward={reward:.3f} done={done} legal={selected.legal} '
            f'oracle_match={bool(diagnostics["oracle_agreement"])} '
            f'override={diagnostics["planner_override"]}')
        if done:
          break

      steps_executed = len(step_logs)
      episode_result = {
          'episode_index': episode_index,
          'seed': seed,
          'total_reward': total_reward,
          'success': total_reward > 0.99,
          'steps_executed': steps_executed,
          'legal_move_rate': legal_moves / max(steps_executed, 1),
          'oracle_move_agreement': oracle_matches / max(steps_executed, 1),
          'executed_oracle_move_agreement': (
              executed_oracle_matches / max(steps_executed, 1)),
          'transporter_latency_s': transporter_latency / max(steps_executed, 1),
          'reranker_latency_s': reranker_latency / max(steps_executed, 1),
          'vlm_latency_s': vlm_latency / max(steps_executed, 1),
          'total_latency_s': total_latency / max(steps_executed, 1),
          'vlm_fallback_rate': (
              fallback_count / max(vlm_steps, 1)
              if args.agent == 'vlm-transporter' else 0.0),
          'vlm_parse_success_rate': (
              parse_success_count / max(vlm_steps, 1)
              if args.agent == 'vlm-transporter' else 0.0),
          'vlm_invalid_response_rate': (
              invalid_response_count / max(vlm_steps, 1)
              if args.agent == 'vlm-transporter' else 0.0),
          'legal_candidate_coverage': (
              legal_candidate_coverage_count / max(candidate_scoring_steps, 1)
              if args.agent in ('vlm-transporter', 'reranker-transporter')
              else 0.0),
          'planner_override_rate': (
              planner_override_count / max(vlm_steps, 1)
              if args.agent == 'vlm-transporter' else 0.0),
          'repeat_state_rate': repeat_state_count / max(steps_executed, 1),
          'mean_history_images_used': (
              history_images_used_total / max(vlm_steps, 1)
              if args.agent == 'vlm-transporter' else 0.0),
          'vlm_backend': vlm_backend,
          'step_logs': step_logs,
      }
      results.append(episode_result)

      if recorder is not None:
        recorder.close()
        recorder = None

  finally:
    if recorder is not None:
      recorder.close()
    if worker is not None:
      worker.close()
    env.close()

  selected_n_demos, selected_run_index, selected_step = (
      _selected_transporter_run_spec(
          selected_transporter, args.n_demos, args.run_index, args.n_steps)
      if args.agent == 'reranker-transporter'
      else (args.n_demos, args.run_index, args.n_steps))

  payload = {
      'task': args.task,
      'agent': args.agent,
      'candidate_mode': args.candidate_mode,
      'decision_policy': args.decision_policy,
      'history_images': args.history_images,
      'n_demos': selected_n_demos,
      'checkpoint_step': selected_step,
      'run_index': selected_run_index,
      'reranker_dir': args.reranker_dir if args.agent == 'reranker-transporter' else '',
      'selected_transporter': selected_transporter if args.agent == 'reranker-transporter' else {},
      'vlm_backend': (
          next((episode.get('vlm_backend', '') for episode in results
                if episode.get('vlm_backend')), '')
          if args.agent == 'vlm-transporter' else ''),
      'summary': summarize_episodes(results),
      'episodes': results,
  }
  output_path = os.path.join(args.output_dir, f'{result_name}.json')
  with open(output_path, 'w', encoding='utf-8') as stream:
    json.dump(json_ready(payload), stream, indent=2)
  print(f'Saved benchmark results to {output_path}')


def main():
  evaluate(parse_args())


if __name__ == '__main__':
  main()
