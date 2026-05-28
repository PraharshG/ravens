# coding=utf-8
"""Persistent line-delimited JSON worker for SmolVLM candidate selection."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, List


PROMPT_TEMPLATE = """Choose the next Towers of Hanoi move.

Goal: move the full stack to the right peg.

Image instructions:
- The first image is the CURRENT numbered candidate board.
- Any additional images are older unnumbered scene snapshots for context only.
- Only use candidate numbers from the first image.

Current peg state:
{peg_state}

Recent move history:
{move_history}

Scene history:
{history_images}

Reference hints:
{prompt_examples}

Available moves:
{candidates}

Rules:
- All listed moves are legal.
- Ignore confidence scores in the image; choose the move that is actually correct for Hanoi.
- Return exactly one JSON object on one line:
{{"candidate_index": <number>, "valid": true, "rationale": "<short reason>"}}
"""

DEFAULT_MODEL_IDS = {
    'smolvlm': 'HuggingFaceTB/SmolVLM-256M-Instruct',
    'hf-chat': 'Qwen/Qwen2.5-VL-7B-Instruct',
}


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--backend',
                      default=os.environ.get('HANOI_VLM_BACKEND', 'smolvlm'),
                      choices=['smolvlm', 'hf-chat', 'heuristic'])
  parser.add_argument('--model_id', default='')
  parser.add_argument('--device', default='auto')
  parser.add_argument('--max_new_tokens', type=int, default=96)
  return parser.parse_args()


def heuristic_select(request: Dict[str, Any]) -> Dict[str, Any]:
  """Select the highest-scoring candidate."""
  candidates = request['candidates']
  best = max(candidates, key=lambda item: float(item.get('score', 0.0)))
  return {
      'candidate_index': int(best['index']),
      'valid': True,
      'parse_success': True,
      'backend': 'heuristic',
      'rationale': (
          f'Heuristic selected #{best["index"]}: '
          f'{best.get("description", "unknown move")}'
      ),
  }


def resolve_model_id(args) -> str:
  if args.model_id.strip():
    return args.model_id.strip()
  env_model = os.environ.get('HANOI_VLM_MODEL', '').strip()
  if env_model:
    return env_model
  return DEFAULT_MODEL_IDS.get(args.backend, DEFAULT_MODEL_IDS['smolvlm'])


def build_prompt(request: Dict[str, Any]) -> str:
  candidate_lines = []
  for candidate in request['candidates']:
    candidate_lines.append(f"{candidate['index']}: {candidate['description']}")

  examples = request.get('prompt_examples', [])
  if examples:
    example_lines = []
    for idx, example in enumerate(examples, start=1):
      example_lines.append(
          f"- Example {idx}: when {example['peg_state'].replace(chr(10), '; ')}, "
          f"a good move is {example['selected_description']}.")
    prompt_examples = '\n'.join(example_lines)
  else:
    prompt_examples = '- No extra examples.'

  move_history = request.get('move_history', [])
  if move_history:
    history_lines = []
    for move in move_history[-3:]:
      history_lines.append(
          f"- {move.get('description', 'unknown move')} "
          f"(reward={move.get('reward', 0.0):.3f})")
    history_text = '\n'.join(history_lines)
  else:
    history_text = '(no previous moves)'

  history_image_paths = request.get('history_image_paths', [])
  if history_image_paths:
    history_images = (
        f'{len(history_image_paths)} previous raw scene image(s) are attached '
        'in chronological order from oldest to newest.')
  else:
    history_images = 'No previous raw scene images are attached.'

  return PROMPT_TEMPLATE.format(
      peg_state=request.get('peg_state', 'unknown'),
      move_history=history_text,
      history_images=history_images,
      prompt_examples=prompt_examples,
      candidates='\n'.join(candidate_lines))


def extract_json_robust(text: str, num_candidates: int) -> Dict[str, Any]:
  matches = list(re.finditer(r'\{.*?\}', text, re.DOTALL))
  for match in reversed(matches):
    try:
      parsed = json.loads(match.group(0))
      idx = parsed.get('candidate_index')
      if isinstance(idx, str) and idx.isdigit():
        idx = int(idx)
      if isinstance(idx, int) and 0 <= idx < num_candidates:
        parsed['candidate_index'] = idx
        parsed['valid'] = True
        parsed.setdefault('rationale', text.strip()[:160])
        return parsed
    except json.JSONDecodeError:
      continue

  number_patterns = [
      r'candidate[_ ]index["\']?\s*[:=]\s*["\']?(\d+)',
      r'\bindex\b["\']?\s*[:=]\s*["\']?(\d+)',
      r'\bchoose\b(?:\s+candidate)?\s*["\']?(\d+)',
      r'\[(\d+)\]',
      r'^\s*(\d+)\s*$',
  ]
  for pattern in number_patterns:
    matches = list(re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE))
    for match in reversed(matches):
      idx = int(match.group(1))
      if 0 <= idx < num_candidates:
        return {
            'candidate_index': idx,
            'valid': True,
            'rationale': f'Parsed candidate index {idx} from model output.',
        }

  raise ValueError(f'Could not parse a valid candidate index from: {text[:120]}')


def encode_image_as_data_url(path: str) -> str:
  mime_type, _ = mimetypes.guess_type(path)
  if mime_type is None:
    mime_type = 'image/png'
  with open(path, 'rb') as stream:
    encoded = base64.b64encode(stream.read()).decode('utf-8')
  return f'data:{mime_type};base64,{encoded}'


class ModelResponseParseError(ValueError):
  """Raised when a model response cannot be mapped to a candidate."""

  def __init__(self, message: str, raw_output: str):
    super().__init__(message)
    self.raw_output = raw_output


class SmolVLMBackend:
  """Minimal SmolVLM inference wrapper."""

  def __init__(self, model_id: str, device: str, max_new_tokens: int):
    import torch
    from PIL import Image
    from transformers import AutoProcessor
    try:
      from transformers import AutoModelForVision2Seq as AutoVisionModel
    except ImportError:
      from transformers import AutoModelForImageTextToText as AutoVisionModel

    self.torch = torch
    self.Image = Image
    self.device = self._resolve_device(device)
    self.max_new_tokens = max_new_tokens
    self.processor = AutoProcessor.from_pretrained(model_id)
    self.max_image_side = self._resolve_max_image_side()

    kwargs = {
        'torch_dtype': (torch.bfloat16
                        if self.device != 'cpu' else torch.float32),
    }
    if self.device != 'cpu':
      kwargs['_attn_implementation'] = 'flash_attention_2'
    self.model = AutoVisionModel.from_pretrained(model_id,
                                                 **kwargs).to(self.device)

  def _resolve_max_image_side(self) -> int:
    image_processor = getattr(self.processor, 'image_processor', None)
    if image_processor is None:
      return 512
    max_image_size = getattr(image_processor, 'max_image_size', None)
    if isinstance(max_image_size, dict):
      longest = max_image_size.get('longest_edge')
      if longest is not None:
        return int(longest)
    if isinstance(max_image_size, int):
      return int(max_image_size)
    return 512

  def _prepare_image(self, image):
    width, height = image.size
    longest = max(width, height)
    if longest <= self.max_image_side:
      return image
    scale = self.max_image_side / float(longest)
    new_size = (max(1, int(round(width * scale))),
                max(1, int(round(height * scale))))
    resampling = getattr(self.Image, 'Resampling', self.Image)
    return image.resize(new_size, resampling.BICUBIC)

  def _resolve_device(self, device: str) -> str:
    if device != 'auto':
      return device
    return 'cuda' if self.torch.cuda.is_available() else 'cpu'

  def _decode_generated_text(self, generated_ids, inputs) -> str:
    input_ids = inputs.get('input_ids')
    if input_ids is None:
      return self.processor.batch_decode(
          generated_ids, skip_special_tokens=True)[0]

    prompt_length = int(input_ids.shape[1])
    sequence = generated_ids[0]
    if sequence.shape[0] > prompt_length:
      suffix_ids = sequence[prompt_length:]
      suffix_text = self.processor.batch_decode(
          suffix_ids[None, :], skip_special_tokens=True)[0]
      if suffix_text.strip():
        return suffix_text

    return self.processor.batch_decode(
        generated_ids, skip_special_tokens=True)[0]

  def select(self, request: Dict[str, Any]) -> Dict[str, Any]:
    current_image_path = (request.get('current_request_image_path')
                          or request['image_path'])
    images = [self._prepare_image(
        self.Image.open(current_image_path).convert('RGB'))]
    for history_path in request.get('history_image_paths', []):
      images.append(self._prepare_image(
          self.Image.open(history_path).convert('RGB')))
    prompt = build_prompt(request)
    content = [{'type': 'image'} for _ in images]
    content.append({'type': 'text', 'text': prompt})
    messages = [{
        'role': 'user',
        'content': content,
    }]
    rendered_prompt = self.processor.apply_chat_template(
        messages, add_generation_prompt=True)
    inputs = self.processor(
        text=rendered_prompt,
        images=images,
        return_tensors='pt',
        do_resize=False)
    inputs = inputs.to(self.device)

    generated_ids = self.model.generate(
        **inputs,
        max_new_tokens=self.max_new_tokens,
        do_sample=False)
    generated_text = self._decode_generated_text(generated_ids, inputs)
    try:
      response = extract_json_robust(generated_text, len(request['candidates']))
    except ValueError as exc:
      raise ModelResponseParseError(str(exc), generated_text) from exc
    response.setdefault('valid', True)
    response.setdefault('rationale', generated_text.strip()[:160])
    response['parse_success'] = True
    response['backend'] = 'smolvlm'
    response['raw_output'] = generated_text.strip()[:500]
    return response


class HFChatBackend:
  """Hosted Hugging Face chat-completions backend for open-source VLMs."""

  def __init__(self, model_id: str, max_new_tokens: int):
    self.model_id = model_id
    self.max_new_tokens = max_new_tokens
    self.api_url = os.environ.get(
        'HF_CHAT_API_URL', 'https://router.huggingface.co/v1/chat/completions')
    self.api_key = os.environ.get('HF_TOKEN', '').strip()
    self.request_timeout_s = float(os.environ.get('HF_REQUEST_TIMEOUT_S', '60'))
    if not self.api_key:
      raise ValueError(
          'HF_TOKEN is required for --backend=hf-chat. Create a fine-grained '
          'Hugging Face token with Inference Providers permission.')

  def _extract_message_text(self, payload: Dict[str, Any]) -> str:
    choices = payload.get('choices', [])
    if not choices:
      raise ModelResponseParseError('Missing choices in HF response.',
                                    json.dumps(payload)[:500])
    message = choices[0].get('message', {})
    content = message.get('content', '')
    if isinstance(content, str):
      return content
    if isinstance(content, list):
      text_parts = []
      for part in content:
        if isinstance(part, dict) and part.get('type') == 'text':
          text_parts.append(str(part.get('text', '')))
      return '\n'.join(part for part in text_parts if part)
    return str(content)

  def _build_messages(self, request: Dict[str, Any]) -> List[Dict[str, Any]]:
    current_image_path = (request.get('current_request_image_path')
                          or request['image_path'])
    content: List[Dict[str, Any]] = [{
        'type': 'image_url',
        'image_url': {'url': encode_image_as_data_url(current_image_path)},
    }]
    for history_path in request.get('history_image_paths', []):
      content.append({
          'type': 'image_url',
          'image_url': {'url': encode_image_as_data_url(history_path)},
      })
    content.append({'type': 'text', 'text': build_prompt(request)})
    return [{'role': 'user', 'content': content}]

  def select(self, request: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        'model': self.model_id,
        'messages': self._build_messages(request),
        'max_tokens': self.max_new_tokens,
        'temperature': 0,
        'stream': False,
    }
    http_request = urllib.request.Request(
        self.api_url,
        data=json.dumps(payload).encode('utf-8'),
        headers={
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        },
        method='POST')
    try:
      with urllib.request.urlopen(
          http_request, timeout=self.request_timeout_s) as response:
        raw_response = response.read().decode('utf-8')
    except urllib.error.HTTPError as exc:
      error_body = exc.read().decode('utf-8', errors='replace')
      raise RuntimeError(
          f'HF chat request failed with HTTP {exc.code}: {error_body[:400]}'
      ) from exc
    except urllib.error.URLError as exc:
      raise RuntimeError(f'HF chat request failed: {exc}') from exc

    response_json = json.loads(raw_response)
    generated_text = self._extract_message_text(response_json)
    try:
      response = extract_json_robust(generated_text, len(request['candidates']))
    except ValueError as exc:
      raise ModelResponseParseError(str(exc), generated_text) from exc
    response.setdefault('valid', True)
    response.setdefault('rationale', generated_text.strip()[:160])
    response['parse_success'] = True
    response['backend'] = 'hf-chat'
    response['model_id'] = self.model_id
    response['raw_output'] = generated_text.strip()[:500]
    return response


def main():
  args = parse_args()
  backend = None
  model_id = resolve_model_id(args)
  if args.backend == 'smolvlm':
    backend = SmolVLMBackend(model_id, args.device, args.max_new_tokens)
  elif args.backend == 'hf-chat':
    backend = HFChatBackend(model_id, args.max_new_tokens)

  for line in sys.stdin:
    raw = line.strip()
    if not raw:
      continue
    request = json.loads(raw)
    if request.get('type') == 'shutdown':
      break
    try:
      if args.backend == 'heuristic':
        response = heuristic_select(request)
      else:
        response = backend.select(request)
    except Exception as exc:  # pylint: disable=broad-except
      response = heuristic_select(request)
      response['valid'] = False
      response['parse_success'] = False
      response['backend'] = args.backend
      response['error_type'] = type(exc).__name__
      response['error_msg'] = str(exc)[:400]
      raw_output = getattr(exc, 'raw_output', '')
      if raw_output:
        response['raw_output'] = raw_output.strip()[:500]
      response['rationale'] = f'Fallback after VLM error: {exc}'
    sys.stdout.write(json.dumps(response) + '\n')
    sys.stdout.flush()


if __name__ == '__main__':
  main()
