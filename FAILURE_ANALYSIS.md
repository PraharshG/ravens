# VLM Worker Implementation & Benchmark Failure Analysis

## Executive Summary

The VLM worker implementation uses a line-delimited JSON protocol over stdin/stdout. The main failure point is **unhandled exceptions when the VLM worker process exits**, which causes the entire benchmark to crash. There is no error recovery mechanism in the benchmark's main loop.

---

## 1. VLM Worker Implementation (`hanoi_vlm_worker.py`)

### Overview
The worker operates as a persistent subprocess that reads JSON requests from stdin and writes JSON responses to stdout.

### Architecture
- **Main Loop**: [Lines 237-259]
  - Reads line-delimited JSON from `sys.stdin`
  - Processes each request based on `backend` type
  - Writes JSON response to `sys.stdout` with `sys.stdout.flush()`
  - Handles shutdown via `{'type': 'shutdown'}` message

### Two Backends

#### 1. Heuristic Backend [Lines 41-52]
```python
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
```
- Fast, simple selection of highest-scoring candidate
- Never fails (unless candidates list is empty)

#### 2. SmolVLM Backend [Lines 54-232]
```python
class SmolVLMBackend:
```
- Loads pretrained SmolVLM-256M-Instruct model
- Accepts image path and structured prompt
- Uses vision-to-sequence model for inference
- **Critical**: Imports `torch`, `transformers`, PIL at initialization

### Error Handling in Main Loop [Lines 237-259]
```python
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
  response['error_msg'] = str(exc)[:200]
  response['rationale'] = f'Fallback after VLM error: {exc}'
sys.stdout.write(json.dumps(response) + '\n')
sys.stdout.flush()
```

**Key Point**: Catches all exceptions and returns fallback response with error details. Worker does NOT exit on error; it continues processing requests.

---

## 2. VLM Worker Client (`hanoi_utils.py` - VLMWorkerClient class)

### Location
[Lines 619-700 in `hanoi_utils.py`]

### Implementation

#### Startup [Lines 628-642]
```python
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
```

#### Request Handling [Lines 644-663]
```python
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
    line = self.process.stdout.readline()
    if line:
      return json.loads(line)
    if self.process.poll() is not None:  # <-- CRITICAL CHECK
      stderr = ''
      if self.process.stderr is not None:
        stderr = self.process.stderr.read()
      raise RuntimeError(
          f'VLM worker exited with code {self.process.returncode}: {stderr}')
    time.sleep(0.05)
  raise TimeoutError(f'VLM worker timed out after {self.timeout_s} seconds.')
```

### Critical Failure Points

**1. Process Exit Detection (Lines 657-661)**
```python
if self.process.poll() is not None:  # Process has exited
  stderr = ''
  if self.process.stderr is not None:
    stderr = self.process.stderr.read()
  raise RuntimeError(
      f'VLM worker exited with code {self.process.returncode}: {stderr}')
```

**Problem**: Raises `RuntimeError` with worker's exit code and stderr. This exception is NOT caught by the benchmark!

**2. Timeout (Line 664)**
```python
raise TimeoutError(f'VLM worker timed out after {self.timeout_s} seconds.')
```

**Problem**: Also not caught by benchmark. Default timeout is 120 seconds (configurable via `--vlm_timeout_s`).

---

## 3. Benchmark Error Handling (`hanoi_benchmark.py`)

### Main Try-Finally Block [Lines 364-505]
The benchmark wraps the entire episode loop in a try-finally, **BUT there is no except clause**:

```python
try:
  for episode_index in range(num_episodes):
    # ... episode loop with step iteration ...
    for step_index in range(task.max_steps):
      action, candidates, selected, diagnostics = select_action_for_step(
          args, episode_index, step_index, env, obs, info,
          agent, worker, request_dir, move_history, prompt_examples)
      # ... process step results ...
      obs, reward, done, info = env.step(action)
      # ... record step logs ...
      if done:
        break
  # ... append episode results ...
finally:
  if recorder is not None:
    recorder.close()
  if worker is not None:
    worker.close()
  env.close()
```

### The Problem

**No error handling for exceptions from `select_action_for_step()`**:
- If the worker process exits, `VLMWorkerClient.request()` raises `RuntimeError`
- If timeout occurs, raises `TimeoutError`
- These propagate through `choose_vlm_candidate()` → `select_action_for_step()` → main loop
- Benchmark crashes without saving partial results to JSON
- Only the finally block executes (cleanup)

---

## 4. Failure Mode Analysis

### Scenario: Worker Process Exits After 2 Steps

**Chain of Events**:
1. Benchmark starts with `--vlm_command="python hanoi_vlm_worker.py --backend=heuristic"`
2. First 1-2 steps work fine
3. VLM worker process encounters:
   - Segmentation fault (e.g., from torch/GPU issue)
   - Out of memory
   - Import error on attempt to load SmolVLM model
   - Signal termination (SIGKILL, SIGTERM, or user Ctrl+C = exit code 130)
   - Broken pipe when trying to write response

4. Process exits with non-zero exit code
5. `VLMWorkerClient.request()` detects exit via `process.poll()`
6. Raises `RuntimeError: VLM worker exited with code {code}: {stderr}`
7. Exception propagates uncaught through entire stack
8. Benchmark terminates
9. Partial results lost (not yet written to JSON)

### Example Exit Codes
- **130**: SIGINT (user Ctrl+C)
- **139**: SIGSEGV (segmentation fault)
- **124**: SIGTERM (timeout)
- **1**: Generic error
- **137**: SIGKILL (forceful kill)

---

## 5. Result Files Analysis

### hanoi-results-legal-smoke/towers-of-hanoi-vlm-transporter-10-0-1000.json

**Status**: Full 5-episode run completed successfully
- Episode 0: 14 steps executed
- Episode 1-4: Variable steps (8-14 range)
- Mean steps: 12.0
- Legal move rate: 1.0 (all legal)
- Oracle agreement: 0.62

**Configuration**:
- `--candidate_mode=legal` (scores only legal moves)
- `--backend=heuristic` (fast selection)
- `--n_steps=1000` (checkpoint loaded)

**Observation**: When using heuristic backend with legal-only candidates, the system completes full episodes without crashing.

### hanoi-results-smoke/towers-of-hanoi-vlm-transporter-10-0-0.json

**Status**: Only 1 episode completed (test run)
- Episode 0: 14 steps executed
- Mean steps: 14.0
- Legal move rate: 0.857
- Oracle agreement: 0.0

**Configuration**:
- `--candidate_mode=raw` (uses all candidates from transporter)
- `--backend=heuristic`
- `--n_steps=0` (untrained model)

**Observation**: Also completed successfully. Worker did not crash.

### VLM Request Images Generated

Found in `hanoi-results-legal-smoke/vlm-requests/`:
```
request-ep000-step00.png  → 14 images
request-ep001-step00.png  → 14 images
request-ep002-step00.png  → 14+ images
request-ep003-step00.png  → 14+ images
request-ep004-step01.png  → 3 images
```

**Analysis**: Requests were successfully generated and processed for multiple episodes, indicating the worker handled at least 3 full episodes.

---

## 6. Root Cause Summary

### Primary Issue
**No exception handling in benchmark main loop for VLM worker errors.**

When `VLMWorkerClient.request()` encounters:
1. Worker process exit → `RuntimeError` (uncaught)
2. Response timeout → `TimeoutError` (uncaught)

These errors crash the entire benchmark without saving partial results.

### Secondary Issues

**1. VLM Worker Fragility**
- If SmolVLM backend is enabled and model loading fails → Worker exits
- Any GPU/CUDA error → Worker exits
- Broken pipe from benchmark → Worker may crash

**2. No Worker Health Checking**
- Benchmark doesn't verify worker process is alive before sending requests
- Only detects failure during `readline()` after request is sent
- No heartbeat or keep-alive mechanism

**3. Timeout Too Strict**
- Default 120s timeout may be insufficient for:
  - First SmolVLM model inference (model loading + inference)
  - Large image processing
  - GPU memory allocation on first run

**4. VLM Worker Logging**
- Worker logs to stderr, but stderr only read on exit
- No way to track worker state during execution
- Hard to debug worker-side failures

---

## 7. Recommended Fixes

### Immediate (High Priority)

**A. Add exception handling in benchmark main loop** (`hanoi_benchmark.py` lines 400-430)
```python
for step_index in range(task.max_steps):
  try:
    action, candidates, selected, diagnostics = select_action_for_step(...)
  except (RuntimeError, TimeoutError) as e:
    print(f'ERROR in step {step_index}: {type(e).__name__}: {e}')
    # Log error and move to next episode
    break
  # ... continue with step processing ...
```

**B. Add worker health check** (`hanoi_utils.py` VLMWorkerClient)
```python
def is_alive(self) -> bool:
  if self.process is None:
    return False
  return self.process.poll() is None
```

### Medium Priority

**C. Capture worker logs** (`hanoi_vlm_worker.py`)
- Redirect stderr to file
- Log all requests/responses for debugging
- Include timestamp on each message

**D. Increase default timeout**
- First inference: 300s
- Subsequent: 120s
- Allow per-request timeout override

**E. Graceful worker restart**
- Detect worker exit in benchmark
- Attempt to restart before moving to next episode
- Track restart attempts and give up after N failures

### Low Priority

**F. Worker heartbeat**
- Periodic keep-alive messages
- Detect hung worker (no response to ping)

**G. Worker process monitoring**
- Log worker memory/CPU usage
- Detect resource exhaustion early

---

## 8. Testing Recommendations

1. **Test with SmolVLM backend**:
   - Currently only tested with `--backend=heuristic`
   - `--backend=smolvlm` likely to crash due to missing error handling

2. **Test worker crashes**:
   - Kill worker process mid-benchmark
   - Verify benchmark handles gracefully

3. **Test timeouts**:
   - Set `--vlm_timeout_s=2` to force timeout
   - Verify exception is caught

4. **Test with different n_steps**:
   - `--n_steps=0` (untrained model)
   - `--n_steps=1000` (trained model)
   - Different models may fail at different stages

---

## Code Location Summary

| Component | File | Lines |
|-----------|------|-------|
| Worker main loop | `hanoi_vlm_worker.py` | 237-259 |
| Worker error handling | `hanoi_vlm_worker.py` | 248-256 |
| SmolVLM backend | `hanoi_vlm_worker.py` | 54-232 |
| VLMWorkerClient | `hanoi_utils.py` | 619-700 |
| Worker request method | `hanoi_utils.py` | 644-663 |
| Process exit detection | `hanoi_utils.py` | 657-661 |
| Benchmark main loop | `hanoi_benchmark.py` | 364-505 |
| Try-finally structure | `hanoi_benchmark.py` | 364-505 |
| Step selection call | `hanoi_benchmark.py` | 390-397 |
