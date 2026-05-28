# Hanoi Architecture Comparison Report

## Executive Summary

- The current branch now has three distinct Hanoi evaluation modes worth comparing:
  - base `transporter`
  - older `vlm-transporter`
  - current `reranker-transporter`
- The newest architecture is the two-stage reranker path:
  - transporter proposes and scores legal symbolic moves
  - a small trained MLP reranks those candidates
- On the fresh like-for-like smoke slice run during this turn, `reranker-transporter` matched the older guarded VLM architecture on success and oracle agreement, while avoiding a live VLM worker and reducing selector overhead.
- Historical branch artifacts still matter because they explain why the reranker was added:
  - earlier plain transporter results did not solve the full task reliably
  - earlier VLM-over-transporter results were model-limited and often failed or fell back heavily

## Scope

This report combines two different evidence sources:

1. Historical branch artifacts already present in the workspace
2. Fresh directly comparable smoke runs executed on the current branch

These should not be conflated.

- Historical artifacts show how earlier branch states behaved.
- Fresh smoke runs isolate the current code paths on the same checkpoint family and test slice.

## Definitions

### Base Transporter

- `agent=transporter`
- no explicit legal-symbolic reranking stage
- chooses actions directly from transporter attention and transport heads

### Older Approach

- `agent=vlm-transporter`
- legal symbolic candidate set scored by transporter
- final choice made by a VLM worker
- optional guarded planner override

### Current Approach

- `agent=reranker-transporter`
- legal symbolic candidate set scored by transporter
- final choice made by the trained MLP reranker
- no live VLM worker required at inference

## Historical Reference Results

### Historical Base Transporter

Artifact:

- `./hanoi-results/towers-of-hanoi-transporter-100-0-1000.json`

Summary:

- episodes: `100`
- success rate: `0.0`
- mean reward: `0.0029`
- legal move rate: `0.8169`
- oracle move agreement: `0.7655`
- mean steps: `11.84`

Interpretation:

- The raw transporter could often produce partly sensible moves but still failed to finish full episodes in this older branch result.

### Historical Older VLM-Transporter

Artifact:

- `./hanoi-results/towers-of-hanoi-vlm-transporter-100-0-1000.json`

Summary:

- episodes: `100`
- success rate: `0.0`
- mean reward: `0.0129`
- legal move rate: `0.8007`
- oracle move agreement: `0.7327`
- mean steps: `11.87`
- fallback rate: `1.0`

Interpretation:

- The older VLM path did not solve the task in this historical run and was effectively relying on fallback behavior.

### Historical “Improved” VLM Snapshot

Artifact:

- `./hanoi-results-improved/towers-of-hanoi-vlm-transporter-10-0-1000.json`

Summary:

- episodes: `1`
- success rate: `0.0`
- mean reward: `0.4286`
- legal move rate: `0.4444`
- oracle move agreement: `0.3333`
- mean steps: `9.0`

Interpretation:

- This shows that branch-side structural improvements alone were not enough to produce reliable full-episode performance before the later guarded and reranker work.

## Fresh Like-for-Like Smoke Comparison

These runs were executed on the current branch with the same checkpoint family and the same five test episodes.

Artifacts:

- base transporter:
  - `/tmp/codex-hanoi-compare-base/towers-of-hanoi-transporter-10-0-4000.json`
- older architecture:
  - `/tmp/codex-hanoi-compare-old/towers-of-hanoi-vlm-transporter-legal-guarded-h2-10-0-4000.json`
- current architecture:
  - `/tmp/codex-hanoi-reranker/benchmark-smoke/towers-of-hanoi-reranker-transporter-legal-10-0-4000.json`
- aggregated comparison report:
  - `/tmp/codex-hanoi-compare-all/report/summary.json`

### Comparison Table

| Approach | Episodes | Success Rate | Mean Reward | Oracle Agreement | Executed Oracle Agreement | Legal Move Rate | Repeat State Rate | Mean Steps | Mean Total Latency (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base `transporter` | 5 | 0.6 | 0.7429 | 0.0 | 0.0 | 0.8714 | 0.2 | 9.8 | 2.8155 |
| Older `vlm-transporter` | 5 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 7.0 | 5.1444 |
| Current `reranker-transporter` | 5 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 7.0 | 4.3625 |

### Important Note About the Older Smoke Run

The fresh older-architecture smoke run used:

- `--backend=heuristic`
- `--decision_policy=guarded`

This was intentional. It isolates the older architecture without letting external VLM quality dominate the result.

It should not be read as a claim that local SmolVLM or any hosted VLM necessarily matches those numbers.

## What Changed Between the Older and Current Architectures

### Older VLM-Transporter

- needed a worker process
- needed image/request formatting for every step
- depended on response parsing
- could require planner overrides or fallbacks when the selector was weak
- carried higher per-step latency

### Current Reranker-Transporter

- reuses the transporter as a legal-candidate generator
- replaces the live VLM selector with a trained symbolic reranker
- has effectively zero selector overhead compared with transporter inference
- avoids prompt parsing, model API calls, and worker failure modes
- is easier to evaluate deterministically on CPU

## Why the Current Architecture Is the Better Default

- It preserves the symbolic/legal move formulation that fixed the earlier physics failure mode.
- It removes the highest-variance component from inference: the live VLM chooser.
- It matches perfect oracle agreement on the fresh smoke comparison slice.
- It is faster than the older guarded VLM architecture on the same slice.
- It does not depend on HF tokens, hosted API availability, or a separate SmolVLM runtime.

## Recommendation for Full Comparison Runs

For the proper full 100-episode comparison:

1. Use [hanoi_reranker_full_runbook.md](/home/praharsh/perception/ravens/docs/hanoi_reranker_full_runbook.md).
2. Run all three benchmark families into one output directory.
3. Keep two comparison views:
   - architecture-only: old VLM path with `--backend=heuristic`
   - real-model: old VLM path with `hf-chat` or `smolvlm`
4. Treat `reranker-transporter` as the default branch winner if it:
   - matches or beats full-episode success
   - matches or beats executed oracle agreement
   - lowers mean total latency
   - avoids planner overrides and worker dependencies

## Current Conclusion

- Historical branch evidence shows that plain transporter and early VLM-over-transporter results were not enough.
- Fresh current-branch smoke evidence shows that the reranker closes the selection gap cleanly.
- The runbook above is the correct path to regenerate the full comparison and promote the latest architecture from smoke validation to full reported results.
