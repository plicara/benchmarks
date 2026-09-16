# Jev evidence mapping onto frozen AdventureBench schemas (2026-09-16)

Goal: publishable Jev Pareto point without touching frozen v1 semantics.
Principle: no schema-file changes; values carry the runtime distinction;
chat replay path byte-identical.

## Prompt identity
- Chat: `prompt_sha256 = sha256(SYSTEM_PROMPT)`. Jev: `prompt_sha256 = sha256(JEV_SPEC)`
  where JEV_SPEC = canonical JSON of {action_criteria, action_instructions,
  target_instructions, threshold} from `jev.py` (threshold 0.9 frozen IN the spec;
  any threshold change = new spec = incompatible release, by construction).
- rescore.py must accept both hashes (3 check sites: manifest benchmark, record
  request, aggregate same-hash rule) and branch replay by hash. Chat branch untouched.

## Attempts (schema already fits: raw_completion string|null, parsed outcome|null, 1-2 attempts)
- Jev: exactly 1 attempt per case/repetition (no retry; a repeat call adds nothing).
- `raw_completion` = JSON dump of the Jev answers object (choice, probabilities,
  confidence per question, usage, model). `parsed` = {action, target} outcome.
- Unknown action value (shouldn't happen; typed API) = parsing error, outcome unclear.
- Transport failure shape identical to chat. Threshold flip is NOT an error:
  outcome unclear with `flipped_to_unclear` noted in raw JSON only.

## Replay (rescore `_replay` branch by prompt hash)
- Parse raw_completion JSON -> answers -> outcome via the same mapping as `run_case_jev`
  (NO_TARGET rule, DIRECTIONS normalization, threshold from request config).
- `passed` recomputation unchanged (outcome in expected_outcomes).

## Cost (generic already: sum of per-record cost)
- Jev collector computes per-record USD = input_tokens / 1e9 * 42 (outputs free).
  Expected full 3-rep run ≈ $0.03.

## Latency (NO derived-schema change: modelResult is additionalProperties:false)
- Evidence already records per-attempt and per-record `latency_ms` (generic).
- Latency Pareto is computed at RENDER time from evidence runs/ JSONL
  (median per model) + release scores. rescore/aggregate untouched for latency.

## Manifest / collect
- `collect_run` is generic except prompt hash + `complete()` shape: add
  `collect_run_jev` reusing it via parameters (prompt_sha, request template,
  JevClient adapter), or a thin Jev entry in `adventure-bench-collect --jev`.
- resolved provider `typesafe` (validity needs exactly one); resolved model per
  record (expect jev-1.13.0); runtime string `typesafe-systemone:<requested>`.
- Release membership: aggregate demands same dataset+prompt hash, unique slugs,
  >=2 run IDs. Jev release = two Jev run IDs with slugs `jev-latest` (alias) and
  `jev-1-13-0` (pinned) — doubles as an alias-stability check; label as such.
  Chat runs can never join a Jev release (prompt hash differs) and vice versa.

## Out of scope
- result-schema.json / evidence-schema.json version bumps (not needed).
- run_spec staged plans (chat-release tooling; Jev release assembled directly).
- New-set (2595) release mechanics (future v1.x work).
