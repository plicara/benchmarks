# Jev evidence mapping onto frozen AdventureBench schemas

Goal: publishable Jev Pareto point without touching frozen v1 semantics.

Principle: no schema-file changes; values carry the runtime distinction; chat replay path byte-identical.

## Prompt identity

- Chat uses `prompt_sha256 = sha256(SYSTEM_PROMPT)`. Jev uses the SHA-256 of the canonical sorted JSON returned by `jev_spec()`.
- Jev v1 (`7d98e814…`, released 2026-09-16) freezes a 0.9 confidence gate.
- Jev v2 (`a5df020f…`, released 2026-09-17) freezes a 0.75 gate for mapping tags, a 0.9 gate for calibration tags, missing-target-to-unclear routing, and an explicit relative-direction rule. These changes were selected on synthetic dev data only.
- A changed Jev spec is a distinct prompt and release. The replay path dispatches on the recorded hash, preserving the v1 mapping exactly rather than rescoring historic evidence with v2 rules.
- `rescore.py` requires the manifest and collection prompt hashes to agree, requires each record request to equal that collection configuration, and rejects any unknown prompt hash. The chat branch remains untouched.

## Attempts (schema already fits: raw_completion string|null, parsed outcome|null)
- Jev: exactly 1 attempt per case/repetition (no retry; a repeat call adds nothing).
- `raw_completion` = JSON dump of the Jev answers object (choice, probabilities, confidence per question, usage, model). `parsed` = {action, target} outcome.
- Unknown action value (shouldn't happen; typed API) = parsing error, outcome unclear.
- Transport failure shape is identical to chat. A threshold flip is not an error: outcome unclear with `flipped_to_unclear` noted in raw JSON only.

## Replay (rescore `_replay` branch by prompt hash)
- Parse raw_completion JSON -> answers -> outcome via the recorded v1 or v2 mapping (no-target handling, direction normalization, and frozen gates).
- `passed` recomputation unchanged (outcome in expected_outcomes).

## Cost (generic already: sum of per-record cost)
- Jev collector computes per-record USD = input_tokens / 1e9 * 42 (outputs free). Expected full 3-rep run ≈ $0.03.

## Latency (NO derived-schema change: modelResult is additionalProperties:false)
- Evidence already records per-attempt and per-record `latency_ms` (generic).
- Latency Pareto is computed at render time from evidence `runs/` JSONL (median per model) plus release scores. `rescore` and `aggregate` are untouched for latency.

## Manifest / collect
- `collect_run_jev` writes the same manifest and evidence schema as chat while using a Jev request template and `JevClient` adapter.
- Resolved provider is `typesafe` (validity needs exactly one); resolved model is recorded per response (expect `jev-1.13.0`); requested and resolved runtime provenance must be `typesafe-systemone:<requested-model>`.
- Release membership requires the same dataset and prompt hash, unique slugs, and at least two run IDs. The Jev release has `jev-latest` (alias) and `jev-1-13-0` (pinned), so it doubles as an alias-stability check and is labelled as such. Chat runs cannot join a Jev release because their prompt hash differs.

## Out of scope
- result-schema.json / evidence-schema.json version bumps (not needed).
- run_spec staged plans (chat-release tooling; Jev release assembled directly).
- New-set (2595) release mechanics (future v1.x work).
