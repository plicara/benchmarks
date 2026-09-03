# Adventure Bench

**Can your model play a text adventure?** A benchmark for *grounded instruction-following*: mapping free-form player input onto a small structured action vocabulary against a described scene — including knowing when to refuse.

Every case is a real, documented human utterance (or a faithful reconstruction of a documented pattern) drawn from five decades of text-adventure play: Infocom-era walkthroughs, transcript studies of novice players, and the genre's most famous parser failures. Every case has an unambiguous ground truth checkable by code. **No LLM judge anywhere.**

```
> get ye flask
```

A 1980 parser said *"You can't get ye flask."* Your model should do better — and when the flask genuinely isn't there, it should say so instead of grabbing something else.

## Why this is a good benchmark

1. **It separates two abilities models love to blur.** *Mapping* (synonyms, paraphrase, typos, pronouns, "go to the library" → east) and *calibration* (refusing what the schema can't express, not hallucinating items that exist only in prose). Frontier models ace the first and still fail the second — the current SOTA ceiling is ~90%, and its failures are almost all calibration.
2. **Signal-heavy, cheap, deterministic.** 244 single-turn cases, ~500 tokens each, scored by exact structured comparison with engine-style noun resolution. A full run costs pennies and minutes.
3. **Grounded provenance.** Inputs come from documented sources: ~2,300 verbatim commands across 8 classic walkthroughs, Aaron Reed's 72+139-transcript novice studies, the Smarter Parser test corpus, the Starborn 23,896-turn statistics, and canonical parser-failure lore (babel fish, guess-the-verb, ye flask). See `DATASET_CARD.md`.
4. **A pattern taxonomy, not one number.** 29 tags; the per-tag table is the real output. "83%" says little — "typos 100%, out-of-vocab 62%" tells you what to fix.

## Quickstart

Zero dependencies (Python 3.10+, stdlib only). From this directory:

```sh
export ADVENTURE_BENCH_API_KEY=sk-...   # or OPENROUTER_API_KEY / OPENROUTER_KEY

uv run adventure-bench --model mistralai/ministral-3b-2512 --provider mistral
uv run adventure-bench --model z-ai/glm-5.2 --out results.json
uv run adventure-bench --tag out-of-vocab --verbose     # one pattern family
uv run adventure-bench --model qwen3.5-4b --base-url http://localhost:11434/v1   # local (Ollama etc.)

uv run python -m adventure_bench.validate    # dataset lints + stats
uv run python -m unittest discover           # offline harness tests
make check                                   # validate, replay evidence, and run tests
```

Works against any OpenAI-compatible chat endpoint. On OpenRouter, pass `--provider` to pin the hosting provider — hosts serving the same model id are **not** behaviorally equivalent (we measured one returning empty completions where another was flawless).

## Auditable result collection

Use `adventure-bench-collect` for any result intended for publication. It keeps
the original CLI compatible while writing raw completion evidence and a
provenance manifest under `runs/<run-id>/`. It preserves the frozen v1 prompt,
parser, retry, and score semantics.

```sh
uv run adventure-bench-collect \
  --model example/model \
  --provider pinned-provider \
  --runtime openai-compatible:release-id \
  --repetitions 3 \
  --run-id 20260901t120000z-example-model
```

The evidence records model/provider/runtime identity, prompt and dataset hashes,
UTC timestamps, endpoint hostname, settings, raw completion attempts, parsed
outcomes, deterministic pass/fail, latency, optional usage/cost, and structured
errors. It resumes without duplicating a completed case/repetition. Runs with
an unresolved provider or any transport failure are invalid for publication.
Credentials, authorization headers, endpoint paths, and query strings are never
written. See [the evidence schema](docs/evidence-schema.md).

To compare completed, valid one-model runs, rescore each first and create a
release set from the explicit ordered IDs. The release contains per-model and
per-tag scores plus deterministic case-clustered 95% score and paired
difference intervals; it does not assert a winner.

```sh
uv run adventure-bench-aggregate --release-id 20260901-pilot \
  --run-id 20260901t120000z-model-a \
  --run-id 20260901t121000z-model-b
```

See [the methodology](docs/methodology.md) for the release contract and
interval definition.

## Staged release collection

Before a publishable collection, write a public JSON release specification
that freezes the release ID, pinned model/provider/runtime identities,
repetitions, and a small explicit smoke subset. The specification contains no
credentials, endpoint URL, or local path. Its format is described by
[`docs/release-spec-schema.json`](docs/release-spec-schema.json).

`adventure-bench-run-plan` is intentionally a dry run unless `--execute` is
provided. It prints a stable plan with the derived smoke and full run IDs plus
the worst-case completion-call ceiling, without reading an API key or making
provider requests. Execution must acknowledge that ceiling with
`--max-completions`. During execution it
collects each model's smoke run first; an invalid smoke run stops before that
model's full run. Any invalid smoke or full run stops before later models. Both
stages use the normal resume-safe evidence collector.

```sh
uv run adventure-bench-run-plan --spec release-spec.json
uv run adventure-bench-run-plan --spec release-spec.json --execute --max-completions 2048 --max-output-tokens 128
```

Use only the full-run IDs from a completed plan when creating a release. Smoke
evidence is a guardrail, not a release member.

## Static publication render

An audited release can render a standalone technical benchmark page without
calling a provider. The renderer first reads the
release's explicit run IDs, replays every run, and requires the release JSON to
match the canonical aggregate byte-for-byte. It then writes
`site/benchmarks/adventurebench/index.html`.

```sh
make render-site RELEASE_ID=20260901-pilot
make site-check RELEASE_ID=20260901-pilot  # fail on release or HTML drift
```

The same entry point accepts `--runs-dir`, `--results-dir`, `--releases-dir`,
`--data`, and `--site-dir` for an offline publication workspace:

```sh
uv run adventure-bench-render-site --release-id 20260901-pilot --check
```

## Repository audit

`adventure-bench-audit` is the repository-wide, read-only counterpart to the
per-ID checks. It discovers every committed `runs/<run-id>/manifest.json`,
replays each against its matching derived results directory, canonical-checks
every committed release, and verifies the optional published Adventure Bench
page against exactly one audited release. It rejects missing or orphaned
derived results and unexpected audit artifacts. Empty pre-release trees pass with zero counts.
It does not call a provider, read credentials, or write files.

```sh
make audit
uv run adventure-bench-audit
```

`make check` and the verification workflow run this audit automatically. The
existing `rescore-check`, `aggregate-check`, and `site-check` commands remain
available for focused explicit-ID investigations.

Submissions: use `adventure-bench-collect`, then open a PR with the resulting
`runs/<run-id>/` evidence, including the model ID, resolved provider/runtime,
and run date. Runs with any transport failures or unresolved providers are
invalid. Local runs (any OpenAI-compatible server via `--base-url`) are welcome
— note the quantization. **Wanted:** Qwen3.5-4B (Q4 GGUF) and PrismML
Bonsai-27B (1-bit / ternary) — same memory footprint, very different
architectures; the capability-per-gigabyte head-to-head is the on-device
question in miniature.

## How it works

Each case gives the model a scene (room name + description, exits, visible items with ids, inventory) and one player input, via a **frozen system prompt** that defines a 7-action vocabulary (`move/take/drop/examine/use/look/inventory`) plus `unclear`, with a strict-JSON reply schema. The model sees visible state only.

Scoring is deterministic: the reply's action + target is normalized (direction aliases; noun resolved against item ids *and* names with underscore/space normalization, the way a game engine would) and compared to the case's explicit set of acceptable outcomes. Malformed replies get exactly one corrective retry, then score as `unclear`. Refusal cases *expect* `unclear` — out-of-vocabulary verbs (open, put-in, talk-to: top-10 verbs in real classic play that this schema deliberately lacks), prose-only nouns, absent objects, nonsense.

Infrastructure failures cannot contaminate scores: a preflight call aborts unreachable runs, and per-case transport errors are counted separately and invalidate the run.

The harness — prompt, retry policy, scoring — is part of the benchmark definition and frozen per major version. Temperature 0; hybrid-reasoning models get thinking disabled (measured ~6x latency for identical answers on this task).

## Provenance & licensing

Scene prose is original to this project. Player inputs are short functional commands — either verbatim from cited studies and community documentation, or composed in documented patterns. Full per-source table, methodology, and limitations: `DATASET_CARD.md`.

**License: CC BY-NC 4.0** (dataset, harness, and docs — see `LICENSE`). Free to use, share, and adapt with attribution; commercial use is not permitted. Publishing benchmark scores for your model is permitted use with attribution.
