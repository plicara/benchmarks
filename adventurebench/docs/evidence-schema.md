# Auditable run evidence

Adventure Bench v1 collection evidence is an append-only JSONL record for each
model and a run-level manifest. The collector writes:

```text
runs/<run-id>/
  manifest.json
  responses/<model-slug>.jsonl
```

The canonical machine-readable contract is
[`evidence-schema.json`](evidence-schema.json). This document explains the
publication rules that accompany it.

## Run manifest

`manifest.json` identifies the result schema version, Adventure Bench version,
imported source commit, checked-out repository commit, dataset and frozen-prompt
SHA-256 hashes, the ordered selected-case ID hash, UTC start/end times, the
selected case count and repetitions, and the collector command/version,
and the collection configuration. Configuration includes requested and resolved
model/provider/runtime identities, the endpoint hostname only, decoding
settings, retry policy, and timeout. It intentionally excludes endpoint paths,
query strings, headers, and credentials.

The collector treats one run as one immutable model/provider/runtime
configuration. Its single entry in `models` records the evidence file, expected
and recorded case/repetition count, observed resolved provider,
transport-failure count, and validity. Comparing several models is a separate
aggregation step over several completed runs; a run ID is never reused for a
different model. A run is valid only when it has complete evidence, one resolved
provider across its records, and no transport failures. A malformed model reply
is not a transport failure: v1 gives it the frozen corrective retry, then scores
its final malformed outcome as `unclear`.

## Response records

Every nonblank line of `responses/<model-slug>.jsonl` is one immutable
case/repetition record. It contains:

- the case ID and one-based repetition;
- safe request provenance and its prompt hash;
- every raw completion attempt, the parsed action/target (or `null` when
  malformed), attempt latency, and any structured error;
- the final parsed outcome, explicit acceptable outcomes, deterministic pass or
  fail, transport flag and structured error;
- end-to-end latency; optional provider-supplied usage and cost; and
- resolved model, provider, and runtime identity for that response.

The built-in OpenAI-compatible adapter retains the response's observed model,
token-usage object, and cost when supplied. On OpenRouter it also opts into
routing metadata and records the uniquely selected provider rather than merely
trusting the requested provider name. The metadata itself is not serialized.

The full system prompt and user message remain independently reconstructible
from the frozen benchmark, case ID, and prompt hash. This avoids duplicating
credentials or incidental transport details in every line while preserving the
actual raw model completion needed for later offline rescoring.

## Safety and resuming

Run IDs accept only lowercase letters, digits, and hyphens. Model filenames use
a deterministic lowercase slug; paths outside the chosen `runs/` directory are
rejected. Re-running a collection with the same run ID reads existing JSONL and
does not append a duplicate `(case_id, repetition)` pair. Existing duplicate or
malformed JSONL is an error rather than something silently repaired.

Before serialization, fields whose names identify authorization or credential
material are replaced, and configured API-key values plus Bearer authorization
strings are redacted from response/error text. Never put credentials in model
IDs, provider IDs, run IDs, runtime descriptions, or model prompts.

## Collection command

Use the separate command so the historical `adventure-bench` CLI remains
unchanged:

```sh
export ADVENTURE_BENCH_API_KEY=...
uv run adventure-bench-collect \
  --model example/model \
  --provider pinned-provider \
  --runtime openai-compatible:release-identifier \
  --repetitions 3 \
  --run-id 20260901t120000z-example-model
```

For a hosted OpenAI-compatible endpoint, `--provider` pins the requested host
and is used as the resolved provider unless a trusted runtime adapter supplies
`--resolved-provider`. A collection with no resolved provider is deliberately
marked invalid and must not be published as a benchmark result.

## Derived results and offline replay

The collector's `outcome`, `passed`, and `expected_outcomes` fields are useful
diagnostics, but are not score authority. `adventure-bench-rescore` reads the
manifest, raw JSONL, and dataset offline; it checks provenance and coverage,
then replays each raw completion through the frozen v1 `extract_json` and
`reply_to_outcome` functions. Any disagreement with a recorded parsed attempt,
outcome, expected outcome, pass flag, transport flag, or run/model validity is
an error. This makes an edited score claim fail rather than silently becoming a
new result.

```sh
uv run adventure-bench-rescore --run-id 20260901t120000z-example-model
uv run adventure-bench-rescore --run-id 20260901t120000z-example-model --check
# equivalent Make target:
make rescore-check RUN_ID=20260901t120000z-example-model
```

The first command writes only deterministic derived artifacts:

```text
results/<run-id>/<model-slug>.json
results/<run-id>/summary.json
results/<run-id>/tag-matrix.json
```

The canonical machine-readable result contract is
[`result-schema.json`](result-schema.json). Every file uses derived-result
schema version `1.0`, sorted JSON keys, and no clock or environment values. A
model result includes stable benchmark hashes,
SHA-256 hashes of its manifest and exact JSONL evidence bytes, requested and
resolved model/provider/runtime identities, validity reasons, overall and
per-repetition scores, per-tag pass/total/score tables, and transport/parsing
failure counts. It also contains a sorted `case_results` table with each
case ID, repetition, dataset tags, replayed action/target outcome, and replayed
pass flag. It deliberately does not copy raw completions out of the evidence
JSONL. `summary.json` contains the corresponding compact model rows;
`tag-matrix.json` transposes per-tag numerator/denominator values across
models. `--check` writes nothing and fails if any expected result file is
missing or byte-different, or if an unexpected derived JSON file is present.

Transport failures intentionally remain represented in derived output, but
make their model and the containing run invalid. Persistent malformed replies
are not transport failures: the frozen retry is replayed and their final
outcome is `unclear`.

## Cross-model release artifacts

After each member run has been rescored, create a release from an explicit,
ordered list of run IDs:

```sh
uv run adventure-bench-aggregate \
  --release-id 20260901-pilot \
  --run-id 20260901t120000z-model-a \
  --run-id 20260901t121000z-model-b
uv run adventure-bench-aggregate \
  --release-id 20260901-pilot \
  --run-id 20260901t120000z-model-a \
  --run-id 20260901t121000z-model-b \
  --check
```

The command replays every raw run with `rescore_run(check=True)` before it
writes `releases/<release-id>/release.json`. `--check` writes nothing and
requires that canonical artifact to match byte-for-byte. Release and run IDs
use the same lowercase, path-safe identifier rule. See the release-set contract
and interval interpretation in [the methodology](methodology.md).
