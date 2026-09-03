# Methodology

AdventureBench evaluates grounded action interpretation for interactive-fiction
contexts. A case supplies a player utterance and the state visible to that
player. A system returns a structured action or a calibrated refusal.

## Evaluation principles

- **Grounded context:** expected actions must be justified by the visible scene,
  exits, items, and inventory supplied with the case.
- **Deterministic scoring:** outputs are compared with explicit acceptable
  outcomes. The benchmark does not use a generative model as a judge.
- **Mapping and calibration:** reports distinguish successful interpretation
  from correct refusal of unsupported, absent, ambiguous, or otherwise
  ungrounded requests.
- **Reproducible runs:** a published result identifies the benchmark version,
  model, provider or runtime, decoding configuration, and any infrastructure
  failures.

## Versioning

A benchmark version fixes its dataset, action contract, prompt or interface,
normalization, scoring rules, and retry policy. Changes to any of these are
reported as a new version or an explicitly documented revision; results are not
compared across incompatible definitions.

## Release-set comparisons

A cross-model release is an explicitly ordered set of at least two completed,
one-model collection runs. Before publication, every member is replayed from
its raw evidence with the offline scorer in check mode. All members must be
valid, have distinct model slugs, use the same benchmark name/version, dataset
SHA-256, and prompt SHA-256, and cover exactly the same `(case_id, repetition)`
pairs. This is the release contract: incomplete, drifted, or incompatible runs
are not silently combined.

`adventure-bench-aggregate` writes one canonical file at
`releases/<release-id>/release.json`. It includes compact model summaries, the
tag score matrix, the UTC collection window, and manifest/response SHA-256
provenance. It copies no raw model completion from collection evidence. The
machine-readable contract is [`release-schema.json`](release-schema.json).

Overall uncertainty is reported as a deterministic two-sided 95% percentile
bootstrap interval. The bootstrap resamples case IDs with replacement and
keeps all repetitions of a sampled case together, so repeated attempts are not
treated as independent cases. It uses seed `20260901`, 10,000 replicates, an
explicit xorshift32 generator with rejection sampling, and nearest-rank
percentiles. Paired score-difference intervals use the same bootstrap samples;
their sign is always `model_a - model_b`. These intervals describe the measured
release set, not a universal ordering or a winner declaration.
