---
license: cc-by-nc-4.0
task_categories:
  - text-classification
  - text2text-generation
language:
  - en
tags:
  - instruction-following
  - grounding
  - calibration
  - interactive-fiction
  - text-adventure
  - structured-output
pretty_name: Adventure Bench
size_categories:
  - n<1K
---

# Dataset Card — Adventure Bench v1.0

## Summary

244 single-turn evaluation cases for grounded instruction-following. Each case pairs a self-contained scene (room description, exits, visible items, inventory) with one free-form player input and an explicit set of acceptable structured outcomes over a 7-action vocabulary plus `unclear`. The dataset measures two abilities separately, via 29 pattern tags: **mapping** natural language to available actions, and **calibration** — refusing inputs the schema cannot express, without hallucinating.

## Fields

| field | type | description |
|---|---|---|
| `id` | string | unique dotted identifier (`scene.pattern.slug`) |
| `source` | string | provenance key (see table below) |
| `tags` | list[string] | pattern taxonomy, controlled vocabulary (29 tags) |
| `input` | string | the player's utterance, verbatim from source or composed in a documented pattern |
| `context` | object | `room {name, description}`, `exits` (canonical directions), `items` / `carrying` (`{id, name}` each) |
| `expect` | list[[action, target]] | acceptable outcomes; `target` is an item id, canonical direction, or null. Any match passes. |

One JSON object per line (`cases.jsonl`). Single split (`test`); this is an evaluation set, not training data.

## Provenance

Inputs derive from three research passes over primary and community sources (July 2026):

| source key | n | what it is |
|---|---|---|
| `original` | ~95 | composed by the authors in documented patterns, over original scenes |
| `reed-study` | ~35 | Aaron Reed, "Whom The Playing Changed" (72 transcripts, 2006) & Blue Lacuna follow-up (139 transcripts) — verbatim novice inputs and measured error classes |
| `smarter-parser` | ~32 | verbatim test inputs and documented rewrite-rule classes from Reed's Smarter Parser Inform 7 extension |
| `zork1`, `zork2` | ~21 | Zork walkthrough commands (MIT transcript, commands-only solutions) and Infocom manual examples |
| `infocom-manual` | ~19 | documented abbreviations, error-message examples, meta commands from Infocom documentation |
| `inform7` | ~13 | the post-1995 standard verb set and conventions (Inform 7 Standard Rules) |
| `colossal-cave` | ~8 | Crowther & Woods walkthrough commands, magic words, two-word-parser lore |
| `hhgg`, `planetfall`, `enchanter`, `adventureland` | ~11 | game-specific famous commands (babel fish sequence, spell system, GO-noun) |
| `thy-dungeonman`, `kings-quest-1` | ~4 | "ye flask" and guess-the-noun lore |
| `starborn-study` | ~4 | Juhana Leinonen's Starborn statistics (1,557 transcripts, 23,896 turns; 53% of invalid input = typos) |
| `iftf-faq`, `ai-dungeon` | ~4 | IFTF's canonical "reasonable things parsers reject"; AI Dungeon input registers |

Tag weights follow measured novice error frequencies where known (Reed: 19% missing-preposition, 19% unsupported synonym, 14% non-imperative, 25%+ typos).

## Construction & quality control

- All scene prose is original to this project (some scenes are *structural* homages to famous game moments; no source text is reproduced).
- Player inputs are short functional commands (1–10 words). Verbatim inputs are limited to command strings documented in the cited studies/references — facts about what players typed, not creative expression.
- Every case passes machine validation (`adventure_bench/validate.py`): unique ids, controlled tag/source vocabularies, canonical exits, and every expected outcome realizable in its context (expected items exist; expected moves have exits).
- `expect` sets are explicit about legitimate multiplicity: compound commands accept any constituent action or an honest `unclear`; disambiguation cases accept the sensible reading or a clarifying refusal.
- The evaluation harness (frozen prompt, retry policy, deterministic scorer) ships with the dataset and is part of the benchmark definition.

## Intended use & limitations

- **Intended:** comparing models' grounded instruction-following and refusal calibration; regression-testing prompt/schema changes in parser-replacement systems; studying small-model on-device viability.
- **Single-turn only.** No dialogue history, so anaphora across turns ("go back", "take it" referring to prior narration) is only lightly covered.
- **English only**, and skewed to the conventions of Anglophone parser IF.
- **The 7-action schema is deliberately narrow.** Out-of-vocabulary cases expecting `unclear` encode the judgment "refusing honestly beats guessing" — systems with richer schemas will find some of these cases express *their* in-vocabulary actions; the tags make it easy to exclude or re-map them.
- Scores near the ceiling should be read against the harness's acceptance sets, not as "solved": the set intentionally contains judgment calls (documented per-case via tags like `ambiguity`, `guess-the-noun`).
- No personal data; no offensive content; inputs include mild frustration expressions ("wtf") documented in player studies.

## Versioning

v1.0 — 244 cases, 29 tags, 22 scenes, 17 sources. The case set and harness are frozen; additions will be v1.x (new cases only) or v2 (schema/prompt changes).

## Citation

See `CITATION.cff`. Key underlying studies: Aaron Reed's transcript analyses and Smarter Parser extension; Juhana Leinonen's Starborn statistics; the Infocom manuals and historical ZIL sources.
