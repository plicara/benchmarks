"""Dataset lints for Adventure Bench. Run: python -m adventure_bench.validate

Every case is machine-checked so the released dataset cannot contain the
classes of authoring error we already made once (dangling item references,
malformed expectations, uncontrolled tag vocabulary).
"""
from __future__ import annotations

import json
import sys
from collections import Counter

from .runner import ACTIONS, DATA_PATH, DIRECTIONS, NO_TARGET_ACTIONS, load_cases

TAGS = {
    # positive mapping ability
    "exact-verb", "verb-alias", "abbreviation", "synonym", "missing-preposition",
    "paraphrase", "direction-as-place", "go-to-place", "relative-direction",
    "politeness", "full-sentence", "question", "adverb", "typo", "pronoun",
    "compound", "multi-object",
    # grounding / calibration
    "flavor-noun", "absent-object", "ambiguity", "guess-the-noun",
    "out-of-vocab", "meta", "classic-command", "help-seeking",
    "greeting", "frustration", "story-mode", "nonsense",
}

SOURCES = {
    "original", "zork1", "zork2", "colossal-cave", "hhgg", "planetfall",
    "enchanter", "adventureland", "thy-dungeonman", "kings-quest-1",
    "reed-study", "starborn-study", "smarter-parser", "iftf-faq",
    "ai-dungeon", "infocom-manual", "inform7",
}

REQUIRED = ("id", "source", "tags", "input", "context", "expect")


def validate(cases: list[dict]) -> list[str]:
    problems: list[str] = []
    ids = [c.get("id") for c in cases]
    for dup, n in Counter(ids).items():
        if n > 1:
            problems.append(f"duplicate id {dup!r} ({n}x)")

    for c in cases:
        cid = c.get("id", "<no id>")
        for field in REQUIRED:
            if field not in c:
                problems.append(f"{cid}: missing field {field!r}")
                break
        else:
            if unknown := set(c["tags"]) - TAGS:
                problems.append(f"{cid}: unknown tags {sorted(unknown)}")
            if not c["tags"]:
                problems.append(f"{cid}: no tags")
            if c["source"] not in SOURCES:
                problems.append(f"{cid}: unknown source {c['source']!r}")
            if not c["input"].strip():
                problems.append(f"{cid}: empty input")

            ctx = c["context"]
            if "room" not in ctx or "name" not in ctx["room"] or "description" not in ctx["room"]:
                problems.append(f"{cid}: context.room needs name and description")
            item_ids = {i["id"] for i in ctx.get("items", []) + ctx.get("carrying", [])}
            exits = set(ctx.get("exits", []))
            for d in exits:
                if d not in DIRECTIONS.values():
                    problems.append(f"{cid}: non-canonical exit {d!r}")

            if not c["expect"]:
                problems.append(f"{cid}: empty expect")
            for pair in c["expect"]:
                if not (isinstance(pair, list) and len(pair) == 2):
                    problems.append(f"{cid}: malformed expect entry {pair!r}")
                    continue
                kind, target = pair
                if kind not in ACTIONS:
                    problems.append(f"{cid}: unknown action {kind!r} in expect")
                elif kind in NO_TARGET_ACTIONS:
                    if target is not None:
                        problems.append(f"{cid}: {kind} expects null target, got {target!r}")
                elif kind == "move":
                    if target not in exits:
                        problems.append(f"{cid}: expected move {target!r} but exits are {sorted(exits)}")
                elif target not in item_ids:
                    problems.append(f"{cid}: expected {kind} {target!r} but context items are {sorted(item_ids)}")
    return problems


def stats(cases: list[dict]) -> str:
    tags = Counter(t for c in cases for t in c["tags"])
    sources = Counter(c["source"] for c in cases)
    scenes = Counter(c["context"]["room"]["name"] for c in cases)
    lines = [f"{len(cases)} cases, {len(tags)} tags, {len(sources)} sources, {len(scenes)} scenes", ""]
    lines.append(f"{'tag':<24}{'n':>4}")
    for t, n in sorted(tags.items(), key=lambda kv: -kv[1]):
        lines.append(f"{t:<24}{n:>4}")
    lines.append("")
    lines.append(f"{'source':<24}{'n':>4}")
    for s, n in sorted(sources.items(), key=lambda kv: -kv[1]):
        lines.append(f"{s:<24}{n:>4}")
    return "\n".join(lines)


def main() -> None:
    cases = load_cases(DATA_PATH)
    problems = validate(cases)
    if problems:
        print("INVALID dataset:\n  - " + "\n  - ".join(problems))
        sys.exit(1)
    print(f"dataset valid ✓\n\n{stats(cases)}")


if __name__ == "__main__":
    main()
