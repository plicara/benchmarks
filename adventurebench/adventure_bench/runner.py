"""Adventure Bench runner: evaluate any OpenAI-compatible chat model.

Usage:
    adventure-bench --model mistralai/ministral-3b-2512 --provider mistral
    adventure-bench --model z-ai/glm-5.2 --out results.json
    adventure-bench --tag out-of-vocab --verbose

Environment:
    ADVENTURE_BENCH_API_KEY   API key (falls back to OPENROUTER_API_KEY or OPENROUTER_KEY)
    ADVENTURE_BENCH_BASE_URL  chat-completions base (default: OpenRouter)

The benchmark definition is FROZEN per major version: the system prompt, the
user-message format, the retry policy, and the scoring rules below are part
of the benchmark itself. Comparable numbers require running all of it as-is.

Design notes (why the harness looks like this):
- Scoring is deterministic — no LLM judge. The model's noun is resolved with
  the same normalization a game engine would apply, then compared against an
  explicit set of acceptable outcomes.
- Refusal cases are first-class: out-of-vocabulary and impossible requests
  expect "unclear". They measure calibration, not just mapping.
- Transport failures (auth/network/HTTP) can never masquerade as model
  answers: a preflight call aborts the run loudly, and per-case transport
  errors are reported separately from scored outcomes.
- One retry on a malformed reply, with the bad reply shown back to the model.
  Persistent garbage scores as "unclear" (the game would have shrugged too).
- Hybrid-reasoning models get thinking disabled: measured on this task, it
  multiplies latency ~6x for identical answers.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent / "data" / "cases.jsonl"

ACTIONS = ("move", "take", "drop", "examine", "use", "look", "inventory", "unclear")
NO_TARGET_ACTIONS = ("look", "inventory", "unclear")

DIRECTIONS = {
    "north": "north", "n": "north", "south": "south", "s": "south",
    "east": "east", "e": "east", "west": "west", "w": "west",
    "up": "up", "u": "up", "down": "down", "d": "down",
}

# Frozen benchmark prompt (v1). Example lines are deliberately prefixed:
# at least one hosting provider returns EMPTY completions when system-prompt
# lines begin with a bare "{" (unescaped template rendering, presumably).
SYSTEM_PROMPT = """You translate a player's natural-language input into ONE structured action for a text adventure game. Reply with a single JSON object and nothing else — no prose, no code fences.

Schema: {"action": "move" | "take" | "drop" | "examine" | "use" | "look" | "inventory" | "unclear", "target": string | null, "reason": string | null}

Rules:
- "move": target is one of the exits listed in the context (a direction like "north"). If the player names a place the room description says lies in some direction, move that direction.
- "take"/"drop"/"examine"/"use": target names an item from items_here or carrying (prefer the item's id). Map synonyms and paraphrases to what the player MEANT: "grab the light" -> take the lantern; "peek under the carpet" -> examine the rug.
- "look": re-describe the surroundings (target null).
- "inventory": what the player carries (target null).
- "unclear": the intent maps to no available action, or refers to something that is not in the context. Set reason to one short, in-world, player-facing sentence. Never invent items, directions, or facts.

Examples (input -> reply):
"walk north" -> {"action": "move", "target": "north", "reason": null}
"grab the light" (a lantern is present) -> {"action": "take", "target": "lantern", "reason": null}
"take the sword" (no sword in context) -> {"action": "unclear", "target": null, "reason": "There is no sword here."}"""


class TransportError(Exception):
    pass


def api_key_from_env() -> str | None:
    """Return an API key from the supported local environment names."""
    return (
        os.environ.get("ADVENTURE_BENCH_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("OPENROUTER_KEY")
    )


class ChatClient:
    """Minimal OpenAI-compatible chat-completions client (stdlib only)."""

    def __init__(self, api_key: str, model: str, base_url: str | None = None,
                 provider: str | None = None, timeout: float = 60.0,
                 max_output_tokens: int | None = None):
        if max_output_tokens is not None and (isinstance(max_output_tokens, bool) or max_output_tokens < 1):
            raise ValueError("max_output_tokens must be at least one")
        self.api_key = api_key
        self.model = model
        self.base_url = (base_url or os.environ.get(
            "ADVENTURE_BENCH_BASE_URL", "https://openrouter.ai/api/v1")).rstrip("/")
        self.provider = provider
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens

    @staticmethod
    def response_text(data: dict) -> str:
        """Extract text from one OpenAI-compatible response payload."""
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as err:
            raise TransportError(f"unexpected response shape: {json.dumps(data)[:300]}") from err
        return content if isinstance(content, str) else ""

    def complete_response(self, messages: list[dict], *, routing_metadata: bool = False) -> dict:
        """Return the complete response when an evidence collector needs metadata."""
        payload: dict = {"model": self.model, "messages": messages, "temperature": 0,
                         "reasoning": {"enabled": False}}
        if self.max_output_tokens is not None:
            payload["max_tokens"] = self.max_output_tokens
        if self.provider:
            payload["provider"] = {"order": [self.provider], "allow_fallbacks": False}
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json",
                   "X-Title": "Adventure Bench"}
        if routing_metadata:
            headers["X-OpenRouter-Metadata"] = "enabled"
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers=headers,
        )
        return self._post(request)

    def complete(self, messages: list[dict]) -> str:
        # content: null (provider refusal/empty completion) -> retryable-malformed
        return self.response_text(self.complete_response(messages))

    def _post(self, request: urllib.request.Request) -> dict:
        detail = ""
        for delay in (2, 4, 8):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.load(response)
            except urllib.error.HTTPError as err:
                detail = err.read().decode(errors="replace")[:300]
                if err.code == 429:
                    time.sleep(delay)
                    continue
                raise TransportError(f"HTTP {err.code}: {detail}") from err
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as err:
                raise TransportError(str(err)) from err
        raise TransportError(f"HTTP 429: still rate-limited after retries: {detail}")


# --- interpretation (model reply -> outcome) ---

def user_message(case: dict) -> str:
    ctx = case["context"]
    return json.dumps({
        "input": case["input"],
        "room": {"name": ctx["room"]["name"], "description": ctx["room"]["description"]},
        "exits": ctx.get("exits", []),
        "items_here": [{"id": i["id"], "name": i["name"]} for i in ctx.get("items", [])],
        "carrying": [{"id": i["id"], "name": i["name"]} for i in ctx.get("carrying", [])],
    })


def extract_json(text: str | None) -> dict | None:
    """First JSON object in a reply, tolerating fences and preamble."""
    if not text:
        return None
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
                return obj if isinstance(obj, dict) else None
    return None


def resolve_target(case: dict, noun: str) -> str:
    """Normalize the model's noun the way a game engine would: underscores to
    spaces, then substring-match against reachable item ids and names.
    Returns the item id on a match, else the raw noun."""
    noun = noun.lower().strip().replace("_", " ")
    ctx = case["context"]
    for item in ctx.get("items", []) + ctx.get("carrying", []):
        if noun in item["name"].lower() or noun in item["id"].replace("_", " "):
            return item["id"]
    return noun


def reply_to_outcome(case: dict, obj: dict | None):
    """Validated reply JSON -> (kind, target) outcome. None means malformed
    (worth one retry)."""
    if obj is None:
        return None
    kind = str(obj.get("action", "")).lower().strip()
    if kind not in ACTIONS:
        return None
    if kind in NO_TARGET_ACTIONS:
        return (kind, None)
    target = obj.get("target")
    target = str(target).lower().strip() if target is not None else ""
    if not target:
        return None
    if kind == "move":
        return (kind, DIRECTIONS.get(target, target))
    return (kind, resolve_target(case, target))


def run_case(case: dict, complete):
    """Returns ((kind, target), transport_error: bool)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message(case)},
    ]
    for _ in range(2):
        try:
            reply = complete(messages)
        except TransportError:
            return ("unclear", None), True
        outcome = reply_to_outcome(case, extract_json(reply))
        if outcome is not None:
            return outcome, False
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content":
                         "That was not a single valid JSON object matching the schema. Reply with only the JSON object."})
    return ("unclear", None), False


# --- dataset ---

def load_cases(path: Path = DATA_PATH) -> list[dict]:
    cases = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            cases.append(json.loads(line))
    return cases


# --- CLI ---

def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv

    def opt(flag, default=None):
        return args[args.index(flag) + 1] if flag in args else default

    model = opt("--model")
    if not model:
        sys.exit("required: --model <id>  (e.g. --model mistralai/ministral-3b-2512)")
    api_key = api_key_from_env()
    if not api_key:
        sys.exit("set ADVENTURE_BENCH_API_KEY, OPENROUTER_API_KEY, or OPENROUTER_KEY")
    max_output_tokens = opt("--max-output-tokens")
    client = ChatClient(
        api_key, model=model, base_url=opt("--base-url"), provider=opt("--provider"),
        max_output_tokens=int(max_output_tokens) if max_output_tokens is not None else None,
    )
    verbose = "--verbose" in args

    cases = load_cases(Path(opt("--data")) if opt("--data") else DATA_PATH)
    if tag := opt("--tag"):
        cases = [c for c in cases if tag in c["tags"]]
    if limit := opt("--limit"):
        cases = cases[: int(limit)]
    if not cases:
        sys.exit("no cases matched")
    print(f"Adventure Bench v{__import__('adventure_bench').__version__}")
    print(f"model: {model}" + (f" (provider: {client.provider})" if client.provider else "") + f"   cases: {len(cases)}\n")

    # Preflight: if the endpoint is unreachable, abort loudly instead of
    # scoring a run of silent transport failures as model refusals.
    _, transport = run_case(cases[0], client.complete)
    if transport:
        sys.exit("endpoint unreachable (auth/network/HTTP error on preflight) — aborting, nothing scored")

    by_tag: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    results, passed, transport_failures = [], 0, 0
    for case in cases:
        got, transport = run_case(case, client.complete)
        if transport:
            transport_failures += 1
        expected = [tuple(e) for e in case["expect"]]
        ok = got in expected and not transport
        passed += ok
        for t in case["tags"]:
            by_tag[t][0] += ok
            by_tag[t][1] += 1
        results.append({"id": case["id"], "ok": ok, "got": list(got),
                        "transport_error": transport, "tags": case["tags"]})
        if not ok:
            note = "TRANSPORT" if transport else "FAIL"
            print(f"  {note:9} {case['id']:36} {case['input']!r:50} -> {got}  wanted {expected}")
        elif verbose:
            print(f"  pass      {case['id']:36} {case['input']!r:50} -> {got}")

    print(f"\noverall: {passed}/{len(cases)} ({100 * passed // len(cases)}%)")
    if transport_failures:
        print(f"WARNING: {transport_failures} transport failure(s) — treat this run as invalid")
    print(f"\n{'tag':<24}{'pass':>6}{'total':>7}")
    for t in sorted(by_tag, key=lambda t: by_tag[t][0] / by_tag[t][1]):
        p, n = by_tag[t]
        print(f"{t:<24}{p:>6}{n:>7}   {'#' * round(20 * p / n):<20} {100 * p // n}%")

    if out := opt("--out"):
        Path(out).write_text(json.dumps({
            "benchmark": "adventure-bench",
            "version": __import__("adventure_bench").__version__,
            "model": model, "provider": client.provider,
            "overall": {"passed": passed, "total": len(cases)},
            "transport_failures": transport_failures,
            "by_tag": {t: {"passed": v[0], "total": v[1]} for t, v in by_tag.items()},
            "results": results,
        }, indent=2))
        print(f"\nresults written to {out}")


if __name__ == "__main__":
    main()
