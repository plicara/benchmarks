"""Render an audited Adventure Bench release as a static technical site page.

Rendering is deliberately downstream of :mod:`adventure_bench.aggregate`.
Before a page is written, this module replays every explicit release member and
requires the checked-in release JSON to be its exact canonical representation.
"""
from __future__ import annotations

import argparse
import html
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

from .aggregate import ReleaseError, aggregate_release
from .collect import validate_run_id


REPOSITORY_URL = "https://github.com/plicara/benchmarks"
PROJECT_URL = f"{REPOSITORY_URL}/tree/main/adventurebench"
DATASET_CARD_URL = f"{REPOSITORY_URL}/blob/main/adventurebench/DATASET_CARD.md"
METHODOLOGY_URL = f"{REPOSITORY_URL}/blob/main/adventurebench/docs/methodology.md"
RERUN_URL = f"{PROJECT_URL}#auditable-result-collection"


class SiteRenderError(ValueError):
    """A release cannot be safely turned into a publication page."""


def _inside(root: Path, child: Path) -> Path:
    resolved_root = root.resolve()
    resolved_child = child.resolve()
    if resolved_child != resolved_root and resolved_root not in resolved_child.parents:
        raise SiteRenderError("refusing a site path outside the configured site directory")
    return resolved_child


def _escaped(value: Any) -> str:
    """Escape every value that originates in an audited artifact."""
    return html.escape(str(value), quote=True)


def _release_identity(releases_dir: Path, release_id: str) -> list[str]:
    """Read only the identity needed to ask the aggregate checker to replay it."""
    validate_run_id(release_id)
    release_path = _inside(releases_dir, releases_dir / release_id / "release.json")
    try:
        value = json.loads(release_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as err:
        raise SiteRenderError(f"cannot read release artifact: {release_path}") from err
    if not isinstance(value, dict):
        raise SiteRenderError("release artifact must be a JSON object")
    if value.get("release_id") != release_id:
        raise SiteRenderError("release artifact ID does not match the requested release")
    if value.get("kind") != "adventure-bench-release-set" or value.get("schema_version") != "1.0":
        raise SiteRenderError("unsupported Adventure Bench release artifact")
    run_ids = value.get("run_ids")
    if (not isinstance(run_ids, list) or len(run_ids) < 2 or
            not all(isinstance(run_id, str) for run_id in run_ids)):
        raise SiteRenderError("release artifact has malformed explicit run IDs")
    try:
        for run_id in run_ids:
            validate_run_id(run_id)
    except ValueError as err:
        raise SiteRenderError("release artifact has malformed explicit run IDs") from err
    return run_ids


def _percentage(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise SiteRenderError("release artifact has an invalid score")
    return f"{value * 100:.1f}%"


def _interval(value: Any, *, signed: bool = False) -> str:
    if (not isinstance(value, list) or len(value) != 2 or
            any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value)):
        raise SiteRenderError("release artifact has a malformed bootstrap interval")
    if value[0] > value[1]:
        raise SiteRenderError("release artifact has a reversed bootstrap interval")
    prefix = "+" if signed else ""
    return f"{prefix}{value[0] * 100:.1f}% to {prefix}{value[1] * 100:.1f}%"


def _collection_date_label(started_at: str, completed_at: str) -> str:
    try:
        started = datetime.fromisoformat(started_at.replace("Z", "+00:00")).date().isoformat()
        completed = datetime.fromisoformat(completed_at.replace("Z", "+00:00")).date().isoformat()
    except ValueError as err:
        raise SiteRenderError("release artifact has an invalid collection window") from err
    if started == completed:
        return f"collected on {started} (UTC)"
    return f"collected from {started} through {completed} (UTC)"


def _resolved_counts(artifact: dict[str, Any]) -> dict[str, tuple[int, int]]:
    models = {row["model_slug"] for row in artifact["models"]}
    counts = {slug: [0, 0] for slug in models}
    comparisons = artifact["bootstrap"]["paired_score_difference_intervals"]
    if not isinstance(comparisons, list):
        raise SiteRenderError("release artifact has malformed paired comparisons")
    for row in comparisons:
        if not isinstance(row, dict):
            raise SiteRenderError("release artifact has malformed paired comparisons")
        left, right, interval = row.get("model_a"), row.get("model_b"), row.get("interval")
        if left not in counts or right not in counts or left == right:
            raise SiteRenderError("release artifact has an unknown paired comparison model")
        if (not isinstance(interval, list) or len(interval) != 2 or
                any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in interval)):
            raise SiteRenderError("release artifact has a malformed paired comparison interval")
        low, high = interval
        if low > high:
            raise SiteRenderError("release artifact has a reversed paired comparison interval")
        if low > 0:
            counts[left][0] += 1
            counts[right][1] += 1
        elif high < 0:
            counts[left][1] += 1
            counts[right][0] += 1
    return {slug: (value[0], value[1]) for slug, value in counts.items()}


def _recorded_cost(model: dict[str, Any]) -> float | None:
    evidence = model.get("evidence")
    if not isinstance(evidence, dict):
        raise SiteRenderError("release artifact has malformed evidence metadata")
    value = evidence.get("recorded_cost_usd")
    if value is None:
        return None
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0):
        raise SiteRenderError("release artifact has an invalid recorded cost")
    return float(value)


def _cost_label(value: float | None) -> str:
    return "not recorded" if value is None else f"${value:.3f}"


def _pareto_frontier(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return nondominated models when all members have recorded full-run cost."""
    points: list[dict[str, Any]] = []
    for model in models:
        if not isinstance(model, dict):
            raise SiteRenderError("release artifact has malformed model data")
        cost = _recorded_cost(model)
        overall = model.get("overall")
        if cost is None:
            return []
        if not isinstance(overall, dict) or isinstance(overall.get("score"), bool) or not isinstance(overall.get("score"), (int, float)):
            raise SiteRenderError("release artifact has invalid model score")
        score = float(overall["score"])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise SiteRenderError("release artifact has invalid model score")
        provenance = model.get("provenance")
        resolved = provenance.get("resolved") if isinstance(provenance, dict) else None
        name = resolved.get("model") if isinstance(resolved, dict) else None
        if not isinstance(name, str) or not name:
            raise SiteRenderError("release artifact has missing resolved model provenance")
        points.append({"model": model, "name": name, "cost": cost, "score": score})
    frontier = [
        point for point in points
        if not any(
            other["cost"] <= point["cost"] and other["score"] >= point["score"]
            and (other["cost"] < point["cost"] or other["score"] > point["score"])
            for other in points
        )
    ]
    return sorted(frontier, key=lambda point: (point["cost"], point["name"].casefold()))


def _pareto_section(artifact: dict[str, Any]) -> str:
    models = artifact.get("models")
    if not isinstance(models, list) or len(models) < 2:
        raise SiteRenderError("release artifact has malformed models")
    frontier = _pareto_frontier(models)
    if not frontier:
        return ""
    points = []
    for model in models:
        assert isinstance(model, dict)
        provenance = model.get("provenance")
        resolved = provenance.get("resolved") if isinstance(provenance, dict) else None
        model_id = resolved.get("model") if isinstance(resolved, dict) else None
        if not isinstance(model_id, str) or not model_id:
            raise SiteRenderError("release artifact has missing resolved model provenance")
        points.append({
            "model_id": model_id,
            "name": _pareto_name(model),
            "cost": _recorded_cost(model),
            "score": float(model["overall"]["score"]),
        })
    assert all(point["cost"] is not None for point in points)
    costs = [float(point["cost"]) for point in points]
    scores = [point["score"] for point in points]
    minimum_cost, maximum_cost = min(costs), max(costs)
    minimum_score, maximum_score = min(scores), max(scores)
    cost_padding = max((maximum_cost - minimum_cost) * 0.08, 0.001)
    score_padding = max((maximum_score - minimum_score) * 0.12, 0.01)
    x_low, x_high = max(0.0, minimum_cost - cost_padding), maximum_cost + cost_padding
    y_low, y_high = max(0.0, minimum_score - score_padding), min(1.0, maximum_score + score_padding)
    left, top, width, height = 86, 36, 704, 296

    def x(value: float) -> float:
        return left + (value - x_low) / (x_high - x_low) * width

    def y(value: float) -> float:
        return top + height - (value - y_low) / (y_high - y_low) * height

    grid = "\n".join(
        f'<line class="pareto-grid" x1="{left}" y1="{y(y_low + (y_high - y_low) * index / 4):.2f}" x2="{left + width}" y2="{y(y_low + (y_high - y_low) * index / 4):.2f}" />'
        for index in range(5)
    )
    y_ticks = "\n".join(
        f'<text class="pareto-tick" x="{left - 12}" y="{y(y_low + (y_high - y_low) * index / 4) + 4:.2f}" text-anchor="end">{_percentage(y_low + (y_high - y_low) * index / 4)}</text>'
        for index in range(5)
    )
    x_ticks = "\n".join(
        f'<text class="pareto-tick" x="{x(x_low + (x_high - x_low) * index / 4):.2f}" y="{top + height + 23}" text-anchor="middle">${x_low + (x_high - x_low) * index / 4:.3f}</text>'
        for index in range(5)
    )
    frontier_path = " ".join(
        f'{"M" if index == 0 else "L"} {x(point["cost"]):.2f} {y(point["score"]):.2f}'
        for index, point in enumerate(frontier)
    )
    frontier_names = {point["name"] for point in frontier}
    frontier_order = {point["name"]: index for index, point in enumerate(frontier)}
    marks = []
    leaders = []
    labels = []
    for point in sorted(points, key=lambda item: (item["cost"], item["name"].casefold())):
        on_frontier = point["model_id"] in frontier_names
        point_x = x(point["cost"])
        point_y = y(point["score"])
        if on_frontier:
            index = frontier_order[point["model_id"]]
            if index == 0:
                label_x, label_y, label_anchor = point_x, point_y + 34, "middle"
            elif index == len(frontier) - 1:
                label_x, label_y, label_anchor = point_x - 12, point_y + 22, "end"
            elif index % 2:
                label_x, label_y, label_anchor = point_x + 10, point_y - 17, "start"
            else:
                label_x, label_y, label_anchor = point_x - 10, point_y + 24, "end"
            leader_x = label_x + ({"start": -4, "middle": 0, "end": 4}[label_anchor])
            leader_y = label_y + (5 if label_y < point_y else -10)
            leaders.append(
                f'<line class="pareto-leader" x1="{point_x:.2f}" y1="{point_y:.2f}" '
                f'x2="{leader_x:.2f}" y2="{leader_y:.2f}" />'
            )
            labels.append(
                f'<text class="pareto-label" x="{label_x:.2f}" y="{label_y:.2f}" '
                f'text-anchor="{label_anchor}">{_escaped(point["name"])}</text>'
            )
        identity = point["model_id"]
        title = point["name"] if identity == point["name"] else f'{point["name"]} ({identity})'
        marks.append(
            f'<g><title>{_escaped(title)}: {_percentage(point["score"])} at ${point["cost"]:.3f}</title>'
            f'<circle class="pareto-point {"frontier" if on_frontier else "dominated"}" cx="{point_x:.2f}" cy="{point_y:.2f}" r="6.5" /></g>'
        )
    frontier_label = " &rarr; ".join(_escaped(_pareto_name(point["model"])) for point in frontier)
    return f'''      <section class="pareto-section" aria-labelledby="pareto-heading">
        <div class="pareto-heading"><div><p class="eyebrow">the cost curve</p><h2 id="pareto-heading">The score&ndash;cost frontier</h2></div><p class="pareto-meta">Recorded full-run cost &middot; {_escaped(len(points))} models</p></div>
        <p class="pareto-lede">Every labeled frontier point is a model for which no cheaper tested model scored as well. The other points are dominated on this release; exact values for every point appear in the table below.</p>
        <div class="pareto-figure">
          <svg viewBox="0 0 840 410" role="img" aria-labelledby="pareto-title pareto-description">
            <title id="pareto-title">Adventure Bench score versus recorded full-run cost</title>
            <desc id="pareto-description">Pareto frontier: {frontier_label}.</desc>
            {grid}
            <line class="pareto-axis" x1="{left}" y1="{top + height}" x2="{left + width}" y2="{top + height}" /><line class="pareto-axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + height}" />
            {y_ticks}
            {x_ticks}
            <text class="pareto-axis-title" x="{left + width / 2:.2f}" y="{top + height + 60}" text-anchor="middle">RECORDED FULL-RUN COST (USD)</text><text class="pareto-axis-title" x="19" y="{top + height / 2:.2f}" text-anchor="middle" transform="rotate(-90 19 {top + height / 2:.2f})">ADVENTURE BENCH SCORE</text>
            <path class="pareto-frontier" d="{frontier_path}" />
            {"".join(leaders)}
            {"".join(marks)}
            {"".join(labels)}
          </svg>
          <div class="pareto-legend"><span><i class="pareto-swatch frontier"></i>Pareto frontier</span><span><i class="pareto-swatch dominated"></i>Dominated in this release</span></div>
        </div>
        <p class="note">The frontier is a cost-efficiency view, not a ranking or a statistical claim. Costs are summed from provider-recorded completion evidence; score uncertainty and paired comparisons remain in the table below.</p>
      </section>'''


def _pareto_name(model: dict[str, Any]) -> str:
    provenance = model.get("provenance")
    resolved = provenance.get("resolved") if isinstance(provenance, dict) else None
    name = resolved.get("model") if isinstance(resolved, dict) else None
    if not isinstance(name, str) or not name:
        raise SiteRenderError("release artifact has missing resolved model provenance")
    labels = {
        "ibm-granite/granite-4.0-h-micro": "Granite Micro",
        "mistralai/ministral-3b-2512": "Ministral 3B",
        "mistralai/ministral-8b-2512": "Ministral 8B",
        "mistralai/ministral-14b-2512": "Ministral 14B",
        "z-ai/glm-5.2": "GLM 5.2",
    }
    return labels.get(name, name)


def _model_rows(artifact: dict[str, Any]) -> str:
    intervals = artifact["bootstrap"].get("score_intervals")
    if not isinstance(intervals, list):
        raise SiteRenderError("release artifact has malformed score intervals")
    interval_by_slug: dict[str, list[Any]] = {}
    for row in intervals:
        if not isinstance(row, dict) or not isinstance(row.get("model_slug"), str):
            raise SiteRenderError("release artifact has malformed score intervals")
        slug = row["model_slug"]
        if slug in interval_by_slug or row.get("estimate") is None:
            raise SiteRenderError("release artifact has duplicate or incomplete score intervals")
        interval_by_slug[slug] = row.get("interval")
    counts = _resolved_counts(artifact)
    rows: list[str] = []
    models = artifact.get("models")
    if not isinstance(models, list) or len(models) < 2:
        raise SiteRenderError("release artifact has malformed models")
    for model in sorted(models, key=lambda row: str(row.get("model_slug", "")).casefold()):
        if not isinstance(model, dict):
            raise SiteRenderError("release artifact has malformed model data")
        slug = model.get("model_slug")
        overall, failures, provenance = model.get("overall"), model.get("failures"), model.get("provenance")
        if (not isinstance(slug, str) or slug not in interval_by_slug or slug not in counts or
                not isinstance(overall, dict) or not isinstance(failures, dict) or not isinstance(provenance, dict)):
            raise SiteRenderError("release artifact has incomplete model data")
        passed, total, score = overall.get("passed"), overall.get("total"), overall.get("score")
        if (isinstance(passed, bool) or not isinstance(passed, int) or passed < 0 or
                isinstance(total, bool) or not isinstance(total, int) or total < 1 or
                passed > total):
            raise SiteRenderError("release artifact has invalid model totals")
        resolved = provenance.get("resolved")
        if not isinstance(resolved, dict):
            raise SiteRenderError("release artifact has missing resolved provenance")
        model_name, provider, runtime = resolved.get("model"), resolved.get("provider"), resolved.get("runtime")
        if not isinstance(model_name, str) or not isinstance(provider, str) or runtime is not None and not isinstance(runtime, str):
            raise SiteRenderError("release artifact has invalid resolved provenance")
        parsing, transport = failures.get("parsing"), failures.get("transport")
        if (isinstance(parsing, bool) or not isinstance(parsing, int) or parsing < 0 or
                isinstance(transport, bool) or not isinstance(transport, int) or transport < 0):
            raise SiteRenderError("release artifact has invalid failure totals")
        ahead, behind = counts[slug]
        runtime_label = runtime if runtime is not None else "not recorded"
        cost = _recorded_cost(model)
        rows.append(
            "            <tr>"
            f'<th scope="row" class="mono">{_escaped(model_name)}</th>'
            f'<td class="num mono">{_percentage(score)} <span class="visually-hidden">; 95% cluster-bootstrap interval </span>({ _interval(interval_by_slug[slug]) })</td>'
            f'<td class="num mono">{passed}/{total}</td>'
            f'<td class="num mono">{_cost_label(cost)}</td>'
            f'<td class="mono">{_escaped(provider)} / {_escaped(runtime_label)}</td>'
            f'<td class="num mono">{parsing} / {transport}</td>'
            f'<td class="num mono">+{ahead} / &minus;{behind}</td>'
            "</tr>"
        )
    if set(interval_by_slug) != set(counts):
        raise SiteRenderError("release artifact score intervals do not match its models")
    return "\n".join(rows)


def _tag_rows(artifact: dict[str, Any]) -> tuple[str, str]:
    models = sorted(artifact["models"], key=lambda row: str(row.get("model_slug", "")).casefold())
    slugs = [row.get("model_slug") for row in models]
    if not all(isinstance(slug, str) for slug in slugs):
        raise SiteRenderError("release artifact has malformed model slugs")
    matrix = artifact.get("tag_matrix")
    if not isinstance(matrix, dict):
        raise SiteRenderError("release artifact has malformed tag matrix")
    headings = "\n".join(f'              <th scope="col" class="num">{_escaped(slug)}</th>' for slug in slugs)
    rows: list[str] = []
    for tag in sorted(matrix, key=str.casefold):
        values = matrix[tag]
        if not isinstance(tag, str) or not isinstance(values, dict) or set(values) != set(slugs):
            raise SiteRenderError("release artifact has incomplete tag data")
        cells: list[str] = []
        for slug in slugs:
            value = values[slug]
            if not isinstance(value, dict):
                raise SiteRenderError("release artifact has malformed tag score")
            passed, total = value.get("passed"), value.get("total")
            if (isinstance(passed, bool) or not isinstance(passed, int) or passed < 0 or
                    isinstance(total, bool) or not isinstance(total, int) or total < 1 or passed > total):
                raise SiteRenderError("release artifact has malformed tag score")
            cells.append(f'<td class="num mono">{passed}/{total}</td>')
        rows.append(f'            <tr><th scope="row" class="mono">{_escaped(tag)}</th>{"".join(cells)}</tr>')
    return headings, "\n".join(rows)


def render_html(artifact: dict[str, Any]) -> str:
    """Return the deterministic publication page for an already checked artifact."""
    benchmark, coverage, window = artifact.get("benchmark"), artifact.get("coverage"), artifact.get("collection_window")
    if not isinstance(benchmark, dict) or not isinstance(coverage, dict) or not isinstance(window, dict):
        raise SiteRenderError("release artifact has incomplete publication metadata")
    version = benchmark.get("version")
    case_count, case_repetitions, repetitions = (
        coverage.get("case_count"), coverage.get("case_repetitions"), coverage.get("repetitions")
    )
    started_at, completed_at = window.get("started_at"), window.get("completed_at")
    if (not isinstance(version, str) or isinstance(case_count, bool) or not isinstance(case_count, int) or case_count < 1 or
            isinstance(case_repetitions, bool) or not isinstance(case_repetitions, int) or case_repetitions < case_count or
            not isinstance(repetitions, list) or not repetitions or
            not all(isinstance(number, int) and not isinstance(number, bool) and number > 0 for number in repetitions) or
            not isinstance(started_at, str) or not isinstance(completed_at, str)):
        raise SiteRenderError("release artifact has invalid publication metadata")
    release_id = artifact.get("release_id")
    if not isinstance(release_id, str):
        raise SiteRenderError("release artifact has no release ID")
    release_url = f"{REPOSITORY_URL}/blob/main/adventurebench/releases/{release_id}/release.json"
    raw_evidence_url = f"{REPOSITORY_URL}/tree/main/adventurebench/runs"
    model_rows = _model_rows(artifact)
    pareto_section = _pareto_section(artifact)
    tag_headings, tag_rows = _tag_rows(artifact)
    repetitions_label = ", ".join(str(number) for number in repetitions)
    collection_label = _collection_date_label(started_at, completed_at)
    dataset_hash = benchmark.get("dataset_sha256")
    prompt_hash = benchmark.get("prompt_sha256")
    if not isinstance(dataset_hash, str) or not isinstance(prompt_hash, str):
        raise SiteRenderError("release artifact has incomplete benchmark provenance")
    return f'''<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Adventure Bench &middot; Plicara Labs</title>
    <meta name="description" content="Audited Adventure Bench release {_escaped(release_id)}: grounded action interpretation results with raw evidence and offline replay." />
    <link rel="canonical" href="https://plicara.ai/benchmarks/adventurebench/" />
    <meta name="theme-color" media="(prefers-color-scheme: dark)" content="#082C35" />
    <meta name="theme-color" media="(prefers-color-scheme: light)" content="#ffffff" />
    <link rel="icon" href="/assets/favicon.svg" type="image/svg+xml" />
    <link rel="apple-touch-icon" href="/assets/brand/apple-touch-icon-180.png" />
    <link rel="preload" as="font" type="font/woff2" href="/assets/fonts/archivo.woff2" crossorigin />
    <link rel="preload" as="font" type="font/woff2" href="/assets/fonts/jetbrains-mono-400.woff2" crossorigin />
    <link rel="preload" as="font" type="font/woff2" href="/assets/fonts/jetbrains-mono-700.woff2" crossorigin />
    <link rel="stylesheet" href="/assets/tokens.css" />
    <link rel="stylesheet" href="/assets/style.css" />
    <style>
      .pareto-section {{ margin: 3rem 0; }}
      .pareto-heading {{ display: flex; align-items: end; justify-content: space-between; gap: 1.5rem; }}
      .pareto-heading h2 {{ margin: 0; }}
      .pareto-meta, .pareto-lede {{ color: var(--pl-text-muted); }}
      .pareto-meta {{ margin: 0; font: 0.78rem var(--pl-font-mono); white-space: nowrap; }}
      .pareto-lede {{ max-width: 62ch; margin: .8rem 0 1.25rem; }}
      .pareto-figure {{ border: 1px solid var(--pl-rule); background: var(--pl-surface); padding: clamp(.8rem, 3vw, 1.75rem); }}
      .pareto-figure svg {{ display: block; width: 100%; height: auto; }}
      .pareto-grid {{ stroke: var(--pl-rule); stroke-width: 1; }}
      .pareto-axis {{ stroke: var(--pl-text); stroke-width: 1; }}
      .pareto-tick {{ fill: var(--pl-text-muted); font: .72rem var(--pl-font-mono); }}
      .pareto-axis-title {{ fill: var(--pl-text); font: 700 .71rem var(--pl-font-mono); letter-spacing: .06em; }}
      .pareto-frontier {{ fill: none; stroke: var(--pl-series-1); stroke-width: 3; stroke-dasharray: 7 5; }}
      .pareto-point {{ stroke: var(--pl-bg); stroke-width: 2; }}
      .pareto-point.frontier, .pareto-swatch.frontier {{ fill: var(--pl-series-1); background: var(--pl-series-1); }}
      .pareto-point.dominated, .pareto-swatch.dominated {{ fill: var(--pl-series-2); background: var(--pl-series-2); }}
      .pareto-leader {{ stroke: var(--pl-text-muted); stroke-width: 1; stroke-linecap: round; }}
      .pareto-label {{ fill: var(--pl-text); font: 600 .75rem var(--pl-font-body); }}
      .pareto-legend {{ display: flex; flex-wrap: wrap; gap: 1.25rem; margin-top: .9rem; color: var(--pl-text-muted); font-size: .82rem; }}
      .pareto-legend span {{ display: inline-flex; align-items: center; gap: .45rem; }}
      .pareto-swatch {{ display: inline-block; width: .65rem; height: .65rem; border-radius: 50%; }}
      @media (max-width: 42rem) {{ .pareto-heading {{ display: block; }} .pareto-meta {{ margin-top: .5rem; white-space: normal; }} .pareto-label, .pareto-leader {{ display: none; }} }}
    </style>
  </head>
  <body data-scheme="technical">
    <a class="skip-link" href="#main">Skip to content</a>
    <header class="site-header">
      <div class="wrap">
        <a class="brand" href="/"><span class="brand-mark" aria-hidden="true"></span>Plicara Labs</a>
        <nav class="site-nav" aria-label="Primary">
          <a href="/#mission">Mission</a><a href="/#models">Models</a><a href="/#tools">Tools</a><a href="/research/">Research</a><a href="/benchmarks/">Benchmarks</a><a href="/#principles">Principles</a><a href="https://github.com/plicara">GitHub</a>
        </nav>
      </div>
    </header>
    <main class="wrap results" id="main">
      <p class="eyebrow">benchmarks &middot; adventurebench</p>
      <h1>Can your model play a text adventure?</h1>
      <p class="lede">Adventure Bench measures grounded action interpretation: map a player utterance onto the visible scene&rsquo;s small action vocabulary, or refuse when the request is not grounded.</p>
{pareto_section}
      <div class="table-wrap">
        <table class="data-table">
          <caption class="visually-hidden">Audited Adventure Bench release {_escaped(release_id)}. Models are alphabetical by their canonical slug; comparison counts show only paired bootstrap intervals that exclude zero.</caption>
          <thead><tr><th scope="col">Model</th><th scope="col" class="num">Score (95% CI)</th><th scope="col" class="num">Passed</th><th scope="col" class="num">Recorded cost</th><th scope="col">Provider / runtime</th><th scope="col" class="num">Parse / transport</th><th scope="col" class="num">Separates</th></tr></thead>
          <tbody>
{model_rows}
          </tbody>
        </table>
      </div>
      <p class="note"><strong>These results are not a ranking.</strong> The separates column is +ahead / &minus;behind and counts only paired 95% case-cluster bootstrap intervals that exclude zero. Every comparison whose interval includes zero is unresolved; unresolved comparisons are unresolved.</p>
      <p class="note">Release <code>{_escaped(release_id)}</code> covers {_escaped(case_count)} cases &times; {_escaped(len(repetitions))} repetitions ({_escaped(case_repetitions)} case-repetitions; repetition numbers {_escaped(repetitions_label)}), {_escaped(collection_label)}. Benchmark version: <code>{_escaped(version)}</code>. <a href="{release_url}">Open the audited release artifact.</a></p>
      <p class="note">Failure policy: any transport failure invalidates a collection run and prevents it from entering this release. Parsing failures are counted separately after the frozen malformed-output retry and remain visible in the table. See the <a href="{METHODOLOGY_URL}">full methodology</a>.</p>
      <p class="note">Every number above regenerates offline from committed raw responses. The release pins every manifest and response hash plus dataset <code>{_escaped(dataset_hash[:12])}&hellip;</code> and prompt <code>{_escaped(prompt_hash[:12])}&hellip;</code>. Audit it with <code>make site-check RELEASE_ID={_escaped(release_id)}</code>.</p>
      <div class="table-wrap">
        <table class="data-table">
          <caption>Per-tag passed/total coverage for the audited release. Columns are models in alphabetical canonical-slug order.</caption>
          <thead><tr><th scope="col">Tag</th>
{tag_headings}
          </tr></thead>
          <tbody>
{tag_rows}
          </tbody>
        </table>
      </div>
      <p class="note">Per-tag cells are numerators and denominators, not pooled rankings; they show which grounded mapping and calibration patterns each audited run passed.</p>
      <div class="cta-row">
        <a class="btn btn-primary" href="{PROJECT_URL}">Repository</a>
        <a class="btn btn-ghost" href="{raw_evidence_url}">Raw evidence</a>
        <a class="btn btn-ghost" href="{DATASET_CARD_URL}">Dataset Card</a>
        <a class="btn btn-ghost" href="{RERUN_URL}">Re-run it</a>
        <a class="btn btn-ghost" href="/benchmarks/">&larr; All benchmarks</a>
      </div>
    </main>
    <footer class="site-footer">
      <div class="wrap"><span>&copy; Plicara Labs</span><div class="footer-links"><a href="mailto:info@plicara.ai">info@plicara.ai</a><a href="https://github.com/plicara">GitHub</a><a href="/">Home</a></div></div>
    </footer>
  </body>
</html>
'''


def render_audited_release(*, release_id: str, runs_dir: Path, results_dir: Path,
                           releases_dir: Path, dataset_path: Path) -> str:
    """Replay a release and return its one canonical publication page.

    This deliberately performs no site I/O.  Publication handoff code can use
    it to stage a complete site change before writing any destination file.
    """
    run_ids = _release_identity(releases_dir, release_id)
    try:
        artifact = aggregate_release(
            runs_dir=runs_dir,
            results_dir=results_dir,
            releases_dir=releases_dir,
            release_id=release_id,
            run_ids=run_ids,
            dataset_path=dataset_path,
            check=True,
        )
    except (ReleaseError, ValueError) as err:
        raise SiteRenderError(f"release cannot be rendered: {err}") from err
    return render_html(artifact)


def render_site(*, release_id: str, runs_dir: Path, results_dir: Path, releases_dir: Path,
                dataset_path: Path, site_dir: Path, check: bool = False) -> Path:
    """Revalidate a release and write (or parity-check) its static HTML page."""
    output_path = _inside(site_dir, site_dir / "benchmarks" / "adventurebench" / "index.html")
    rendered = render_audited_release(
        release_id=release_id,
        runs_dir=runs_dir,
        results_dir=results_dir,
        releases_dir=releases_dir,
        dataset_path=dataset_path,
    )
    if check:
        try:
            current = output_path.read_text(encoding="utf-8")
        except OSError as err:
            raise SiteRenderError(f"site render drift: {output_path}") from err
        if current != rendered:
            raise SiteRenderError(f"site render drift: {output_path}")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
    return output_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Render a checked Adventure Bench release as a static site page.")
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--releases-dir", default="releases")
    parser.add_argument("--data", default="adventure_bench/data/cases.jsonl")
    parser.add_argument("--site-dir", default="site")
    parser.add_argument("--check", action="store_true", help="fail if the checked page differs from the deterministic render")
    args = parser.parse_args(argv)
    try:
        output = render_site(
            release_id=args.release_id,
            runs_dir=Path(args.runs_dir),
            results_dir=Path(args.results_dir),
            releases_dir=Path(args.releases_dir),
            dataset_path=Path(args.data),
            site_dir=Path(args.site_dir),
            check=args.check,
        )
    except (SiteRenderError, ValueError) as err:
        parser.error(str(err))
    print(json.dumps({"release_id": args.release_id, "output": str(output), "check": args.check}, sort_keys=True))


if __name__ == "__main__":
    main()
