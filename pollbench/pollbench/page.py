"""Render the benchmark ratings as a plicara.ai results page.

The page follows the site's results pages (``regexeval-2026``,
``adventurebench``): inline SVG figures, plain notes. It is written to be read
top to bottom by someone who has not seen the method: each term is explained
the first time it is used, and the full table sits in a collapsed section at
the end of the ratings. Benchmarks are reported in groups by what the posterior
resolves, not as a ranking, and within a group they are alphabetical.

    python -m pollbench.page ratings.json --out site/index.html [--placeholder TEXT]

``--placeholder`` marks the page as a preview built on non-measured numbers:
it adds a visible notice and a noindex tag.
"""

from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from .figures import curve_panels, grade_strip, methods_chart
from .ratings import GRADE_BANDS, GRADE_PROB, MAX_RHAT, MIN_MODELS_FOR_GRADE

TITLE = "A pollster-style rating for AI benchmarks"
GROUP_TEXT = {
    "A": f"Reliability clears {GRADE_BANDS[0][0]:.2f} with at least {GRADE_PROB:.0%} probability",
    "B": f"Clears {GRADE_BANDS[1][0]:.2f}, not {GRADE_BANDS[0][0]:.2f}",
    "C": f"Clears {GRADE_BANDS[2][0]:.2f}, not {GRADE_BANDS[1][0]:.2f}",
    "D": f"Not shown to clear {GRADE_BANDS[2][0]:.2f} with {GRADE_PROB:.0%} probability",
    "ungraded": f"Not graded: fewer than {MIN_MODELS_FOR_GRADE} models, or the sampler did not converge (R-hat above {MAX_RHAT})",
}
ORDER = ["A", "B", "C", "D", "ungraded"]
# Sources, numbered in the order the page first cites them. Footnote markup follows the site's research pages.
REFERENCES = [
    'Nate Silver and Eli McKown-Dawson. <a href="https://www.natesilver.net/p/pollster-ratings-silver-bulletin">Silver Bulletin pollster ratings, January 2026 update</a>. Silver Bulletin, 2026.',
    "Frederic M. Lord and Melvin R. Novick. Statistical Theories of Mental Test Scores. Addison-Wesley, 1968.",
    'Epoch AI. <a href="https://epoch.ai/benchmarks/use-this-data">Capabilities &amp; benchmarking</a>. Published online at epoch.ai, accessed 10 October 2026. CC BY 4.0.',
    'Epoch AI. <a href="https://epoch.ai/eci">Epoch Capabilities Index</a>. Published online at epoch.ai, 2025.',
    'Anson Ho, Jean-Stanislas Denain, David Atanasov, Samuel Albanie and Rohin Shah. <a href="https://arxiv.org/abs/2512.00193">A Rosetta Stone for AI Benchmarks</a>. arXiv:2512.00193, 2025.',
    'Tilmann Gneiting and Adrian E. Raftery. <a href="https://doi.org/10.1198/016214506000001437">Strictly Proper Scoring Rules, Prediction, and Estimation</a>. Journal of the American Statistical Association, 2007.',
    'Aki Vehtari, Andrew Gelman, Daniel Simpson, Bob Carpenter and Paul-Christian B&uuml;rkner. <a href="https://doi.org/10.1214/20-BA1221">Rank-Normalization, Folding, and Localization: An Improved R-hat for Assessing Convergence of MCMC</a>. Bayesian Analysis, 2021.',
]


def cite(n: int, first: bool = True) -> str:
    """Footnote marker for reference ``n`` (1-based); only the first citation carries the back-link anchor."""
    anchor = f' id="fnref:{n}"' if first else ""
    return f'<sup{anchor}><a class="footnote-ref" href="#fn:{n}">{n}</a></sup>'


def references() -> str:
    items = "".join(f'<li id="fn:{i}"><p>{ref}&#160;<a class="footnote-backref" href="#fnref:{i}" title="Jump back to footnote {i} in the text">&#8617;</a></p></li>'
                    for i, ref in enumerate(REFERENCES, 1))
    return f'<div class="footnote"><hr><ol>{items}</ol></div>'


METHOD_TEXT = {
    "hier_beta": "The rating model",
    "eci_ls": "Epoch Capabilities Index form",
    "mean_rank": "Average standardised score",
    "benchmark_mean": "Guess the benchmark&rsquo;s average",
}

STYLE = """
      .results { grid-template-columns: minmax(0, 1fr); }
      .data-table td.num { white-space: nowrap; }
      .data-table th[scope="col"] { white-space: nowrap; }
      .rating-section { margin: 3.6rem 0 0; }
      /* One column for the whole page: headings, prose, figures and tables share one width, wide enough that the figures read at their drawn size (56rem less the figure padding is the 840-unit drawing width). */
      .results { --pb-measure: 56rem; width: min(100% - 2.5rem, var(--pb-measure)); }
      .rating-preview { box-sizing: border-box; }
      .results .note { max-width: none; }  /* notes fill the column like the prose around them */
      .results sup { line-height: 0; font-size: .72em; vertical-align: super; }
      .results .footnote { font-size: .88rem; color: var(--pl-text-muted); max-width: var(--pb-measure); }
      .results .footnote hr { border: 0; border-top: 1px solid var(--pl-rule); margin: 1.2rem 0; }
      .results .footnote li p { margin: 0 0 .5rem; }
      .rating-figure { border: 1px solid var(--pl-rule); background: var(--pl-surface); padding: clamp(.8rem, 3vw, 1.75rem); margin: 1.2rem 0 .8rem; }
      .rating-figure svg { display: block; width: 100%; height: auto; }
      .rating-figure svg.compact { display: none; }
      @media (max-width: 640px) {
        .rating-figure svg.wide { display: none; }
        .rating-figure svg.compact { display: block; }
      }
      .figure-title { font: 700 .78rem var(--pl-font-mono); letter-spacing: .04em; margin: 0 0 .7rem; }
      .legend { display: flex; flex-wrap: wrap; gap: .5rem 1.4rem; margin: 0 0 1rem; padding: 0; list-style: none; font-size: .86rem; color: var(--pl-text-muted); }
      .legend li { display: flex; align-items: center; gap: .5rem; }
      .legend svg { width: 34px; height: 14px; flex: none; }
      .rf-grid { stroke: var(--pl-rule); stroke-width: 1; }
      .rf-cut { stroke: var(--pl-text-muted); stroke-width: 1; stroke-dasharray: 3 4; }
      .rf-axis { stroke: var(--pl-text); stroke-width: 1; }
      .rf-band { fill: var(--pl-text-muted); opacity: .07; }
      .rf-tick { fill: var(--pl-text-muted); font: .72rem var(--pl-font-mono); }
      .rf-title { fill: var(--pl-text); font: 700 .71rem var(--pl-font-mono); letter-spacing: .04em; }
      .rf-hint { fill: var(--pl-text-muted); font: .7rem var(--pl-font-mono); }
      .rf-label { fill: var(--pl-text); font: 500 .74rem var(--pl-font-mono); }
      .rf-group { fill: var(--pl-text); font: 700 .72rem var(--pl-font-mono); letter-spacing: .04em; }
      .rf-bandlabel { fill: var(--pl-text); font: 700 .74rem var(--pl-font-mono); }
      .rf-bar { stroke: var(--pl-series-1); stroke-width: 3; stroke-linecap: round; }
      .rf-dot { fill: var(--pl-series-1); stroke: var(--pl-surface); stroke-width: 2; }
      .rf-bar { stroke-width: 6; opacity: .5; }
      .rf-ring { fill: var(--pl-surface); stroke: var(--pl-text); stroke-width: 1.6; }
      .rf-bar.off { stroke: var(--pl-text-muted); opacity: .45; } .rf-dot.off { fill: var(--pl-text-muted); }
      .rf-signal { fill: var(--pl-series-1); } .rf-noise { fill: var(--pl-series-2); }
      .rf-inbar { fill: var(--pl-bg); font: 700 .72rem var(--pl-font-mono); }
      .rf-hit { fill: transparent; }
      .rf-row:hover .rf-hit { fill: var(--pl-text); opacity: .06; }
      .rf-gA { fill: var(--pl-series-1); } .rf-gB { fill: var(--pl-series-1); opacity: .72; }
      .rf-gC { fill: var(--pl-series-1); opacity: .46; } .rf-gD { fill: var(--pl-series-1); opacity: .24; }
      .rf-off { fill: var(--pl-text-muted); opacity: .25; }
      .rf-gA-ink, .rf-gB-ink { fill: var(--pl-bg); opacity: 1; } .rf-gC-ink, .rf-gD-ink, .rf-off-ink { fill: var(--pl-text); opacity: 1; }
      .rf-band-A { fill: var(--pl-series-1); opacity: .13; } .rf-band-B { fill: var(--pl-series-1); opacity: .09; }
      .rf-band-C { fill: var(--pl-series-1); opacity: .05; } .rf-band-D { fill: var(--pl-series-1); opacity: .015; }
      .rf-gap { stroke: var(--pl-text-muted); stroke-width: 1.2; stroke-dasharray: 2 3; }
      .rf-strip-n { font: 700 1.05rem var(--pl-font-mono); }
      .rf-panel { fill: var(--pl-text); font: 700 .86rem var(--pl-font-mono); }
      .rf-noiseband { fill: var(--pl-series-2); opacity: .24; }
      .rf-curve { fill: none; stroke: var(--pl-text); stroke-width: 2; stroke-linejoin: round; }
      .rf-pt { fill: var(--pl-series-1); fill-opacity: .7; stroke: var(--pl-surface); stroke-width: 1; }
      .rf-pt:hover { fill-opacity: 1; stroke: var(--pl-text); }
      .rf-whisk { stroke: var(--pl-series-1); stroke-width: 1.4; opacity: .45; }
      .rf-bub { fill: var(--pl-series-1); fill-opacity: .78; stroke: var(--pl-surface); stroke-width: 1.5; }
      g:hover > .rf-bub { fill-opacity: 1; stroke: var(--pl-text); }
      .rf-trend { stroke: var(--pl-series-2); stroke-width: 2; stroke-dasharray: 6 5; }
      .rf-target { fill: var(--pl-series-1); opacity: .12; }
      .rf-mbar { fill: var(--pl-text-muted); opacity: .45; } .rf-mbar.mine { fill: var(--pl-series-2); opacity: 1; }
      .rf-mdot { fill: var(--pl-text-muted); } .rf-mdot.mine { fill: var(--pl-series-2); }
      .rf-label.mine { font-weight: 700; }
      .panels { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1.2rem 1.6rem; }
      @media (max-width: 640px) { .panels { grid-template-columns: minmax(0, 1fr); } }
      .strip { display: flex; gap: 3px; margin: .2rem 0 .4rem; }
      .strip .seg { min-width: 3.2rem; border-radius: 4px; padding: .55rem .7rem .5rem; display: flex; flex-direction: column; gap: .1rem; }
      .strip .seg-n { font: 700 1.5rem/1 var(--pl-font-mono); }
      .strip .seg-l { font: .72rem var(--pl-font-mono); }
      .seg-A, .seg-B { background: var(--pl-series-1); color: var(--pl-bg); }
      .seg-B { background: color-mix(in srgb, var(--pl-series-1) 72%, var(--pl-surface)); }
      .seg-C { background: color-mix(in srgb, var(--pl-series-1) 46%, var(--pl-surface)); color: var(--pl-text); }
      .seg-D { background: color-mix(in srgb, var(--pl-series-1) 24%, var(--pl-surface)); color: var(--pl-text); }
      .seg-ungraded { background: transparent; border: 1px dashed var(--pl-text-muted); color: var(--pl-text-muted); }
      .hero-stats { display: flex; flex-wrap: wrap; gap: .4rem 2rem; margin: 0 0 1.1rem; font: .8rem var(--pl-font-mono); color: var(--pl-text-muted); }
      .hero-stats b { display: block; font: 700 1.6rem/1.1 var(--pl-font-mono); color: var(--pl-text); }
      .rating-preview { border-left: 4px solid var(--pl-orange); padding: .6rem 1rem; background: var(--pl-surface); margin: 0 0 1.6rem; }
      .terms dt { font-weight: 700; margin-top: .9rem; }
      .terms dd { margin: .2rem 0 0; }
      .rating-section h3 { font-size: 1.15rem; margin: 1.8rem 0 .2rem; }
      .formula { font: .95rem var(--pl-font-mono); background: var(--pl-surface); border-left: 3px solid var(--pl-series-1); padding: .7rem 1rem; }
      .grade-table { max-width: var(--pb-measure); min-width: 0; }  /* three short columns fit a phone without scrolling */
      .grade-table tbody th { text-transform: none; font-family: var(--pl-font-mono); }
      .grade-table td { color: var(--pl-text); }
      .steps li { margin: .45rem 0; }
      details.full-table { margin: 1.6rem 0 0; border-top: 1px solid var(--pl-rule); padding-top: .8rem; }
      details.full-table summary { cursor: pointer; font-weight: 700; }
      details.full-table[open] summary { margin-bottom: 1rem; }
"""


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def _fx(x: float, x0: float, x1: float) -> float:
    return x0 + (x1 - x0) * min(max(x, 0.0), 1.0)


def _skill(data: dict, name: str):
    got = data.get("heldout", {}).get("kfold", {}).get("skill", {}).get(name)
    return got  # [skill, lo, hi, n] or None


def _group(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {g: [] for g in ORDER}
    for r in rows:
        out[r["grade"] if r["grade"] in out else "ungraded"].append(r)
    for g in out:
        out[g].sort(key=lambda r: r["benchmark"].lower())
    return out


def _headroom(r: dict) -> str:
    if r.get("recent_ceiling_share") is None:
        return "unknown (no recent models)"
    return "saturated" if r["saturated"] else ("near ceiling" if r["ceiling_share"] >= 0.25 else "open")


def _band_edges() -> list[tuple[str, float, float]]:
    """(letter, low, high) for each grade band on the reliability axis, D first."""
    cuts = sorted(GRADE_BANDS)  # ascending thresholds
    edges = [("D", 0.0, cuts[0][0])]
    for i, (thr, letter) in enumerate(cuts):
        edges.append((letter, thr, cuts[i + 1][0] if i + 1 < len(cuts) else 1.0))
    return edges


def table(data: dict, groups: dict[str, list[dict]]) -> str:
    n_total = sum(len(v) for v in groups.values())
    parts = [
        '<div class="table-wrap" tabindex="0" role="region" aria-label="Ratings for every benchmark">',
        '<table class="data-table">',
        f'<caption class="visually-hidden">{n_total} benchmarks grouped by grade, alphabetical within a group.</caption>',
        "<thead><tr>"
        '<th scope="col">Benchmark</th><th scope="col" class="num">AI models</th>'
        '<th scope="col" class="num">reliability (90% CI)</th>'
        '<th scope="col" class="num">if every AI model scored</th>'
        '<th scope="col" class="num">noise (points)</th><th scope="col" class="num">held-out skill</th>'
        '<th scope="col">headroom</th></tr></thead>',
    ]
    for g in ORDER:
        if not groups[g]:
            continue
        parts.append('<tbody class="group">')
        head = "Not graded" if g == "ungraded" else f"Grade {g}"
        parts.append(f'<tr class="group-head"><th scope="rowgroup" colspan="7">{head} &middot; {escape(GROUP_TEXT[g])} &middot; {len(groups[g])}</th></tr>')
        for r in groups[g]:
            sk = _skill(data, r["benchmark"])
            skill = f"{sk[0]:+.2f}" if sk else "&ndash;"
            allm = r.get("all_models_reliability_mean")
            allm = f"{allm:.2f}" if allm is not None and g != "ungraded" else "&ndash;"
            parts.append(
                f'<tr><th scope="row" class="mono">{escape(r["benchmark"])}</th>'
                f'<td class="num mono">{r["n_models"]}</td>'
                + (f'<td class="num mono">{r["reliability_mean"]:.2f} ({r["reliability_lo"]:.2f}&ndash;{r["reliability_hi"]:.2f})</td>'
                   if g != "ungraded" else '<td class="num mono">not graded</td>')
                + f'<td class="num mono">{allm}</td>'
                f'<td class="num mono">{100 * r["noise_points"]:.1f}</td>'
                f'<td class="num mono">{skill}</td><td>{_headroom(r)}</td></tr>'
            )
        parts.append("</tbody>")
    parts += ["</table>", "</div>"]
    return "\n".join(parts)


def examples(groups: dict[str, list[dict]]) -> list[dict]:
    """One benchmark per grade to teach with: the one scored on the most models."""
    return [max(groups[g], key=lambda r: r["n_models"]) for g in "ABCD" if groups[g]]


def split_bars(rows: list[dict], *, compact: bool = False) -> str:
    """100% bars splitting each example benchmark's between-model spread into capability and noise."""
    if compact:
        width, left, right, step, top = 360, 4, 356, 62, 22
    else:
        width, left, right, step, top = 840, 250, 812, 44, 10
    h = top + step * len(rows) + (0 if compact else 16)
    sfx = "-compact" if compact else ""
    out = [f'<svg class="{"compact" if compact else "wide"}" viewBox="0 0 {width} {h}" role="img" aria-labelledby="split-title{sfx} split-desc{sfx}">',
           f'<title id="split-title{sfx}">How the spread of scores splits into capability and noise, for four benchmarks</title>',
           f'<desc id="split-desc{sfx}">' + "; ".join(
               f'{escape(r["benchmark"])}: {_pct(r["reliability_mean"])} capability, {_pct(1 - r["reliability_mean"])} noise' for r in rows) + "</desc>"]
    for i, r in enumerate(rows):
        y = top + step * i
        rel = r["reliability_mean"]
        xm = _fx(rel, left, right)
        if compact:
            out.append(f'<text class="rf-label" x="{left}" y="{y - 7}">{escape(r["benchmark"])} <tspan class="rf-hint">grade {r["grade"]} &middot; {r["n_models"]} models</tspan></text>')
        else:
            out.append(f'<text class="rf-label" x="{left - 12}" y="{y + 15}" text-anchor="end">{escape(r["benchmark"])}</text>')
            out.append(f'<text class="rf-hint" x="{left - 12}" y="{y + 29}" text-anchor="end">grade {r["grade"]} &middot; {r["n_models"]} models</text>')
        out.append(f'<rect class="rf-signal" x="{left}" y="{y}" width="{xm - left - 1:.1f}" height="24" rx="3" />')
        out.append(f'<rect class="rf-noise" x="{xm + 1:.1f}" y="{y}" width="{right - xm - 1:.1f}" height="24" rx="3" />')
        cap = _pct(rel) if compact else f"capability {_pct(rel)}"
        noise = _pct(1 - rel) if compact else f"noise {_pct(1 - rel)}"
        fits = right - xm > 8 * len(noise) + 8
        if not fits:  # the noise segment is too thin to hold its own label: name both inside the capability segment
            cap += f" &middot; {noise}"
        out.append(f'<text class="rf-inbar" x="{left + 8}" y="{y + 16}">{cap}</text>')
        if fits:
            out.append(f'<text class="rf-inbar" x="{right - 8}" y="{y + 16}" text-anchor="end">{noise}</text>')
    out.append("</svg>")
    return "\n".join(out)


def _legend() -> str:
    dot = '<svg viewBox="0 0 34 14" aria-hidden="true"><circle class="rf-dot" cx="17" cy="7" r="4.5" /></svg>'
    bar = '<svg viewBox="0 0 34 14" aria-hidden="true"><line class="rf-bar" x1="4" y1="7" x2="30" y2="7" /></svg>'
    ring = '<svg viewBox="0 0 34 14" aria-hidden="true"><line class="rf-gap" x1="6" y1="7" x2="26" y2="7" /><circle class="rf-dot" cx="6" cy="7" r="4" /><circle class="rf-ring" cx="27" cy="7" r="3.6" /></svg>'
    band = '<svg viewBox="0 0 34 14" aria-hidden="true"><line class="rf-cut" x1="17" y1="0" x2="17" y2="14" /></svg>'
    return ('<ul class="legend">'
            f"<li>{dot}reliability among the AI models scored</li>"
            f"<li>{bar}90% credible interval</li>"
            f"<li>{ring}reliability if every AI model had been scored</li>"
            f"<li>{band}grade threshold</li></ul>")


def forest(groups: dict[str, list[dict]], *, compact: bool = False) -> str:
    """The ratings figure. ``compact`` is the phone layout: names above their bars instead of beside them."""
    rows: list[tuple[str, dict | None]] = []
    for g in ORDER:
        if groups[g]:
            rows.append((g, None))
            rows += [(g, r) for r in groups[g]]
    if compact:
        width, step, top, left, right, bottom, ticks = 360, 36, 76, 10, 346, 72, (0.0, 1.0)
    else:
        width, step, top, left, right, bottom, ticks = 840, 22, 76, 300, 812, 64, (0.0, 0.25, 1.0)
    h = top + step * len(rows) + bottom
    x0, x1 = left, right
    y_end = h - bottom + 4
    sfx = "-compact" if compact else ""
    out = [f'<svg class="{"compact" if compact else "wide"}" viewBox="0 0 {width} {h}" role="img" aria-labelledby="forest-title{sfx} forest-desc{sfx}">',
           f'<title id="forest-title{sfx}">Reliability of each benchmark with its 90% credible interval, grouped by grade</title>',
           f'<desc id="forest-desc{sfx}">One row per benchmark, grouped by grade. The dot is the best estimate of reliability among the models scored and the bar is the 90% credible interval; the open ring is the estimate if every model in the table had been scored; shaded bands and dashed lines mark the grade thresholds.</desc>']
    for letter, lo, hi in _band_edges():
        xa, xb = _fx(lo, x0, x1), _fx(hi, x0, x1)
        out.append(f'<rect class="rf-band rf-band-{letter}" x="{xa:.1f}" y="{top - 28}" width="{xb - xa:.1f}" height="{y_end - top + 28}" />')
        out.append(f'<rect class="rf-g{letter}" x="{(xa + xb) / 2 - 11:.1f}" y="{top - 52}" width="22" height="22" rx="4" />')
        out.append(f'<text class="rf-bandlabel rf-g{letter}-ink" x="{(xa + xb) / 2:.1f}" y="{top - 36}" text-anchor="middle">{letter}</text>')
    for t in sorted(set(ticks) | {thr for thr, _ in GRADE_BANDS}):
        x = _fx(t, x0, x1)
        anchor = "start" if compact and t == 0 else ("end" if compact and t == 1 else "middle")
        out.append(f'<line class="rf-grid" x1="{x:.1f}" y1="{top - 28}" x2="{x:.1f}" y2="{y_end}" />')
        out.append(f'<text class="rf-tick" x="{x:.1f}" y="{y_end + 16}" text-anchor="{anchor}">{t:.2f}</text>')
        out.append(f'<text class="rf-tick" x="{x:.1f}" y="{top - 14}" text-anchor="{anchor}">{t:.2f}</text>')
    if not compact:
        out.append(f'<text class="rf-title" x="{x0 - 10}" y="{top - 36}" text-anchor="end">GRADE BAND</text>')
    for thr, _ in GRADE_BANDS:
        x = _fx(thr, x0, x1)
        out.append(f'<line class="rf-cut" x1="{x:.1f}" y1="{top - 28}" x2="{x:.1f}" y2="{y_end}" />')
    out.append(f'<line class="rf-axis" x1="{x0}" y1="{y_end}" x2="{x1}" y2="{y_end}" />')
    if compact:
        out.append(f'<text class="rf-title" x="{(x0 + x1) / 2:.0f}" y="{y_end + 36}" text-anchor="middle">RELIABILITY</text>')
    else:
        out.append(f'<text class="rf-title" x="{(x0 + x1) / 2:.0f}" y="{y_end + 36}" text-anchor="middle">RELIABILITY: SHARE OF THE SPREAD BETWEEN MODELS THAT IS CAPABILITY</text>')
    out.append(f'<text class="rf-hint" x="{x0}" y="{y_end + 54}">&larr; 0 = all noise</text>')
    out.append(f'<text class="rf-hint" x="{x1}" y="{y_end + 54}" text-anchor="end">1 = all capability &rarr;</text>')
    for i, (g, r) in enumerate(rows):
        y = top + step * i + step / 2
        if r is None:
            label = "NOT GRADED" if g == "ungraded" else f"GRADE {g}"
            gx = x0 if compact else 8
            chip = "rf-off" if g == "ungraded" else f"rf-g{g}"
            out.append(f'<rect class="{chip}" x="{gx}" y="{y - 6:.1f}" width="5" height="12" rx="1.5" />')
            out.append(f'<text class="rf-group" x="{gx + 12}" y="{y + 4:.1f}">{label} &middot; {len(groups[g])}</text>')
            continue
        name = escape(r["benchmark"])
        allm = r.get("all_models_reliability_mean")
        tip = (f'{name}: reliability {r["reliability_mean"]:.2f}, 90% credible interval {r["reliability_lo"]:.2f} to {r["reliability_hi"]:.2f}; '
               f'{r["n_models"]} AI models scored; noise {100 * r["noise_points"]:.1f} points'
               + (f'; {allm:.2f} if every AI model had been scored' if allm is not None and g != "ungraded" else ""))
        yb = y + 6 if compact else y
        label = (f'<text class="rf-label" x="{x0}" y="{y - 5:.1f}">{name}</text>' if compact
                 else f'<text class="rf-label" x="{left - 10}" y="{y + 4:.1f}" text-anchor="end">{name}</text>')
        out.append(f'<g class="rf-row"><title>{tip}</title>'
                   f'<rect class="rf-hit" x="0" y="{y - step / 2:.1f}" width="{width}" height="{step}" />' + label)
        if g == "ungraded":  # no point estimate for a benchmark the protocol does not grade
            out.append(f'<line class="rf-bar off" x1="{_fx(r["reliability_lo"], x0, x1):.1f}" y1="{yb:.1f}" x2="{_fx(r["reliability_hi"], x0, x1):.1f}" y2="{yb:.1f}" /></g>')
            continue
        xd = _fx(r["reliability_mean"], x0, x1)
        gap = (f'<line class="rf-gap" x1="{xd:.1f}" y1="{yb:.1f}" x2="{_fx(allm, x0, x1):.1f}" y2="{yb:.1f}" />'
               f'<circle class="rf-ring" cx="{_fx(allm, x0, x1):.1f}" cy="{yb:.1f}" r="3.6" />') if allm is not None else ""
        out.append(f'<line class="rf-bar" x1="{_fx(r["reliability_lo"], x0, x1):.1f}" y1="{yb:.1f}" x2="{_fx(r["reliability_hi"], x0, x1):.1f}" y2="{yb:.1f}" />'
                   + gap + f'<circle class="rf-dot" cx="{xd:.1f}" cy="{yb:.1f}" r="5" />' + "</g>")
    out.append("</svg>")
    return "\n".join(out)


def calibration(data: dict, rows: list[dict], *, compact: bool = False) -> tuple[str, float | None, int]:
    n_of = {r["benchmark"]: r["n_models"] for r in rows}
    pts = [(r["benchmark"], r["reliability_mean"], _skill(data, r["benchmark"])) for r in rows]
    pts = [(n, rel, sk) for n, rel, sk in pts if sk]
    if len(pts) < 4:
        return "", None, len(pts)
    rho = float(spearmanr([p[1] for p in pts], [p[2][0] for p in pts]).statistic)
    lo = min(-0.2, min(p[2][1] for p in pts))
    hi = max(0.6, max(p[2][2] for p in pts))
    if compact:
        width, height, left, right, top, bottom = 400, 420, 76, 392, 20, 340
    else:
        width, height, left, right, top, bottom = 840, 440, 84, 800, 20, 360
    sfx = "-compact" if compact else ""

    def fy(v):
        return bottom - (bottom - top) * (v - lo) / (hi - lo)

    out = [f'<svg class="{"compact" if compact else "wide"}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="cal-title{sfx} cal-desc{sfx}">',
           f'<title id="cal-title{sfx}">Reliability from the model against how well hidden scores were predicted</title>',
           f'<desc id="cal-desc{sfx}">One point per benchmark. Horizontal position is reliability; vertical position is held-out skill with a 90% interval. Rank correlation {rho:.2f} over {len(pts)} benchmarks.</desc>']
    for t in ((0, .5, 1) if compact else (0, .25, .5, .75, 1)):
        x = _fx(t, left, right)
        anchor = "end" if compact and t == 1 else "middle"
        out.append(f'<line class="rf-grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" />')
        out.append(f'<text class="rf-tick" x="{x:.1f}" y="{bottom + 18}" text-anchor="{anchor}">{t:.2f}</text>')
    step = 0.2
    v = round(lo / step) * step
    while v <= hi + 1e-9:
        y = fy(v)
        if top <= y <= bottom:
            out.append(f'<line class="rf-grid" x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" />')
            out.append(f'<text class="rf-tick" x="{left - 8}" y="{y + 4:.1f}" text-anchor="end">{v:+.1f}</text>')
        v += step
    out.append(f'<line class="rf-cut" x1="{left}" y1="{fy(0):.1f}" x2="{right}" y2="{fy(0):.1f}" />')
    out.append(f'<text class="rf-hint" x="{right - 4}" y="{fy(0) + 14:.1f}" text-anchor="end">{"0 = no better than the average" if compact else "0 = no better than guessing the benchmark&rsquo;s average"}</text>')
    out.append(f'<line class="rf-axis" x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" /><line class="rf-axis" x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" />')
    out.append(f'<text class="rf-title" x="{(left + right) / 2:.0f}" y="{bottom + 44}" text-anchor="middle">{"RELIABILITY &rarr;" if compact else "RELIABILITY, FROM THE FITTED MODEL &rarr;"}</text>')
    out.append(f'<text class="rf-hint" x="{(left + right) / 2:.0f}" y="{bottom + 60}" text-anchor="middle">{"more of the spread is capability" if compact else "more of the spread between models is capability"}</text>')
    mid = (top + bottom) / 2
    ya, yh = (14, 30) if compact else (22, 40)
    out.append(f'<text class="rf-title" x="{ya}" y="{mid:.0f}" text-anchor="middle" transform="rotate(-90 {ya} {mid:.0f})">HELD-OUT SKILL &rarr;</text>')
    out.append(f'<text class="rf-hint" x="{yh}" y="{mid:.0f}" text-anchor="middle" transform="rotate(-90 {yh} {mid:.0f})">{"predicted better" if compact else "hidden scores predicted better"}</text>')
    xs, ys = np.array([p[1] for p in pts]), np.array([p[2][0] for p in pts])
    slope, icept = np.polyfit(xs, ys, 1)
    t0, t1 = xs.min(), xs.max()
    out.append(f'<line class="rf-trend" x1="{_fx(t0, left, right):.1f}" y1="{fy(icept + slope * t0):.1f}" x2="{_fx(t1, left, right):.1f}" y2="{fy(icept + slope * t1):.1f}" />')
    named = {max(pts, key=lambda p: p[2][0])[0]}
    if not compact:  # the phone layout has no room beside the low points
        named |= {min(pts, key=lambda p: p[1])[0], min(pts, key=lambda p: p[2][0])[0]}
    for n, rel, sk in pts:
        x = _fx(rel, left, right)
        out.append(f'<g><title>{escape(n)}: reliability {rel:.2f}; held-out skill {sk[0]:+.2f} (90% interval {sk[1]:+.2f} to {sk[2]:+.2f}) on {sk[3]} hidden scores</title>'
                   f'<line class="rf-whisk" x1="{x:.1f}" y1="{fy(sk[1]):.1f}" x2="{x:.1f}" y2="{fy(sk[2]):.1f}" />'
                   f'<circle class="rf-bub" cx="{x:.1f}" cy="{fy(sk[0]):.1f}" r="{2 + 0.45 * n_of[n] ** 0.5:.1f}" /></g>')
        if n in named:
            room_left = x - 9 - 7.2 * len(n) > left
            anchor, dx = ("end", -9) if room_left and (x < (left + right) / 2 or x + 9 + 7.2 * len(n) > right) else ("start", 9)
            out.append(f'<text class="rf-label" x="{x + dx:.1f}" y="{fy(sk[0]) + 4:.1f}" text-anchor="{anchor}">{escape(n)}</text>')
    out.append("</svg>")
    return "\n".join(out), rho, len(pts)


def render(data: dict, *, placeholder: str | None = None, curves: dict | None = None) -> str:
    rows = data["benchmarks"]
    groups = _group(rows)
    n_models, n_bench = data["n_models"], data["n_benchmarks"]
    graded = sum(len(groups[g]) for g in "ABCD")
    cal_svg, rho, n_cal = calibration(data, rows)
    cal_compact = calibration(data, rows, compact=True)[0]
    ex = examples(groups)
    sf = data.get("second_factor_check")
    pairs = data.get("residual_pairs", {}).get("most_positive", [])[:4]
    cond = data.get("conditions", {})
    (thr_a, _), (thr_b, _), (thr_c, _) = GRADE_BANDS

    factor_block = ""
    if sf:
        fl = data.get("floor_model_check")
        floor_note = ""
        if fl:
            f2 = fl["second_factor_check"]
            floor_note = (f" One candidate explanation is guessing: on a multiple-choice test, a model with no relevant skill still scores above zero. "
                          f"Refitting with a separate chance-level floor for every benchmark leaves the excess almost unchanged ({f2['statistic']:.3f} against {f2['reference_mean']:.3f}), so guessing is not the cause.")
        near = "; ".join(f"{escape(q['a'])} and {escape(q['b'])}" for q in pairs)
        found = sf["p_value"] < 0.05
        factor_block = f"""      <section class="rating-section" aria-labelledby="factor-heading">
        <h2 id="factor-heading">Limitation: capability is not exactly one number</h2>
        <p>The rating model gives every AI model a single capability score. If that assumption were exactly right, the part of each score the model cannot explain (the score minus its expected value) would be pure noise, and those leftovers would be unrelated from one benchmark to the next. If AI models also differ in some second skill, say coding or knowledge recall, the leftovers on two benchmarks that both reward that skill would rise and fall together.</p>
        <p>We tested this directly. For each of the {sf["n_pairs"]} pairs of benchmarks that share at least 12 AI models, we computed the correlation between their leftovers and averaged its square over all pairs. In the real data the average is {sf["statistic"]:.3f}. We then simulated {sf.get("n_rep", 500)} tables from the fitted rating model itself, where one capability explains everything by construction, and computed the same number: {sf["reference_mean"]:.3f} on average{f", and only {sf['p_value']:.1%} of simulated tables reached the real value" if found else ""}. {"So the real leftovers move together about " + f"{sf['statistic'] / sf['reference_mean']:.1f}" + " times as much as one capability allows: there is real structure beyond a single number." if found else "So the check finds no structure beyond a single number."}</p>
        <p>The pairs whose leftovers move together most are {near}. Most are older language-understanding benchmarks scored on the same group of older AI models.{floor_note}</p>
        <p>What this means for the ratings: if a benchmark mainly measures a skill that the other benchmarks do not, the rating model counts that skill as noise and the benchmark&rsquo;s reliability reads low. In our simulation study, adding a strong second skill had exactly this effect, and the held-out tests above did not catch it. A grade therefore says how well a benchmark measures general capability, the factor all {n_bench} benchmarks share. It is not a verdict on whether the benchmark is useful.</p>
      </section>
"""
    failed = [k for k, v in cond.items() if not v.get("holds", True)]
    reasons = []
    if failed:
        reasons.append("the rating model did not meet every pre-registered condition")
    if sf and sf["p_value"] < 0.05:
        reasons.append('AI model capability turns out not to be exactly one number in this data (<a href="#factor-heading">see the limitation below</a>)')
    exploratory = (f'      <p class="note rating-preview"><strong>Exploratory.</strong> Treat these ratings as a first release: '
                   f'{" and ".join(reasons)}. Each rating says how well a benchmark measures general capability, not whether the benchmark is useful.</p>\n'
                   if reasons else "")
    robots = '    <meta name="robots" content="noindex" />\n' if placeholder else ""
    banner = exploratory + (f'      <p class="note rating-preview"><strong>Preview.</strong> {escape(placeholder)}</p>\n'
                            if placeholder else "")

    glance = f"""      <section class="rating-section" aria-labelledby="glance-heading">
        <h2 id="glance-heading">At a glance</h2>
        <dl class="terms">
          <dt>What is rated</dt>
          <dd>Benchmarks, not AI models. Each of the {n_bench} benchmarks gets one rating.</dd>
          <dt>The score: reliability</dt>
          <dd>A number from 0 to 1. Take every AI model that has a score on a benchmark. Reliability is the share of the variation in those scores that is explained by the AI models&rsquo; general capability. The rest is noise. A reliability of 0.9 means 90% of the variation reflects real differences in capability and 10% is noise.</dd>
          <dt>Grades</dt>
          <dd>A, B, C or D, from the reliability and how certain it is. A benchmark gets a grade only when its reliability is at least that grade&rsquo;s threshold with 90% probability: A at {thr_a:.2f}, B at {thr_b:.2f}, C at {thr_c:.2f}.</dd>
          <dt>Uncertainty</dt>
          <dd>Every reliability is reported with a 90% credible interval: given the data, there is a 90% probability that the true value lies inside it.</dd>
          <dt>Data</dt>
          <dd>{data["n_cells"]:,} published scores for {n_models} AI models on {n_bench} benchmarks, from Epoch AI.{cite(3)}</dd>
        </dl>
      </section>
"""

    measure_block = ""
    if ex:
        top, low = ex[0], ex[-1]
        legend = ('<ul class="legend">'
                  '<li><svg viewBox="0 0 34 14" aria-hidden="true"><rect class="rf-signal" x="2" y="2" width="30" height="10" rx="2" /></svg>capability</li>'
                  '<li><svg viewBox="0 0 34 14" aria-hidden="true"><rect class="rf-noise" x="2" y="2" width="30" height="10" rx="2" /></svg>noise</li>')
        panels = curve_panels(ex, curves) if curves else ""
        if panels:
            example_figure = f"""        <div class="rating-figure">
          <p class="figure-title">Four benchmarks, one from each grade. Each dot is one AI model&rsquo;s published score, placed left to right by its estimated general capability.</p>
          {legend}<li><svg viewBox="0 0 34 14" aria-hidden="true"><circle class="rf-pt" cx="17" cy="7" r="4" /></svg>one AI model&rsquo;s score</li><li><svg viewBox="0 0 34 14" aria-hidden="true"><line class="rf-curve" x1="2" y1="11" x2="32" y2="3" /></svg>expected score at that capability</li><li><svg viewBox="0 0 34 14" aria-hidden="true"><rect class="rf-noiseband" x="2" y="2" width="30" height="10" /></svg>range holding 90% of scores, from noise alone</li></ul>
          <div class="panels">
{panels}
          </div>
        </div>
        <p><strong>How to read it.</strong> The curve is the score the rating model expects from an AI model of a given capability. The shaded band is how far real scores scatter around that curve. On {escape(top["benchmark"])}, the band is narrow and the AI models are spread far along the curve, so their scores differ mainly because their capability differs. On {escape(low["benchmark"])}, the band is wide compared with how far the AI models are spread, so two models with the same capability can easily score far apart. Hover a dot to see the AI model and its score.</p>
"""
        else:
            example_figure = f"""        <div class="rating-figure">
          <p class="figure-title">Four benchmarks, one from each grade: the variation in scores, split into capability and noise</p>
          {legend}</ul>
{split_bars(ex)}
{split_bars(ex, compact=True)}
        </div>
"""
        measure_block = f"""      <section class="rating-section" aria-labelledby="measure-heading">
        <h2 id="measure-heading">How good are benchmarks at distinguishing AI model capability?</h2>
        <p>When two AI models get different scores on a benchmark, part of the difference is real: one model is more capable than the other. Part of it is noise. Noise here means everything other than general capability:</p>
        <ul class="steps">
          <li>which questions happen to be in the test, and how many there are;</li>
          <li>differences in prompts, tools and settings between whoever ran each evaluation;</li>
          <li>randomness from one run to the next;</li>
          <li>skills that this benchmark rewards and the other benchmarks do not.</li>
        </ul>
        <p><strong>Reliability</strong> measures how much of the variation in a benchmark&rsquo;s scores is the real part. On {escape(top["benchmark"])}, {_pct(top["reliability_mean"])} of the variation in scores across its {top["n_models"]} AI models is explained by their general capability, so a gap between two models there is good evidence that one is more capable. On {escape(low["benchmark"])}, {_pct(low["reliability_mean"])} is, and {_pct(1 - low["reliability_mean"])} is noise, so the same gap means much less.</p>
{example_figure}
        <p class="note">This is the reliability coefficient of classical test theory{cite(2)}, applied to benchmarks instead of exams. It is not about how hard a benchmark is or how high AI models score on it. It is about how far a difference in score can be trusted.</p>
      </section>
"""

    c_eci, c3 = data.get("condition_1b_vs_eci_ls"), cond.get("3_temporal_stability_spearman_ge_0.70")
    gaps = [r for r in rows if r["grade"] in "ABCD" and r.get("all_models_reliability_mean") is not None]
    rr = max(gaps, key=lambda r: r["all_models_reliability_mean"] - r["reliability_mean"]) if gaps else None
    rr_text = (f" For example, {escape(rr['benchmark'])} was scored on {rr['n_models']} AI models whose capabilities vary only {_pct(rr['capability_spread'])} as much as those of all AI models in the data (comparing standard deviations). "
               f"Its reliability is {rr['reliability_mean']:.2f} among those models and would be {rr['all_models_reliability_mean']:.2f} across every AI model in the data."
               if rr and rr.get("capability_spread") is not None else "")
    how_block = f"""      <section class="rating-section" aria-labelledby="how-heading">
        <h2 id="how-heading">How the ratings are calculated</h2>
        <p>Two kinds of model appear on this page. <strong>AI models</strong> are the systems being benchmarked. The <strong>rating model</strong> is the statistical model we fit to their scores to produce the ratings.</p>
        <h3>The data</h3>
        <p>Epoch AI collects published scores for AI models on many benchmarks.{cite(3, first=False)} We use the table Epoch builds its Capabilities Index from:{cite(4)} {data["n_cells"]:,} scores for {n_models} AI models on {n_bench} benchmarks. Most AI models have scores on only a few benchmarks, so most of the table is empty (the field moves fast, and new benchmarks and models arrive constantly, so this is expected). The rating model is built to work with a sparse table.</p>
        <h3>The rating model</h3>
        <p>The rating model describes every score with a small number of quantities:</p>
        <ol class="steps">
          <li>Each AI model has one <strong>capability</strong> score. It is estimated from all of that AI model&rsquo;s results at once, so a model that does well across many benchmarks gets a high capability.</li>
          <li>Each benchmark has a <strong>difficulty</strong>, the capability at which an AI model is expected to score 50%, and a <strong>discrimination</strong>, how steeply expected scores rise with capability around that point. Together they define an S-shaped curve from capability to expected score: the logistic curve of item response theory, which Epoch&rsquo;s Capabilities Index also uses.{cite(5)}</li>
          <li>Each benchmark has a <strong>noise level</strong>: how widely real scores scatter around the curve. Scores are modelled with a Beta distribution centred on the curve, which keeps every score between 0% and 100%.</li>
        </ol>
        <p>All of these are estimated together, from all {data["n_cells"]:,} scores, with Bayesian sampling (the No-U-Turn Sampler, four chains of 1,000 draws). Each draw is one complete set of capabilities, difficulties, discriminations and noise levels that is consistent with the data. Uncertainty comes from how much the draws disagree.</p>
        <h3>Reliability</h3>
        <p>For one benchmark and the AI models scored on it, reliability compares two quantities. The <strong>capability variance</strong> is how much the expected scores (the points on the curve) vary across those AI models. The <strong>noise variance</strong> is the average variance of a score around its expected value. Reliability is the capability variance divided by the sum of the two:</p>
        <p class="formula">reliability = capability variance / (capability variance + noise variance)</p>
        <p>We compute it for 1,000 of the 4,000 draws, taken at random. The reported value is the average over draws, and the <strong>90% credible interval</strong> runs from the 5th to the 95th percentile of the draws. It is the Bayesian counterpart of a confidence interval: given the data and the rating model, the true reliability lies inside it with 90% probability. Benchmarks scored on few AI models get wide intervals.</p>
        <h3>Grades</h3>
        <p>A benchmark gets the highest grade whose threshold its reliability exceeds with at least {GRADE_PROB:.0%} probability, counted over the draws:</p>
        <div class="table-wrap" tabindex="0" role="region" aria-label="Grade rules"><table class="data-table grade-table">
          <thead><tr><th scope="col">Grade</th><th scope="col">Rule</th><th scope="col" class="num">Benchmarks</th></tr></thead>
          <tbody>
            <tr><th scope="row">A</th><td>reliability above {thr_a:.2f} with {GRADE_PROB:.0%} probability</td><td class="num mono">{len(groups["A"])}</td></tr>
            <tr><th scope="row">B</th><td>above {thr_b:.2f}, but not shown to be above {thr_a:.2f}</td><td class="num mono">{len(groups["B"])}</td></tr>
            <tr><th scope="row">C</th><td>above {thr_c:.2f}, but not shown to be above {thr_b:.2f}</td><td class="num mono">{len(groups["C"])}</td></tr>
            <tr><th scope="row">D</th><td>not shown to be above {thr_c:.2f}</td><td class="num mono">{len(groups["D"])}</td></tr>
            <tr><th scope="row">not graded</th><td>scored on fewer than {MIN_MODELS_FOR_GRADE} AI models</td><td class="num mono">{len(groups["ungraded"])}</td></tr>
          </tbody>
        </table></div>
        <p>A grade D means the data cannot show the benchmark is reliable, often because it was scored on few AI models. It does not mean the benchmark has been shown to be unreliable.</p>
        <h3>If every AI model had been scored</h3>
        <p>Reliability depends on which AI models were scored. A benchmark run only on recent frontier models sees AI models that are close to one another in capability. Their expected scores barely differ, so even modest noise is a large share of the variation, and reliability reads low. This is the same effect that makes an exam look unreliable when everyone taking it is equally strong.{rr_text} We report both. The first says how far to trust the score differences that exist today; the second says how good the benchmark would be at telling apart AI models across the whole range.</p>
      </section>
"""

    ratings_block = f"""      <section class="rating-section" aria-labelledby="ratings-heading">
        <h2 id="ratings-heading">The ratings</h2>
        <p>Each row is one benchmark, grouped by grade. The dot is its reliability among the AI models scored on it. The bar is the 90% credible interval. The ring shows where the reliability would move if every AI model in the data had been scored. Inside a grade, benchmarks are listed alphabetically, not ranked. Hover a row to see its numbers.</p>
        <div class="rating-figure">
{_legend()}
{forest(groups)}
{forest(groups, compact=True)}
        </div>
        <p class="note"><strong>Grades are groups, not a ranking.</strong> When a ring sits far to the right of its dot, the benchmark was scored on a narrow band of AI models and its low reading comes mostly from that. The {n_bench - graded} benchmarks with too few AI models to grade are shown in grey with their interval only.</p>
        <details class="full-table">
          <summary>All numbers, per benchmark</summary>
          <dl class="terms">
            <dt>AI models</dt><dd>How many AI models have a score on the benchmark.</dd>
            <dt>Reliability (90% CI)</dt><dd>The average over draws, with the 90% credible interval.</dd>
            <dt>If every AI model scored</dt><dd>Reliability computed over every AI model in the data instead of only those scored.</dd>
            <dt>Noise (points)</dt><dd>The typical distance between a score and the curve, in percentage points, for an AI model expected to score 50%.</dd>
            <dt>Held-out skill</dt><dd>How well the rating model predicted hidden scores, explained in the next section.</dd>
            <dt>Headroom</dt><dd>Saturated when at least half of the newest quarter of scored AI models score 90% or more; unknown when no recent AI model was scored.</dd>
          </dl>
{table(data, groups)}
        </details>
      </section>
"""

    cal_block = ""
    if cal_svg:
        cal_block = f"""      <section class="rating-section" aria-labelledby="cal-heading">
        <h2 id="cal-heading">Testing the rating model on hidden scores</h2>
        <p>A reliability is only as good as the rating model it comes from, so we tested the rating model on scores it had not seen. We split the {data["heldout"]["kfold"]["n_cells"]:,} scores at random into five parts. In each of five rounds we hid one part, fitted the rating model on the other four, and asked it to predict each hidden score. Every score was hidden exactly once.</p>
        <p>For each benchmark, <strong>held-out skill</strong> compares those predictions with the simplest possible prediction: that benchmark&rsquo;s average score. It is one minus the ratio of their errors. A skill of 0 means the rating model did no better than the average. A skill of 0.5 means its errors were half as large. Errors are measured with the continuous ranked probability score, which rewards predictions that are both close to the truth and appropriately certain.{cite(6)}</p>
        <div class="rating-figure">
          <p class="figure-title">Each dot is one benchmark: its reliability across, its held-out skill up</p>
          <ul class="legend">
            <li><svg viewBox="0 0 34 14" aria-hidden="true"><circle class="rf-bub" cx="9" cy="7" r="3" /><circle class="rf-bub" cx="24" cy="7" r="6" /></svg>one benchmark; larger dots have more AI models</li>
            <li><svg viewBox="0 0 34 14" aria-hidden="true"><line class="rf-whisk" x1="17" y1="0" x2="17" y2="14" /></svg>90% interval of its skill</li>
            <li><svg viewBox="0 0 34 14" aria-hidden="true"><line class="rf-trend" x1="2" y1="11" x2="32" y2="3" /></svg>straight-line trend</li>
          </ul>
{cal_svg}
{cal_compact}
        </div>
        <p><strong>How to read it.</strong> If the rating model is right about a benchmark, a high reliability should go with accurate predictions of its hidden scores, so the dot sits high and to the right. A benchmark the rating model calls noisy should be one whose hidden scores it predicts poorly, low and to the left. A dot far from the dashed trend would be a benchmark the rating model gets wrong. Across {n_cal} benchmarks, the order by reliability and the order by held-out skill have a rank correlation of {rho:.2f}, where 1 would mean the same order.</p>
        <p class="note">This is a consistency check, not proof. Both numbers come from the same rating model, so an error it makes the same way on every benchmark would not show up here. The comparison in the next section is a stronger test.{f" Separately, a rating model fitted only to AI models released before {escape(c3['cutoff_date'])} orders the benchmarks much like the full fit (rank correlation {c3['rho']:.2f})." if c3 else ""}</p>
      </section>
"""

    fit_block = ""
    kf_all = data.get("heldout", {}).get("kfold", {})
    if c_eci and kf_all:
        mt = (f'<div class="rating-figure">\n<p class="figure-title">Predicting {kf_all["n_cells"]:,} hidden scores: four methods</p>\n'
              f'{methods_chart(kf_all, METHOD_TEXT)}\n{methods_chart(kf_all, METHOD_TEXT, compact=True)}\n</div>')
        kf, fr = data["heldout"]["kfold"], data["heldout"]["frontier"]
        fit_block = f"""      <section class="rating-section" aria-labelledby="fit-heading">
        <h2 id="fit-heading">Comparison with Epoch&rsquo;s Capabilities Index</h2>
        <p>Epoch&rsquo;s Capabilities Index{cite(4, first=False)} places AI models on one capability scale using the same S-shaped curve.{cite(5, first=False)} Our version of its form fits the curve by least squares and keeps a single best estimate of every quantity; Epoch&rsquo;s published index adds bootstrap intervals, which we did not reproduce. The rating model differs in two ways: it estimates a separate noise level for every benchmark, and it carries the uncertainty in every quantity through to its predictions. We refitted Epoch&rsquo;s form on the same data and gave all four methods the same hidden-score test:</p>
        <ul class="steps">
          <li><strong>{METHOD_TEXT["hier_beta"]}</strong>: the Bayesian rating model described above.</li>
          <li><strong>{METHOD_TEXT["eci_ls"]}</strong>: the same curve fitted by least squares, with a fixed error band per benchmark.</li>
          <li><strong>{METHOD_TEXT["mean_rank"]}</strong>: each AI model&rsquo;s average standardised score across its benchmarks, mapped to each benchmark with a straight line.</li>
          <li><strong>{METHOD_TEXT["benchmark_mean"]}</strong>: always predicting the benchmark&rsquo;s average score.</li>
        </ul>
{mt}
        <p>The left panel is the typical prediction error in percentage points; shorter is better. The right panel is how often each method&rsquo;s 90% prediction interval contained the true score. An honest 90% interval should contain it about 90% of the time.</p>
        <p>The two curve-based methods are almost equally accurate: the rating model&rsquo;s error is {100 * abs(c_eci["mean_diff"]):.1f} points lower (95% interval {100 * abs(c_eci["hi"]):.1f} to {100 * abs(c_eci["lo"]):.1f}). The difference is in the intervals. The rating model&rsquo;s 90% intervals contained the true score {_pct(kf["coverage90"]["hier_beta"])} of the time. The least-squares fit&rsquo;s contained it {_pct(kf["coverage90"]["eci_ls"])} of the time, so it is more confident than the data justify. The gap widens on the newest AI models: when half the scores of the most recent quarter of AI models are hidden, the errors are {100 * fr['mean_crps']['hier_beta']:.1f} and {100 * fr['mean_crps']['eci_ls']:.1f} points and the coverage is {_pct(fr['coverage90']['hier_beta'])} and {_pct(fr['coverage90']['eci_ls'])}. Reliability is a statement about noise, so it needs a model whose uncertainty is right. That is what the rating model adds; it is not a better capability index.</p>
      </section>
"""

    return f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{TITLE} &middot; plicara labs</title>
{robots}    <meta name="description" content="{n_bench} AI benchmarks rated the way pollsters are rated: how much of the difference between models' scores is real capability rather than noise, with uncertainty." />
    <link rel="canonical" href="https://plicara.ai/benchmarks/pollbench/" />
    <meta name="theme-color" media="(prefers-color-scheme: dark)" content="#082C35" />
    <meta name="theme-color" media="(prefers-color-scheme: light)" content="#ffffff" />
    <link rel="icon" href="/assets/favicon.svg" type="image/svg+xml" />
    <link rel="apple-touch-icon" href="/assets/brand/apple-touch-icon-180.png" />
    <link rel="preload" as="font" type="font/woff2" href="/assets/fonts/archivo.woff2" crossorigin />
    <link rel="preload" as="font" type="font/woff2" href="/assets/fonts/jetbrains-mono-400.woff2" crossorigin />
    <link rel="preload" as="font" type="font/woff2" href="/assets/fonts/jetbrains-mono-700.woff2" crossorigin />
    <link rel="stylesheet" href="/assets/tokens.css" />
    <link rel="stylesheet" href="/assets/style.css?v=20260929" />
    <style>{STYLE}    </style>
  </head>
  <body data-scheme="technical">
    <a class="skip-link" href="#main">Skip to content</a>
    <header class="site-header">
      <div class="wrap">
        <a class="brand" href="/"><span class="brand-mark" aria-hidden="true"></span>plicara labs</a>
        <nav class="site-nav" aria-label="Primary">
          <a href="/research/">Research</a>
          <a href="/tools/">Tools</a>
          <a href="/benchmarks/">Benchmarks</a>
          <a href="/about/">About</a>
          <a href="/#contact">Follow</a>
        </nav>
      </div>
    </header>
    <main class="wrap results" id="main">
      <h1>{TITLE}</h1>
      <p class="lede">When two AI models get different scores on a benchmark, how much of the difference is real capability and how much is noise? This page answers that question for {n_bench} AI benchmarks, the way pollster ratings judge how far a poll&rsquo;s numbers can be trusted.{cite(1)} Each benchmark gets a reliability score, a grade, and an interval that says how certain the rating is.</p>
      <div class="hero-stats"><span><b>{n_bench}</b>benchmarks</span><span><b>{n_models}</b>AI models</span><span><b>{data["n_cells"]:,}</b>reported scores</span></div>
      {grade_strip(groups, ORDER)}
{banner}{glance}{measure_block}{how_block}{ratings_block}{cal_block}{fit_block}{factor_block}      <section class="rating-section" aria-labelledby="data-heading">
        <h2 id="data-heading">Data and reproducibility</h2>
        <p>Data: Epoch AI, &lsquo;Capabilities &amp; benchmarking&rsquo;, CC BY 4.0.{cite(3, first=False)} We use the table Epoch builds its Capabilities Index from,{cite(4, first=False)} so every benchmark here is one Epoch already considers suitable for that index: {n_models} AI models, {n_bench} benchmarks and {data["n_cells"]:,} scores. The sampler converged: the worst R-hat across every benchmark&rsquo;s parameters is {max((r["rhat"] for r in data["benchmarks"] if r.get("rhat") is not None), default=float("nan")):.3f}, below the 1.01 limit for grading.{cite(7)} Data fingerprint: <code>{escape(data["data_digest"][:16])}</code>.</p>
        <p>The method and the tests it had to pass were written down before the first fit, and every later change is recorded as a dated amendment. The repository has the code, the exact Epoch table we used, every number on this page, the full method, a simulation study that checks the method on data where the true reliabilities are known, and the complete list of limitations.</p>
      </section>
      <section class="rating-section" aria-labelledby="refs-heading">
        <h2 id="refs-heading">References</h2>
{references()}
      </section>
      <div class="cta-row">
        <a class="btn btn-primary" href="https://github.com/plicara/benchmarks/tree/main/pollbench">Results and method</a>
        <a class="btn btn-ghost" href="/benchmarks/">&larr; All benchmarks</a>
      </div>
    </main>
    <footer class="site-footer">
      <div class="wrap">
        <span>&copy; plicara labs</span>
        <div class="footer-links">
          <a href="mailto:info@plicara.ai">info@plicara.ai</a>
          <a href="https://github.com/plicara">GitHub</a>
          <a href="/">Home</a>
        </div>
      </div>
    </footer>
  </body>
</html>
"""


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ratings")
    ap.add_argument("--out", required=True)
    ap.add_argument("--curves", help="curves JSON from pollbench.curves, for the real-score figure")
    ap.add_argument("--placeholder", help="mark the page as a preview on non-measured numbers; the text is shown")
    args = ap.parse_args(argv)
    data = json.loads(Path(args.ratings).read_text())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    curves = json.loads(Path(args.curves).read_text()) if args.curves else None
    if curves and curves["data_digest"] != data["data_digest"]:
        raise SystemExit("curves and ratings come from different data")
    Path(args.out).write_text(render(data, placeholder=args.placeholder, curves=curves), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
