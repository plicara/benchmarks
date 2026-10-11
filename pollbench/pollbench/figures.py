"""SVG figures for the results page that need more than the per-benchmark summaries.

Everything here returns markup only; ``page.py`` owns the CSS classes it uses
(``rf-*``) so the figures follow the site's theme tokens in light and dark.
"""

from __future__ import annotations

from html import escape

import numpy as np
from scipy.special import expit
from scipy.stats import beta as beta_dist


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def grade_strip(groups: dict[str, list[dict]], order: list[str]) -> str:
    """One bar split by how many benchmarks landed in each grade, darkest for A. HTML, so it reflows on a phone."""
    counts = [(g, len(groups[g])) for g in order if groups[g]]
    label = ", ".join(f'{"not graded" if g == "ungraded" else "grade " + g}: {n}' for g, n in counts)
    segs = "".join(
        f'<div class="seg seg-{g}" style="flex-grow:{n}"><span class="seg-n">{n}</span>'
        f'<span class="seg-l">{"not graded" if g == "ungraded" else "grade " + g}</span></div>'
        for g, n in counts)
    return f'<div class="strip" role="img" aria-label="{label}">{segs}</div>'


def curve_panel(name: str, row: dict, curve: dict, capability: dict, x_range: tuple[float, float], idx: int) -> str:
    """Real scores against fitted capability for one benchmark, with the S-curve and its 90% noise band."""
    W, H, left, right, top, bottom = 400, 300, 44, 388, 74, 262
    lo, hi = x_range

    def fx(t):
        return left + (right - left) * (t - lo) / (hi - lo)

    def fy(s):
        return bottom - (bottom - top) * s

    rel = row["reliability_mean"]
    d, a, k = curve["difficulty"], curve["slope"], curve["precision"]
    grid = np.linspace(lo, hi, 90)
    p = np.clip(expit(a * (grid - d)), 1e-4, 1 - 1e-4)
    q05, q95 = beta_dist.ppf(0.05, p * k, (1 - p) * k), beta_dist.ppf(0.95, p * k, (1 - p) * k)
    band = " ".join(f"{fx(t):.1f},{fy(s):.1f}" for t, s in zip(grid, q95))
    band += " " + " ".join(f"{fx(t):.1f},{fy(s):.1f}" for t, s in zip(grid[::-1], q05[::-1]))
    line = " ".join(f"{fx(t):.1f},{fy(s):.1f}" for t, s in zip(grid, p))
    pid = f"curve-{idx}"
    xm = left + (right - left) * rel
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-labelledby="{pid}-t">',
           f'<title id="{pid}-t">{escape(name)}: {len(curve["scores"])} model scores against fitted capability; reliability {rel:.2f}</title>',
           f'<text class="rf-panel" x="{left}" y="16">{escape(name)}</text>',
           f'<text class="rf-hint" x="{right}" y="16" text-anchor="end">grade {row["grade"]} &middot; {row["n_models"]} AI models</text>',
           f'<rect class="rf-signal" x="{left}" y="28" width="{xm - left - 1:.1f}" height="14" rx="3" />',
           f'<rect class="rf-noise" x="{xm + 1:.1f}" y="28" width="{right - xm - 1:.1f}" height="14" rx="3" />',
           ]
    if right - xm > 72:
        out.append(f'<text class="rf-inbar" x="{left + 6}" y="39">capability {_pct(rel)}</text>')
        out.append(f'<text class="rf-inbar" x="{right - 6}" y="39" text-anchor="end">noise {_pct(1 - rel)}</text>')
    else:  # the noise segment is too thin for its own label
        out.append(f'<text class="rf-inbar" x="{left + 6}" y="39">capability {_pct(rel)} &middot; noise {_pct(1 - rel)}</text>')
    for s in (0, .5, 1):
        out.append(f'<line class="rf-grid" x1="{left}" y1="{fy(s):.1f}" x2="{right}" y2="{fy(s):.1f}" />')
        out.append(f'<text class="rf-tick" x="{left - 6}" y="{fy(s) + 4:.1f}" text-anchor="end">{_pct(s)}</text>')
    out.append(f'<polygon class="rf-noiseband" points="{band}" />')
    out.append(f'<polyline class="rf-curve" points="{line}" />')
    for model, score in curve["scores"]:
        t = capability.get(model)
        if t is None:
            continue
        out.append(f'<circle class="rf-pt" cx="{fx(t[0]):.1f}" cy="{fy(score):.1f}" r="3.2"><title>{escape(model)}: {_pct(score)}</title></circle>')
    out.append(f'<line class="rf-axis" x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" />')
    out.append(f'<text class="rf-hint" x="{left}" y="{bottom + 18}">less capable</text>')
    out.append(f'<text class="rf-hint" x="{right}" y="{bottom + 18}" text-anchor="end">more capable &rarr;</text>')
    out.append("</svg>")
    return "\n".join(out)


def curve_panels(examples: list[dict], curves: dict) -> str:
    caps = curves["capability"]
    names = [r["benchmark"] for r in examples if r["benchmark"] in curves["benchmarks"]]
    if not names:
        return ""
    xs = [caps[m][0] for n in names for m, _ in curves["benchmarks"][n]["scores"] if m in caps]
    pad = 0.05 * (max(xs) - min(xs))
    x_range = (min(xs) - pad, max(xs) + pad)
    rows = {r["benchmark"]: r for r in examples}
    return "\n".join(f'<div class="panel">{curve_panel(n, rows[n], curves["benchmarks"][n], caps, x_range, i)}</div>'
                     for i, n in enumerate(names))


def methods_chart(kf: dict, labels: dict[str, str], *, compact: bool = False) -> str:
    """Two aligned panels: typical error (bars) and how often the 90% range held the truth (dots against the target)."""
    methods = [m for m in labels if m in kf["mean_crps"]]
    if compact:
        W, name_w, row_h, gap = 360, 0, 58, 0
    else:
        W, name_w, row_h, gap = 840, 290, 40, 40
    top = 34
    H = top + row_h * len(methods) + 30
    p1 = (name_w, name_w + (W - name_w - gap) * 0.5) if not compact else (0, 172)
    p2 = (p1[1] + gap, W - 6) if not compact else (188, W - 6)
    emax = max(kf["mean_crps"][m] for m in methods) * 100
    c_lo, c_hi = 0.75, 1.0

    def fe(v):
        return p1[0] + (p1[1] - p1[0]) * v / emax

    def fc(v):
        return p2[0] + (p2[1] - p2[0]) * (min(max(v, c_lo), c_hi) - c_lo) / (c_hi - c_lo)

    sfx = "-compact" if compact else ""
    out = [f'<svg class="{"compact" if compact else "wide"}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="methods-title{sfx}">',
           f'<title id="methods-title{sfx}">Typical error on hidden scores, and how often each method&rsquo;s 90% interval held the true score: '
           + "; ".join(f'{labels[m]} {100 * kf["mean_crps"][m]:.1f} points, {_pct(kf["coverage90"][m])}' for m in methods) + "</title>",
           f'<text class="rf-title" x="{p1[0]}" y="14">TYPICAL ERROR, POINTS</text>',
           f'<text class="rf-hint" x="{p1[0]}" y="27">shorter is better</text>',
           f'<text class="rf-title" x="{p2[0]}" y="14">{"90% INTERVAL HOLDS" if compact else "90% INTERVAL HOLDS THE TRUTH"}</text>',
           f'<text class="rf-hint" x="{p2[0]}" y="27">closer to 90% is better</text>']
    y_end = top + row_h * len(methods)
    out.append(f'<rect class="rf-target" x="{fc(0.85):.1f}" y="{top}" width="{fc(0.95) - fc(0.85):.1f}" height="{y_end - top}" />')
    out.append(f'<line class="rf-cut" x1="{fc(0.9):.1f}" y1="{top}" x2="{fc(0.9):.1f}" y2="{y_end}" />')
    for t in (0.75, 0.9, 1.0):
        out.append(f'<text class="rf-tick" x="{fc(t):.1f}" y="{y_end + 16}" text-anchor="{"end" if t == 1.0 else "middle"}">{_pct(t)}</text>')
    for i, m in enumerate(methods):
        y = top + row_h * i
        mine = m == "hier_beta"
        cls = " mine" if mine else ""
        err, cov = 100 * kf["mean_crps"][m], kf["coverage90"][m]
        if compact:
            out.append(f'<text class="rf-label{cls}" x="0" y="{y + 16}">{labels[m]}</text>')
            yb = y + 34
        else:
            out.append(f'<text class="rf-label{cls}" x="{name_w - 14}" y="{y + 24}" text-anchor="end">{labels[m]}</text>')
            yb = y + 20
        out.append(f'<rect class="rf-mbar{cls}" x="{p1[0]}" y="{yb - 8}" width="{fe(err) - p1[0]:.1f}" height="16" rx="3" />')
        out.append(f'<text class="rf-label" x="{fe(err) + 6:.1f}" y="{yb + 4}">{err:.1f}</text>')
        out.append(f'<line class="rf-grid" x1="{p2[0]}" y1="{yb}" x2="{p2[1]}" y2="{yb}" />')
        out.append(f'<circle class="rf-mdot{cls}" cx="{fc(cov):.1f}" cy="{yb}" r="6" />')
        anchor, dx = ("end", -11) if fc(cov) > p2[1] - 40 else ("start", 11)
        out.append(f'<text class="rf-label" x="{fc(cov) + dx:.1f}" y="{yb + 4}" text-anchor="{anchor}">{_pct(cov)}</text>')
    out.append("</svg>")
    return "\n".join(out)
