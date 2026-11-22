"""Render descriptive M8 figures from the published, qualified analysis CSVs."""

import csv
import html
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "research/analysis/m8-phase2a-five-session-final"
OUT = Path(__file__).resolve().parent
NAVY, TEAL, CORAL, GRID = "#172a46", "#087e83", "#c7504e", "#dce4ec"
WORKLOADS = [(7, "7 × 19 × 11"), (256, "256 × 64 × 384")]


def rows(name):
    with (DATA / name).open(newline="") as f:
        return list(csv.DictReader(f))


def element(tag, **attrs):
    return "<" + tag + " " + " ".join(f'{k.replace("_", "-")}="{html.escape(str(v))}"' for k, v in attrs.items()) + "/>"


def text(x, y, value, size=15, color=NAVY, anchor="start", weight="normal"):
    return f'<text x="{x}" y="{y}" fill="{color}" font-family="Arial, sans-serif" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}">{html.escape(str(value))}</text>'


def line(x1, y1, x2, y2, color=GRID, width=1):
    return element("line", x1=x1, y1=y1, x2=x2, y2=y2, stroke=color, stroke_width=width)


def dot(x, y, color, radius=6, fill=None):
    return element("circle", cx=x, cy=y, r=radius, fill=fill or color, stroke=color, stroke_width=2)


def save(name, title, subtitle, body, height=445):
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="960" height="{height}" viewBox="0 0 960 {height}" role="img" aria-label="{html.escape(title)}">',
           element("rect", x=0, y=0, width=960, height=height, fill="white"),
           text(42, 43, title, 24, weight="bold"), text(42, 69, subtitle, 14, "#52657b"), *body,
           text(42, height - 18, "M8 Phase 2A · Apple M5 / Metal · five physical sessions · descriptive", 12, "#52657b"), "</svg>"]
    (OUT / name).write_text("\n".join(svg) + "\n")


def panels(body, y_top=119, y_bottom=355, low=0, high=1, ticks=None, suffix=""):
    for idx, (_, label) in enumerate(WORKLOADS):
        x0 = 58 + idx * 465
        body.append(text(x0, 104, label, 17, weight="bold"))
        for value in ticks or [low, (low + high) / 2, high]:
            y = y_bottom - (value - low) / (high - low) * (y_bottom - y_top)
            body.extend([line(x0 + 38, y, x0 + 416, y), text(x0 + 29, y + 5, f"{value:g}{suffix}", 12, "#52657b", "end")])
        for session in range(5):
            x = x0 + 75 + session * 75
            body.extend([line(x, y_bottom, x, y_bottom + 5, NAVY), text(x, y_bottom + 24, f"{session + 1:03}", 12, "#52657b", "middle")])
    return lambda idx, session: 58 + idx * 465 + 75 + session * 75, lambda value: y_bottom - (value - low) / (high - low) * (y_bottom - y_top)


agreement = rows("best-set-agreement.csv")
assert len(agreement) == 10
body = []
xpos, ypos = panels(body, low=0, high=1, ticks=[0, 1])
for idx, (m, _) in enumerate(WORKLOADS):
    subset = sorted((r for r in agreement if int(r["m"]) == m), key=lambda r: int(r["session_index"]))
    assert len(subset) == 5
    for r in subset:
        ok = r["exact_agreement"] == "true"
        body.append(dot(xpos(idx, int(r["session_index"])), ypos(1 if ok else 0), TEAL if ok else CORAL, 9))
    body.append(text(58 + idx * 465 + 230, 420, f'{sum(r["exact_agreement"] == "true" for r in subset)}/5 sessions agree', 15, NAVY, "middle", "bold"))
save("m8-best-agreement.svg", "Empirical-best agreement", "Paired panels A and B · each dot is one physical session", body, 465)

regret = rows("cross-panel-regret.csv")
assert len(regret) == 10
body = []
xpos, ypos = panels(body, low=0, high=16, ticks=[0, 4, 8, 12, 16], suffix="%")
for idx, (m, _) in enumerate(WORKLOADS):
    subset = sorted((r for r in regret if int(r["m"]) == m), key=lambda r: int(r["session_index"]))
    for field, color, dx in [("a_to_b_designated", TEAL, -5), ("b_to_a_designated", CORAL, 5)]:
        points = []
        for r in subset:
            value = 100 * float(r[field])
            assert 0 <= value <= 16
            x, y = xpos(idx, int(r["session_index"])) + dx, ypos(value)
            points.append((x, y))
            body.append(dot(x, y, color, 5))
        for a, b in zip(points, points[1:]):
            body.append(line(*a, *b, color, 2))
body.extend([dot(350, 410, TEAL, 5), text(365, 415, "A → B", 13), dot(460, 410, CORAL, 5), text(475, 415, "B → A", 13)])
save("m8-directional-regret.svg", "Observed directional regret", "Receiving panel's measured minimum is the denominator · no panel is an oracle", body)

rank = rows("rank-agreement.csv")
assert len(rank) == 10
body = []
xpos, ypos = panels(body, low=0, high=1, ticks=[0, 0.25, 0.5, 0.75, 1])
for idx, (m, _) in enumerate(WORKLOADS):
    subset = sorted((r for r in rank if int(r["m"]) == m), key=lambda r: int(r["session_index"]))
    for field, color, dx in [("spearman", TEAL, -5), ("kendall_tau_b", CORAL, 5)]:
        for r in subset:
            value = float(r[field])
            assert 0 <= value <= 1
            body.append(dot(xpos(idx, int(r["session_index"])) + dx, ypos(value), color, 5))
body.extend([dot(325, 410, TEAL, 5), text(340, 415, "Spearman ρ", 13), dot(485, 410, CORAL, 5), text(500, 415, "Kendall τ-b", 13)])
save("m8-rank-repeatability.svg", "Candidate ranking repeatability", "Full ten-candidate rankings · paired panels within each physical session", body)

timing = rows("panel-candidates.csv")
assert len(timing) == 200
selected = {7: "tiled-bm8-bn8-bk8-wg8x8x1-shader1", 256: "tiled-bm16-bn16-bk32-wg16x16x1-shader1"}
body = []
for idx, (m, label) in enumerate(WORKLOADS):
    subset = [r for r in timing if int(r["m"]) == m and r["candidate_id"] == selected[m]]
    assert len(subset) == 10
    values = [float(r["latency_ns"]) / 1000 for r in subset]
    lo, hi = (18, 28) if m == 7 else (436, 446)
    x0 = 58 + idx * 465
    body.append(text(x0, 104, label, 17, weight="bold"))
    body.append(text(x0, 126, "8×8×8" if m == 7 else "16×16×32", 13, "#52657b"))
    for value in [lo, (lo + hi) / 2, hi]:
        y = 355 - (value - lo) / (hi - lo) * 215
        body.extend([line(x0 + 38, y, x0 + 416, y), text(x0 + 29, y + 5, f"{value:g}", 12, "#52657b", "end")])
    for session in range(5):
        pair = {r["panel_id"]: float(r["latency_ns"]) / 1000 for r in subset if int(r["session_index"]) == session}
        assert set(pair) == {"A", "B"}
        x = x0 + 75 + session * 75
        ya, yb = [355 - (pair[p] - lo) / (hi - lo) * 215 for p in ("A", "B")]
        body.extend([line(x - 5, ya, x + 5, yb, "#9baabe", 2), dot(x - 5, ya, TEAL, 5), dot(x + 5, yb, CORAL, 5), text(x, 379, f"{session + 1:03}", 12, "#52657b", "middle")])
body.extend([text(42, 417, "µs · median of 30 GPU timestamp samples per candidate-panel", 13, "#52657b"), dot(675, 412, TEAL, 5), text(690, 417, "A", 13), dot(745, 412, CORAL, 5), text(760, 417, "B", 13)])
save("m8-timing-medians.svg", "Paired timing medians", "One fixed candidate per workload · paired sessions, not 10 independent observations", body, 465)
