"""Closing figures for round 9: what was tested, what was not, and why it failed.

Three figures, all drawn from assets/round9_final.json -- frozen at the end of the round so
they cannot silently change if the archive is re-collected:

  fig-r9-scoreboard      every arm, both environments, treatment beside its own control
  fig-r9-action-channel  d_action/|z| against SR, with the disqualification region marked
  fig-r9-p6-mechanism    vel_rel against d_action/|z|: P6 fails BECAUSE it succeeds

DESIGN NOTE. Success rate is a magnitude on a common zero, so bars; the other two are
relationships between two measured quantities, so scatter. Each arm is drawn beside the
control it is DEFINED against, never against a registered mean, because the whole round turns
on treatment-minus-its-own-control. Colour carries one thing only -- whether the arm's action
channel lets it be interpreted at all -- and the legend says so in words, so the figures do
not rely on colour alone.
"""
import json
import os

from analysis.arch_figs import (ACCENT, DIM, INK, MUTED, WHITE, emit, text_width, txt)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "assets", "round9_final.json")

TESTED = ACCENT["blue"]        # the arm's action channel survived -> its number means something
DEAD = ACCENT["crit"]          # channel destroyed -> the number is not evidence
BASEC = ACCENT["green"]        # baseline / control reference
BAND = 0.40                    # d_action/|z| below this = disqualified (see diary s11.4)


def _d():
    return json.load(open(DATA))


def paired(rt, rc):
    """Treatment minus control on SHARED seeds -- the quantity every contrast in this round
    is defined as. The difference of the two arms' MEANS is a different number (it disagreed
    by 0.013 on P5 wall, where the control has one seed fewer)."""
    if not rt or not rc:
        return None
    a, b = rt.get("sr", {}), rc.get("sr", {})
    sh = sorted(set(a) & set(b), key=int)
    if not sh:
        return None
    return sum(a[s] - b[s] for s in sh) / len(sh), len(sh)


def _mean(rec, field="sr"):
    v = rec.get(field, {}) if rec else {}
    return (sum(v.values()) / len(v)) if v else None


def axes(x, y, w, h, ymax, ylab, ticks=5):
    """A plain left axis + baseline. Ticks are drawn, not implied."""
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{WHITE}" '
         f'stroke="{DIM}" stroke-width="1"/>\n')
    for i in range(ticks + 1):
        v = ymax * i / ticks
        yy = y + h - (v / ymax) * h
        s += (f'<line x1="{x}" y1="{yy:.1f}" x2="{x+w}" y2="{yy:.1f}" '
              f'stroke="#E8ECEA" stroke-width="1"/>\n')
        s += txt(x - 8, yy + 4, f"{v:.1f}", 12, anchor="end", fill=MUTED)
    s += txt(x, y - 10, ylab, 12.5, anchor="start", fill=MUTED)
    return s


def bar(x, y, w, h, val, ymax, color, label, sub=""):
    bh = max(1.0, (val / ymax) * h)
    s = (f'<rect x="{x:.1f}" y="{y + h - bh:.1f}" width="{w:.1f}" height="{bh:.1f}" '
         f'rx="3" fill="{color}" opacity="0.88"/>\n')
    s += txt(x + w / 2, y + h - bh - 7, f"{val:.3f}", 12, fill=INK, weight="bold")
    s += txt(x + w / 2, y + h + 16, label, 12, fill=INK)
    if sub:
        s += txt(x + w / 2, y + h + 31, sub, 11, fill=MUTED)
    return s


def clamp(v, lo, hi):
    """Keep a mark FULLY inside its panel. A dot straddling the border is a collision to the
    audit and, more to the point, is genuinely ambiguous to read."""
    return min(max(v, lo), hi)


def dot(cx, cy, r, color, filled=True):
    return (f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" '
            f'fill="{color if filled else WHITE}" stroke="{color}" stroke-width="2.2"/>\n')


def legend(x, y, items):
    s = ""
    for i, (col, lab) in enumerate(items):
        yy = y + i * 20
        s += f'<rect x="{x}" y="{yy-9}" width="13" height="13" rx="2" fill="{col}"/>\n'
        s += txt(x + 20, yy + 2, lab, 12.5, anchor="start", fill=INK)
    return s


# ======================================================= 1. the scoreboard
PAIRS = [("P3", "PiWM-scan2", "PiWM-scan2-fixed", "scan"),
         ("P4", "PiWM-st2", "PiWM-st2-fixed", "scan+state"),
         ("P5", "PiWM-sinv2", "PiWM-sinv2-shuf", "consistency"),
         ("P6", "PiWM-vel2", "PiWM-sum2", "velocity"),
         ("P7", "PiWM-nce2", "PiWM-nce2-shuf", "contrastive")]


def scoreboard():
    d = _d()
    W, H = 1180, 860
    s = txt(W / 2, 44, "Round 9: every arm beside the control it is defined against", 21,
            fill=INK, weight="bold")
    s += txt(W / 2, 68, "bar height = CEM planning success rate;  colour = whether the arm's "
             "action channel survived;  delta is PAIRED on shared seeds", 13, fill=MUTED)
    s += txt(W / 2, 88, "pushT's registered baseline over all 13 archived seeds is 0.357; "
             "the line below is the mean over the three seeds this round paired against.",
             11.5, fill=MUTED)

    for row, (env, tag) in enumerate((("wall", "_wall"), ("pusht", ""))):
        y0 = 148 + row * 340
        base = _mean(d.get(f"{env}|LpWM-ltv"))
        ymax = 0.80 if env == "wall" else 0.50
        s += txt(1096, y0 - 12, f"{env.upper()}", 15, anchor="end", fill=INK, weight="bold")
        s += axes(96, y0, 1000, 218, ymax, "success rate")
        # baseline reference line, labelled where it cannot collide with the bars
        by = y0 + 218 - (base / ymax) * 218
        s += (f'<line x1="96" y1="{by:.1f}" x2="1096" y2="{by:.1f}" stroke="{BASEC}" '
              f'stroke-width="2" stroke-dasharray="7,4"/>\n')
        nb = len(d.get(f"{env}|LpWM-ltv", {}).get("sr", {}))
        s += txt(112, by - 10, f"baseline {base:.3f}  (seeds 3-5, n={nb})", 12,
                 anchor="start", fill=BASEC, weight="bold")
        for i, (p, t, c, what) in enumerate(PAIRS):
            gx = 130 + i * 196
            rt, rc = d.get(f"{env}|{t}"), d.get(f"{env}|{c}")
            vt, vc = _mean(rt), _mean(rc)
            dz = _mean(rt, "dz")
            live = dz is not None and dz > BAND
            col = TESTED if live else DEAD
            if vt is not None:
                s += bar(gx, y0, 58, 218, vt, ymax, col, p, what)
            if vc is not None:
                s += bar(gx + 66, y0, 58, 218, vc, ymax, DIM, "ctrl")
            pr = paired(rt, rc)
            if pr is not None:
                s += txt(gx + 62, y0 + 262, f"{pr[0]:+.3f}", 13,
                         fill=(INK if live else MUTED), weight="bold")
                s += txt(gx + 62, y0 + 277, f"n={pr[1]}", 10.5, fill=MUTED)
            elif vt is None:
                s += txt(gx + 29, y0 + 130, "no data", 11.5, fill=MUTED)
        s += txt(96, y0 + 262, "vs ctrl:", 12, anchor="start", fill=MUTED)

    s += legend(96, 800, [(TESTED, "action channel intact -> the number is evidence"),
                          (DEAD, "action channel destroyed -> NOT evidence, arm untestable"),
                          (DIM, "the arm's own matched control")])
    return s, W, H


# ======================================================= 2. the action channel
def action_channel():
    d = _d()
    W, H = 1180, 800
    x0, y0, pw, ph = 150, 120, 880, 440
    s = txt(W / 2, 44, "Why most arms carry no information: the action channel", 21,
            fill=INK, weight="bold")
    s += txt(W / 2, 68, "d_action/|z| is how far the prediction moves when only the ACTION "
             "changes -- the quantity CEM consumes", 13, fill=MUTED)

    s += (f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="{WHITE}" '
          f'stroke="{DIM}" stroke-width="1"/>\n')
    # disqualification region, drawn rather than described
    bx = x0 + (BAND / 0.75) * pw
    s += (f'<rect x="{x0}" y="{y0}" width="{bx-x0:.1f}" height="{ph}" fill="{DEAD}" '
          f'opacity="0.07"/>\n')
    s += (f'<line x1="{bx:.1f}" y1="{y0}" x2="{bx:.1f}" y2="{y0+ph}" stroke="{DEAD}" '
          f'stroke-width="1.6" stroke-dasharray="6,4"/>\n')
    s += txt(x0 + 14, y0 + 24, "DISQUALIFIED", 13, anchor="start", fill=DEAD, weight="bold")
    s += txt(x0 + 14, y0 + 42, "prediction barely responds to the action;", 11.5,
             anchor="start", fill=MUTED)
    s += txt(x0 + 14, y0 + 58, "CEM has nothing to optimise", 11.5, anchor="start", fill=MUTED)

    for i in range(6):
        v = 0.75 * i / 5
        xx = x0 + (v / 0.75) * pw
        s += (f'<line x1="{xx:.1f}" y1="{y0}" x2="{xx:.1f}" y2="{y0+ph}" '
              f'stroke="#EEF1EF" stroke-width="1"/>\n')
        s += txt(xx, y0 + ph + 20, f"{v:.2f}", 12, fill=MUTED)
    for i in range(5):
        v = 0.8 * i / 4
        yy = y0 + ph - (v / 0.8) * ph
        s += (f'<line x1="{x0}" y1="{yy:.1f}" x2="{x0+pw}" y2="{yy:.1f}" '
              f'stroke="#EEF1EF" stroke-width="1"/>\n')
        s += txt(x0 - 12, yy + 4, f"{v:.1f}", 12, anchor="end", fill=MUTED)
    s += txt(x0, y0 + ph + 48, "action channel   d_action / |z|", 14,
             anchor="start", fill=INK)
    s += txt(x0, y0 - 12, "success rate", 13, anchor="start", fill=INK)

    marks = {}
    stack_n = 0
    for env, tag, filled in (("wall", "_wall", True), ("pusht", "", False)):
        for k, rec in d.items():
            if not k.startswith(env + "|"):
                continue
            arm = k.split("|", 1)[1]
            sr, dz = _mean(rec), _mean(rec, "dz")
            if sr is None or dz is None:
                continue
            cx = clamp(x0 + (min(dz, 0.75) / 0.75) * pw, x0 + 10, x0 + pw - 10)
            cy = clamp(y0 + ph - (min(sr, 0.8) / 0.8) * ph, y0 + 10, y0 + ph - 10)
            if dz <= 0.10:                     # the dead pile: fan it out so it is countable
                cx = clamp(cx + (stack_n % 5) * 15, x0 + 10, x0 + pw - 10)
                cy = clamp(cy - (stack_n // 5) * 15, y0 + 10, y0 + ph - 10)
                stack_n += 1
            col = BASEC if arm == "LpWM-ltv" else (TESTED if dz > BAND else DEAD)
            s += dot(cx, cy, 6.5, col, filled)
            marks[f"{env}|{arm}"] = (cx, cy, col)

    # THREE callouts, hand-placed in empty regions with leader lines. Labelling every point
    # made them collide with each other and with the marks; three named points carry the
    # reading and the rest are the distribution.
    notes = [("wall|LpWM-ltv", 700, 150, "baseline: healthy channel, plans"),
             ("wall|PiWM-enc-ssm", 250, 300,
              f"{stack_n} arms pile up here: channel destroyed,"),
             ("wall|PiWM-sinv2-shuf", 700, 496,
              "sinv2-shuf: channel 0.658, healthier than")]
    extra = {"wall|PiWM-enc-ssm": "every one plans at ~0",
             "wall|PiWM-sinv2-shuf": "the baseline -- and still plans at 0.050"}
    for key, tx, ty, lab in notes:
        if key not in marks:
            continue
        cx, cy, col = marks[key]
        s += (f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{tx}" y2="{ty}" stroke="{col}" '
              f'stroke-width="1.2" stroke-dasharray="3,3" opacity="0.75"/>\n')
        s += txt(tx, ty - 4, lab, 12, anchor="start", fill=col, weight="bold")
        if key in extra:
            s += txt(tx, ty + 12, extra[key], 12, anchor="start", fill=col)

    s += legend(x0, y0 + ph + 84,
                [(TESTED, "channel intact (> 0.40) -- result is interpretable"),
                 (DEAD, "channel destroyed -- result is not"),
                 (BASEC, "baseline")])
    s += txt(x0 + 470, y0 + ph + 88, "filled = wall      hollow = pushT", 12.5,
             anchor="start", fill=MUTED)
    s += txt(x0 + 470, y0 + ph + 108,
             "NOTE: a healthy channel does NOT imply success -- it only licenses reading the "
             "number.", 12, anchor="start", fill=INK)
    s += txt(x0 + 470, y0 + ph + 126,
             "sinv2-shuf sits at 0.658 and still plans at 0.050.", 12, anchor="start",
             fill=MUTED)
    return s, W, H


# ======================================================= 3. P6's mechanism
def p6_mechanism():
    """The round's one positive claim: P6 fails BECAUSE its objective succeeds."""
    d = _d()
    W, H = 1180, 800
    x0, y0, pw, ph = 170, 130, 780, 380
    s = txt(W / 2, 44, "P6 fails because it SUCCEEDS", 21, fill=INK, weight="bold")
    s += txt(W / 2, 68, "vel_rel is the objective's own success measure: 1.0 = the head learned "
             "nothing, 0 = it decodes motion perfectly", 13, fill=MUTED)

    s += (f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="{WHITE}" '
          f'stroke="{DIM}" stroke-width="1"/>\n')
    for i in range(6):
        v = i / 5
        xx = x0 + v * pw
        s += (f'<line x1="{xx:.1f}" y1="{y0}" x2="{xx:.1f}" y2="{y0+ph}" '
              f'stroke="#EEF1EF" stroke-width="1"/>\n')
        s += txt(xx, y0 + ph + 20, f"{v:.1f}", 12, fill=MUTED)
    for i in range(5):
        v = 0.8 * i / 4
        yy = y0 + ph - (v / 0.8) * ph
        s += (f'<line x1="{x0}" y1="{yy:.1f}" x2="{x0+pw}" y2="{yy:.1f}" '
              f'stroke="#EEF1EF" stroke-width="1"/>\n')
        s += txt(x0 - 12, yy + 4, f"{v:.2f}", 12, anchor="end", fill=MUTED)
    s += txt(x0, y0 + ph + 66, "vel_rel  --  objective SUCCESS increases to the LEFT", 13.5,
             anchor="start", fill=INK)
    s += txt(x0, y0 - 12, "action channel  d_action/|z|", 13, anchor="start", fill=INK)
    s += (f'<line x1="{x0+300}" y1="{y0+ph+44}" x2="{x0+150}" y2="{y0+ph+44}" '
          f'stroke="{DEAD}" stroke-width="2.5"/>\n')
    s += txt(x0 + 310, y0 + ph + 48, "objective succeeds this way", 12,
             anchor="start", fill=DEAD)

    pts = []
    for env, tag, filled in (("wall", "_wall", True), ("pusht", "", False)):
        rec = d.get(f"{env}|PiWM-vel2")
        if not rec:
            continue
        for seed in ("3", "4", "5"):
            vr = rec.get("vel_rel", {}).get(seed)
            dz = rec.get("dz", {}).get(seed)
            sr = rec.get("sr", {}).get(seed)
            if vr is None or dz is None:
                continue
            pts.append((vr, min(dz, 0.8), sr, env, filled))
    for j, (vr, dz, sr, env, filled) in enumerate(pts):
        cx = clamp(x0 + min(vr, 1.0) * pw, x0 + 12, x0 + pw - 12)
        cy = clamp(y0 + ph - (dz / 0.8) * ph, y0 + 12, y0 + ph - 12)
        col = TESTED if dz > BAND else DEAD
        s += dot(cx, cy, 7.5, col, filled)
        if sr is not None:
            off = (-17 if env == "wall" else 25) + (j % 2) * (-13 if env == "wall" else 13)
            s += txt(clamp(cx, x0 + 34, x0 + pw - 34), cy + off, f"SR {sr:.2f}", 11,
                     fill=col, weight="bold")
    s += txt(x0 + 30, y0 + ph - 34, "objective NEARLY PERFECT here", 12, anchor="start",
             fill=MUTED)
    s += txt(x0 + 30, y0 + ph - 18, "channel destroyed", 12, anchor="start", fill=DEAD,
             weight="bold")
    s += txt(x0 + pw - 30, y0 + 40, "objective BARELY LEARNED here", 12, anchor="end",
             fill=MUTED)
    s += txt(x0 + pw - 30, y0 + 56, "channel survives", 12, anchor="end", fill=TESTED,
             weight="bold")

    for vr, dz, sr, env, filled in pts:
        if dz >= 0.79:                      # clamped from 1.116 -- above the inverted-U optimum
            ox = clamp(x0 + min(vr, 1.0) * pw, x0 + 12, x0 + pw - 12)
            oy = clamp(y0 + ph - (dz / 0.8) * ph, y0 + 12, y0 + ph - 12)
            s += (f'<line x1="{ox:.1f}" y1="{oy:.1f}" x2="{x0+250}" y2="{y0+104}" '
                  f'stroke="{MUTED}" stroke-width="1.1" stroke-dasharray="3,3"/>\n')
            s += txt(x0 + 26, y0 + 56, "pushT s5: channel 1.12, ABOVE the", 11.5,
                     anchor="start", fill=MUTED)
            s += txt(x0 + 26, y0 + 71, "inverted-U optimum. Contradicts the", 11.5,
                     anchor="start", fill=MUTED)
            s += txt(x0 + 26, y0 + 86, "trend and is unexplained.", 11.5,
                     anchor="start", fill=MUTED)
            break

    s += txt(x0, y0 + ph + 84,
             "Forcing the code to carry inter-frame MOTION competes directly with carrying "
             "what ACTIONS do.", 13.5, anchor="start", fill=INK, weight="bold")
    s += txt(x0, y0 + ph + 106,
             "A trade-off, not a null: the objective is not ignored, it is satisfied at the "
             "cost of the thing planning needs. 5 of 6 runs fit; pushT s5 does not.", 12.5,
             anchor="start", fill=MUTED)
    s += legend(x0, y0 + ph + 142,
                [(TESTED, "channel intact"), (DEAD, "channel destroyed")])
    s += txt(x0 + 320, y0 + ph + 146, "filled = wall     hollow = pushT", 12.5,
             anchor="start", fill=MUTED)
    return s, W, H


FIGS = {"fig-r9-scoreboard": scoreboard,
        "fig-r9-action-channel": action_channel,
        "fig-r9-p6-mechanism": p6_mechanism}


def main():
    out = os.path.join(REPO, "diary", "assets", "2026-09-07")
    os.makedirs(out, exist_ok=True)
    from analysis.fig_audit import audit_all
    bad_total = 0
    for name, fn in FIGS.items():
        body, w, h = fn()
        emit(name + ".svg", body, out, w=w, h=h)
        path = os.path.join(out, name + ".svg")
        allow_regions = ()
        if name == "fig-r9-action-channel":
            allow_regions = ((150, 120, 620, 560),)     # the disqualification band
        bad = audit_all(path, w, h, frames=(), allow_regions=allow_regions)
        bad_total += len(bad)
        print(f"  {name}.svg  {w}x{h}  audit: {len(bad)} issue(s)")
        for b in bad[:6]:
            print(f"      {b}")
    print(f"figures: {len(FIGS)}   total audit issues: {bad_total}")


if __name__ == "__main__":
    main()
