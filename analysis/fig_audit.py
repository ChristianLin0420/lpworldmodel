"""Geometric audit for the SVG figures: shapes as well as text.

`analysis/round8_t_figs.audit()` measures only `<text>` elements, so a figure can pass it and
still have a box sitting on top of another box, or a connector running through a label. Every
"nothing overlaps" claim in rounds 6-8 rested on a check that could not see 1,632 rects, 489
paths and 75 circles.

WHAT COUNTS AS A COLLISION, AND WHAT DOES NOT.  Nesting is the normal state of these figures:
a panel contains frames, a frame contains boxes, a box contains its own caption.  A naive
pairwise intersection reports thousands of those and is useless.  So:

    containment      one box wholly inside another          -> NOT a collision
    partial overlap  two boxes intersect, neither contains  -> COLLISION
    off-canvas       anything outside the page              -> COLLISION

That distinction is the whole reason this file exists.  It is also why `boxes_shapes` skips
the full-page background rect (which contains everything by construction) and everything
inside `<defs>` / `<marker>` (arrowheads live in marker coordinate space, not canvas space,
so their coordinates are meaningless here -- a 10x10 marker path would otherwise appear to
collide with whatever sits near the origin).

    from analysis.fig_audit import audit_all
    bad = audit_all(path, W, h + SHIFT, frames)
"""
import re

from analysis.arch_figs import text_width  # noqa: F401  (kept for parity with the text audit)
from analysis.round8_t_figs import _boxes as boxes_text  # one definition, two consumers

_DEFS = re.compile(r"<defs\b.*?</defs>|<marker\b.*?</marker>", re.S)
_RECT = re.compile(r"<rect ([^>]*)/?>")
_CIRC = re.compile(r"<circle ([^>]*)/?>")
_ELL = re.compile(r"<ellipse ([^>]*)/?>")
_LINE = re.compile(r"<line ([^>]*)/?>")
_PATH = re.compile(r'<path [^>]*?d="([^"]*)"')
_ATTR = re.compile(r'(\S+)="([^"]*)"')
_NUM = re.compile(r"-?\d+\.?\d*")


def _f(d, k, default=0.0):
    try:
        return float(d.get(k, default))
    except (TypeError, ValueError):
        return default


def boxes_shapes(svg, page=None):
    """[(label, x0, y0, x1, y1)] for every DRAWN mark on the canvas.

    `page` is (w, h); the background rect covering it is dropped because it contains
    everything by construction and would make containment meaningless.
    """
    svg = _DEFS.sub("", svg)                       # markers/defs are in their own space
    dy = 0.0
    m = re.search(r'<g transform="translate\(0,(-?[\d.]+)\)"', svg)
    if m:
        dy = float(m.group(1))
    out = []

    for mt in _RECT.finditer(svg):
        a = dict(_ATTR.findall(mt.group(1)))
        x, y = _f(a, "x"), _f(a, "y") + dy
        w, h = _f(a, "width"), _f(a, "height")
        if w <= 0 or h <= 0:
            continue
        if page and w >= page[0] - 1 and h >= page[1] - 1:
            continue                               # the background
        out.append(("rect", x, y, x + w, y + h))

    for mt in _CIRC.finditer(svg):
        a = dict(_ATTR.findall(mt.group(1)))
        cx, cy, r = _f(a, "cx"), _f(a, "cy") + dy, _f(a, "r")
        if r > 0:
            out.append(("circle", cx - r, cy - r, cx + r, cy + r))

    for mt in _ELL.finditer(svg):
        a = dict(_ATTR.findall(mt.group(1)))
        cx, cy = _f(a, "cx"), _f(a, "cy") + dy
        rx, ry = _f(a, "rx"), _f(a, "ry")
        if rx > 0 and ry > 0:
            out.append(("ellipse", cx - rx, cy - ry, cx + rx, cy + ry))

    # lines and paths are CONNECTORS. They are meant to run between things, so they are
    # collected for off-canvas checking but excluded from pairwise overlap -- a wire
    # crossing a wire is a drawing, not a defect.
    for mt in _LINE.finditer(svg):
        a = dict(_ATTR.findall(mt.group(1)))
        x1, y1 = _f(a, "x1"), _f(a, "y1") + dy
        x2, y2 = _f(a, "x2"), _f(a, "y2") + dy
        out.append(("wire:line", min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))

    for mt in _PATH.finditer(svg):
        pts = _path_points(mt.group(1))
        if len(pts) < 2:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] + dy for p in pts]
        out.append(("wire:path", min(xs), min(ys), max(xs), max(ys)))

    return out


# SVG path commands, and how many numbers each consumes per repetition. Parsing `d` as a flat
# list of x,y pairs is WRONG and silently invents points: an arc is
# `A rx ry rot large-arc sweep x y` -- seven numbers -- so the flag triple `0 0 1` reads as a
# coordinate at the origin. Every dome in these figures is an arc, which is why the naive
# version reported three phantom off-canvas marks per figure at (0, -80).
_CMD_ARITY = {"M": 2, "L": 2, "T": 2, "H": 1, "V": 1,
              "C": 6, "S": 4, "Q": 4, "A": 7, "Z": 0}
_TOKEN = re.compile(r"([MmLlHhVvCcSsQqTtAaZz])|(-?\d+\.?\d*(?:[eE][-+]?\d+)?)")


def _path_points(d):
    """Points that bound a path. Control points are included (the curve lies inside their
    hull); an arc contributes its endpoint expanded by (rx, ry), which is conservative."""
    toks = [(c, n) for c, n in _TOKEN.findall(d)]
    pts, i, cmd, cur = [], 0, None, (0.0, 0.0)
    while i < len(toks):
        c, n = toks[i]
        if c:
            cmd = c
            i += 1
            if cmd.upper() == "Z":
                continue
        if cmd is None:
            i += 1
            continue
        k = _CMD_ARITY[cmd.upper()]
        nums = []
        while len(nums) < k and i < len(toks) and toks[i][1]:
            nums.append(float(toks[i][1]))
            i += 1
        if len(nums) < k:
            break
        rel = cmd.islower()
        u = cmd.upper()
        if u == "H":
            cur = (cur[0] + nums[0] if rel else nums[0], cur[1])
            pts.append(cur)
        elif u == "V":
            cur = (cur[0], cur[1] + nums[0] if rel else nums[0])
            pts.append(cur)
        elif u == "A":
            rx, ry, ex, ey = nums[0], nums[1], nums[5], nums[6]
            end = (cur[0] + ex, cur[1] + ey) if rel else (ex, ey)
            pts += [(end[0] - rx, end[1] - ry), (end[0] + rx, end[1] + ry)]
            cur = end
        else:                                     # M/L/T/C/S/Q -- trailing pairs are points
            for j in range(0, k, 2):
                p = (nums[j], nums[j + 1])
                p = (cur[0] + p[0], cur[1] + p[1]) if rel else p
                pts.append(p)
            cur = pts[-1]
    return pts


def _contains(a, b, tol=0.5):
    return (a[1] <= b[1] + tol and a[2] <= b[2] + tol
            and a[3] >= b[3] - tol and a[4] >= b[4] - tol)


def _overlap(a, b):
    return (min(a[3], b[3]) - max(a[1], b[1]),
            min(a[4], b[4]) - max(a[2], b[2]))


def audit_all(path, w, h, frames=(), slack=1.0, pad=9.0, allow=(), allow_regions=()):
    """Text and shape collisions, off-canvas marks, and frame crossings.

    Returns a list of human-readable strings, empty when the figure is clean.

    `allow` holds substrings; a text pair is skipped when either label contains one of them.

    `allow_regions` holds (x0, y0, x1, y1) rectangles in which shape overlap is DELIBERATE --
    a cascade of frames drawn to read as a clip, a stack of cards. Overlap there is the
    drawing, not a defect. `round8_gap_figs.clip3()` is the existing example: three 42-unit
    squares offset by 9, which overlap by 33 on purpose. Declaring the region is better than
    loosening `slack`, which would blind the check everywhere else.
    """
    with open(path) as f:
        svg = f.read()

    from analysis.round8_t_figs import audit as _text_audit
    bad = list(_text_audit(path, w, h, frames, slack=slack, pad=pad))

    shapes = boxes_shapes(svg, page=(w, h))
    marks = [s for s in shapes if not s[0].startswith("wire:")]

    for s in shapes:                                # off-canvas applies to wires too
        if s[1] < -1 or s[3] > w + 1 or s[2] < -1 or s[4] > h + 1:
            bad.append(f"SHAPE OFF-CANVAS  {s[0]}  ({s[1]:.0f},{s[2]:.0f})-"
                       f"({s[3]:.0f},{s[4]:.0f})  canvas {w}x{h}")

    def _declared(a, b):
        cx = (max(a[1], b[1]) + min(a[3], b[3])) / 2.0
        cy = (max(a[2], b[2]) + min(a[4], b[4])) / 2.0
        return any(r[0] <= cx <= r[2] and r[1] <= cy <= r[3] for r in allow_regions)

    for i in range(len(marks)):
        for j in range(i + 1, len(marks)):
            a, b = marks[i], marks[j]
            if _contains(a, b) or _contains(b, a):
                continue                            # nesting is how these figures are built
            if _declared(a, b):
                continue                            # a cascade the figure declares on purpose
            ow, oh = _overlap(a, b)
            if ow > slack and oh > slack:
                bad.append(f"SHAPE OVERLAP  {a[0]} ({a[1]:.0f},{a[2]:.0f}) x "
                           f"{b[0]} ({b[1]:.0f},{b[2]:.0f})   ({ow:.0f}x{oh:.0f} units)")

    texts = boxes_text(svg)
    for t in texts:
        for s in marks:
            if any(k in str(t[0]) for k in allow):
                continue
            if _contains(s, t):
                continue                            # a label inside its own box
            ow, oh = _overlap(t, s)
            if ow > slack and oh > slack:
                bad.append(f"TEXT ON SHAPE  {str(t[0])[:38]!r} x {s[0]} "
                           f"({s[1]:.0f},{s[2]:.0f})   ({ow:.0f}x{oh:.0f} units)")
    return bad
