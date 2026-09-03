"""Recover the block pose (x, y, theta) from a 224x224 PushT render.

Why: the archive stores no per-episode terminal state (M2's traces are the fix, but
they only exist for runs launched from now on).  It DOES store, for episodes 0-9 of
every run, `output_final_real_{i}_{success|failure}.png` -- a strip whose last
executed frame is the state the run was scored at.  Recovering the block pose from
that frame re-scores the whole archive with no GPU.

Method: the block is the only LightSlateGray object.  Segment it by nearest-palette
colour, then fit (x, y, theta) by maximising mask IoU against an analytic
rasterisation of the T polygon.  theta alone is searched (a 1-D scan + refine);
(x, y) follow from matching the observed mask centroid, because the predicted
centroid is pose_xy + R(theta) @ c_local.
"""
import numpy as np
from PIL import Image, ImageDraw

W = 512.0                      # pymunk world size
R = 224                        # render size

# palette, as it lands in the PNG (pymunk debug-draw lightens fills by 1.2)
PALETTE = {
    "bg":         (255, 255, 255),
    "block_fill": (143, 163, 184),
    "block_line": (119, 136, 153),
    "agent_fill": (78, 126, 255),
    "agent_line": (65, 105, 225),
    "goalzone":   (144, 238, 144),
    "wall_fill":  (253, 253, 253),
    "wall_line":  (211, 211, 211),
}
_NAMES = list(PALETTE)
_COLS = np.array([PALETTE[k] for k in _NAMES], dtype=np.float32)
BLOCK_IDX = {_NAMES.index("block_fill"), _NAMES.index("block_line")}
AGENT_IDX = {_NAMES.index("agent_fill"), _NAMES.index("agent_line")}


def classify(img):
    a = np.asarray(img, dtype=np.float32)
    d = ((a[:, :, None, :] - _COLS[None, None, :, :]) ** 2).sum(-1)
    return d.argmin(-1)


def masks(img):
    lab = classify(img)
    block = np.isin(lab, list(BLOCK_IDX))
    agent = np.isin(lab, list(AGENT_IDX))
    return block, agent


# --- the T polygon in body-local coordinates (add_tee, scale=30) -----------------
_S = 30.0
_L = 4.0
_V1 = np.array([(-_L * _S / 2, _S), (_L * _S / 2, _S), (_L * _S / 2, 0), (-_L * _S / 2, 0)])
_V2 = np.array([(-_S / 2, _S), (-_S / 2, _L * _S), (_S / 2, _L * _S), (_S / 2, _S)])
_POLYS = [_V1, _V2]


C_LOCAL = None  # set below, once the dilated outline exists


# The env draws each block edge as a fat segment of radius 2 (world units) in
# `DrawOptions.draw_polygon`, so the rendered block is the T DILATED by 2px.  Model
# that once, in body-local coordinates, instead of fighting the resulting bias.
import shapely.geometry as _sg
import shapely.ops as _so

_UNION = _so.unary_union([_sg.Polygon(p) for p in _POLYS]).buffer(2.0, resolution=4)
_OUTLINE = np.asarray(_UNION.exterior.coords)

# pixel<->world mapping.  pygame draws at world resolution (pixel j <-> world j),
# then cv2.resize(512 -> R) puts dst pixel j at world (j+0.5)*W/R - 0.5.  PIL
# rasterises with pixel centres at integer coordinates, so:
#     px(j) = (world + 0.5) * (R*ss/W) - 0.5
def _to_px(w, size, ss):
    return (w + 0.5) * (size * ss / W) - 0.5


def render_mask(pose, size=R, supersample=2):
    """rasterise the block AS THE ENV DRAWS IT at (x, y, theta) -> bool mask"""
    x, y, th = float(pose[0]), float(pose[1]), float(pose[2])
    c, s = np.cos(th), np.sin(th)
    Rm = np.array([[c, -s], [s, c]])
    im = Image.new("1", (size * supersample, size * supersample), 0)
    dr = ImageDraw.Draw(im)
    w = (_OUTLINE @ Rm.T) + np.array([x, y])
    dr.polygon([tuple(v) for v in _to_px(w, size, supersample)], fill=1)
    m = np.asarray(im, dtype=np.uint8)
    if supersample > 1:
        m = m.reshape(size, supersample, size, supersample).mean((1, 3)) > 0.5
    return m.astype(bool)


def _iou(a, b, valid=None):
    if valid is not None:
        a = a & valid
        b = b & valid
    i = np.count_nonzero(a & b)
    u = np.count_nonzero(a | b)
    return i / u if u else 0.0


def _pose_for(th, c_obs):
    c, s = np.cos(th), np.sin(th)
    Rm = np.array([[c, -s], [s, c]])
    return np.array([*(c_obs - Rm @ C_LOCAL), th])


def _down(m, f):
    h = m.shape[0] // f
    return m[: h * f, : h * f].reshape(h, f, h, f).mean((1, 3)) > 0.5


def fit_pose(img, n_theta=72, coarse=56):
    """returns (pose, iou, n_block_px); pose is (x, y, theta) in world units.

    Coarse-to-fine: a 1-D theta scan on a 4x-downsampled mask, then coordinate
    descent on all three dof at full resolution.  (x, y) are never searched blind --
    they follow from the observed mask centroid for any candidate theta.
    """
    block, agent = masks(img)
    n = int(block.sum())
    if n < 50:
        return None, 0.0, n
    valid = ~agent
    rows, cols = np.nonzero(block)
    c_obs = CENTROID_PX_TO_WORLD(cols.mean(), rows.mean())

    f = R // coarse
    b_c, v_c = _down(block, f), _down(valid, f)
    best_s, best_t = -1.0, 0.0
    for th in np.linspace(0, 2 * np.pi, n_theta, endpoint=False):
        sc = _iou(render_mask(_pose_for(th, c_obs), size=coarse, supersample=1), b_c, v_c)
        if sc > best_s:
            best_s, best_t = sc, th

    pose = _pose_for(best_t, c_obs)
    score = _iou(render_mask(pose), block, valid)
    step = np.array([2.0, 2.0, np.pi / n_theta])
    for _ in range(60):
        improved = False
        for d in range(3):
            for sgn in (1, -1):
                cand = pose.copy()
                cand[d] += sgn * step[d]
                sc = _iou(render_mask(cand), block, valid)
                if sc > score:
                    score, pose, improved = sc, cand, True
        if not improved:
            step = step / 2.0
            if step[0] < 0.05:
                break
    pose[2] = pose[2] % (2 * np.pi)
    return pose, score, n


C_LOCAL = np.asarray(_UNION.centroid.coords)[0]
# also fix the world<->pixel mapping used for the OBSERVED centroid
CENTROID_PX_TO_WORLD = lambda cx, cy: (np.array([cx, cy]) + 0.5) * (W / R) - 0.5
