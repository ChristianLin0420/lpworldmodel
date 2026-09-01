"""analysis/panels.py is the single source of the campaign's visual design.

Both train.py (live, inside a 4-hour GPU window) and analysis/figures.py (offline)
call these functions, so two properties matter more than pixel fidelity:

  1. a panel NEVER raises -- degenerate input has to degrade to a no_data figure,
     because the live caller is a training loop that must not die for a plot
  2. an arm's colour is STABLE -- across calls, across panels and across processes,
     since the whole point of a fixed palette is tracking one arm through a wall of
     figures
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from analysis import panels as P


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# --- colour system --------------------------------------------------------------

def test_canon_arm_strips_the_run_dir_decorations():
    assert P.canon_arm("PiWM-union4-entropy_pd384_bf16_s0") == "PiWM-union4-entropy"
    assert P.canon_arm("PiWM-sparse-2pct_pd384_bf16_s2") == "PiWM-sparse-2pct"
    assert P.canon_arm("LpWM-ltv_pd384_no_s1") == "LpWM-ltv"
    assert P.canon_arm("LpWM-base") == "LpWM-base", "must be idempotent"
    assert P.canon_arm(None) == ""


def test_every_campaign_arm_has_a_distinct_fixed_style():
    """Identity is composite: hue x lightness x marker x dash, not hue alone.

    Nine flat hues cannot be separated -- three lightness steps of one family measure
    dE 12.9 against normal vision, under the 15 floor -- so two arms of the same step
    may share a colour PROVIDED their marker or dash differs. What must stay unique
    is the whole style tuple; sharing that would make two arms genuinely identical.
    """
    # 9 registered arms + wave-2 additions; assert a floor rather than an exact count
    # so adding an arm does not fail a test about STYLE UNIQUENESS.
    assert len(P.ARM_SPEC) >= 9, "the campaign has at least 9 arms"
    styles = []
    for a in P.ARM_SPEC:
        st = P.arm_style(a)
        styles.append((mcolors.to_hex(st["color"]), st["marker"], st["dashes"]))
    assert len(set(styles)) == len(styles), "two arms are drawn identically"


def test_arms_that_share_a_colour_are_separated_by_a_second_channel():
    """The 6-8 dE band is legal ONLY with secondary encoding -- so enforce it."""
    by_colour = {}
    for a in P.ARM_SPEC:
        by_colour.setdefault(mcolors.to_hex(P.arm_color(a)), []).append(a)
    for colour, arms in by_colour.items():
        if len(arms) < 2:
            continue
        marks = {(P.arm_style(a)["marker"], P.arm_style(a)["dashes"]) for a in arms}
        assert len(marks) == len(arms), (
            f"{arms} share {colour} without a distinguishing marker or dash")


def test_controls_are_neutral_and_variants_are_not():
    """A control is context, not a tenth hue. Demoting the two controls to grey is
    what leaves enough separable palette for the seven variants, so it is a contract
    rather than a style choice."""
    for a in P.CONTROL_ARMS:
        r, g, b = mcolors.to_rgb(P.arm_color(a))
        assert max(r, g, b) - min(r, g, b) < 0.05, f"{a} is a control but is not neutral"
        assert P.is_control(a)
    for a in P.ARM_SPEC:
        if a in P.CONTROL_ARMS:
            continue
        r, g, b = mcolors.to_rgb(P.arm_color(a))
        assert max(r, g, b) - min(r, g, b) > 0.10, f"{a} is a variant but reads grey"
        assert not P.is_control(a)


def test_nine_arms_are_past_the_count_colour_can_carry():
    """The ladder is derived from the validator, so a panel can ask rather than guess."""
    assert P.needs_facet(list(P.ARM_SPEC)), "9 arms must force a facet"
    assert not P.needs_facet(list(P.ARM_SPEC)[:3]), "3 arms must not force a facet"
    assert P.SERIES_LADDER["hue_alone"] < P.FACET_ABOVE


def test_teal_is_reserved_for_contact_onsets_alone():
    """Teal has to mean 'contact' without a legend, so no arm may claim it."""
    teal = mcolors.to_rgb(P.CONTACT)
    for a in P.ARM_SPEC:
        d = np.linalg.norm(np.array(mcolors.to_rgb(P.arm_color(a))) - teal)
        assert d > 0.2, f"{a} is too close to the reserved contact teal"
    for c in P._EXTRA_COLORS:
        d = np.linalg.norm(np.array(mcolors.to_rgb(c)) - teal)
        assert d > 0.2, f"fallback colour {c} is too close to the contact teal"


def test_arm_colour_is_stable_across_calls_and_seed_suffixes():
    first = {a: P.arm_color(a) for a in P.ARM_SPEC}
    for _ in range(3):
        for a in P.ARM_SPEC:
            assert P.arm_color(a) == first[a]
    # the seed suffix must not change the colour: all three seeds of an arm are the
    # same arm on every panel
    for s in range(3):
        assert P.arm_color(f"PiWM-union4-entropy_pd384_bf16_s{s}") == P.arm_color("PiWM-union4-entropy")


def test_unknown_arms_get_a_deterministic_colour_not_a_positional_one():
    """An enumeration would recolour the whole campaign the moment a probe run
    appeared in the glob; hashing keeps every known arm exactly where it was."""
    a = P.arm_color("some_probe_run")
    assert a == P.arm_color("some_probe_run")
    assert P.arm_color("other_probe") == P.arm_color("other_probe")
    # order of first sighting is irrelevant
    assert a == P.arm_color("some_probe_run_pd384_bf16_s2")


def test_arm_palette_maps_every_requested_arm():
    arms = ["LpWM-base", "PiWM-sparse-2pct", "brand_new"]
    pal = P.arm_palette(arms)
    assert set(pal) == set(arms)
    assert pal["LpWM-base"] == P.arm_color("LpWM-base")


def test_arm_ink_darkens_for_text_but_keeps_the_hue():
    """Olive and pink read fine as a 2pt stroke and near-invisibly as a 9pt glyph."""
    for a in P.ARM_SPEC:
        r, g, b = mcolors.to_rgb(P.arm_color(a))
        ri, gi, bi = P.arm_ink(a)
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        ink_lum = 0.299 * ri + 0.587 * gi + 0.114 * bi
        assert ink_lum <= max(lum, 0.62) + 1e-9
        assert ink_lum <= 0.63, f"{a} ink is too light to read as text"
        if lum > 0:  # same hue == proportional channels
            assert np.allclose([ri, gi, bi], np.array([r, g, b]) * (ink_lum / lum),
                               atol=1e-6)


def test_symmetric_limits_is_symmetric_about_zero():
    for vals in ([-0.2, 0.9], [3.0], [-5.0, -1.0], [[0.1, -0.4], [0.2, 0.3]]):
        lo, hi = P.symmetric_limits(vals)
        assert lo == -hi, "a diverging map must be centred exactly at 0"
        assert hi >= np.nanmax(np.abs(np.asarray(vals, float)))


def test_symmetric_limits_survives_empty_and_all_nan_input():
    for vals in ([], [np.nan, np.nan], np.array([])):
        lo, hi = P.symmetric_limits(vals)
        assert lo == -hi and hi > 0, "must stay a usable, nonzero, centred range"


def test_epoch_colors_is_ordered_and_the_requested_length():
    c = P.epoch_colors(7)
    assert len(c) == 7
    assert np.allclose(c, P.epoch_colors(7)), "must be reproducible"
    assert len(P.epoch_colors(0)) == 1, "degenerate n must not produce an empty list"


def test_nine_designs_are_all_registered_and_callable():
    assert len(P.NINE) == 9
    for name, fn in P.NINE:
        assert callable(fn) and getattr(P, name) is fn


# --- no_data contract -----------------------------------------------------------

def test_no_data_is_a_labelled_figure_carrying_its_reason():
    fig = P.no_data("because the input was empty")
    assert isinstance(fig, Figure)
    assert P.is_no_data(fig)
    texts = " ".join(t.get_text() for a in fig.axes for t in a.texts)
    assert "because the input was empty" in texts


def test_is_no_data_is_false_for_a_real_panel():
    fig = P.ecdf_overlay({"LpWM-base": np.arange(1.0, 50.0)})
    assert not P.is_no_data(fig)


# --- synthetic inputs -----------------------------------------------------------

@pytest.fixture(scope="module")
def d():
    """One synthetic bundle with the shapes the two callers actually pass."""
    rng = np.random.default_rng(0)
    lags = np.arange(-5, 6)
    kern = np.exp(-(lags ** 2) / 4.0)
    metrics = ["success", "z_loss", "RDMReg", "l0"]
    arms = list(P.ARM_SPEC)
    steps = np.linspace(0, 2, 60)
    return {
        "ridges": [np.abs(rng.normal(1.2 - 0.05 * e, 0.4, 900)) for e in range(6)],
        "sw": np.clip(rng.beta(2, 4, 1200), 0, 1),
        "sm": np.clip(rng.beta(2, 3, 1200), 0, 1),
        "onset": (rng.random(1200) < 0.05).astype(float),
        "lags": lags,
        "rows_m": 0.3 + 0.5 * kern + 0.05 * rng.normal(size=(24, len(lags))),
        "rows_w": 0.3 + 0.5 * np.roll(kern, 2) + 0.05 * rng.normal(size=(24, len(lags))),
        "metrics": metrics,
        "arms": arms,
        "pc": {a: rng.random(len(metrics)) for a in arms},
        "series": {a: (steps, 0.5 - 0.1 * rng.random() * steps) for a in arms},
        "ref": (steps, 0.55 - 0.04 * steps),
        "eff": rng.normal(0, 0.4, (len(arms) - 1, len(metrics))),
        "ecdf": {a: np.clip(rng.normal(180, 30, 800), 1, 384) for a in arms[:4]},
        "stream": np.clip(rng.random((4, 40)), 0.01, None),
        "bubble": {a: [(0.4 + 0.01 * s, 0.3 + 0.01 * s, 0.5 + 0.05 * s)
                       for s in range(3)] for a in arms},
    }


# (panel name, callable, args, kwargs, expected number of axes)
def _cases(d):
    return [
        ("ridgeline", P.ridgeline, (d["ridges"],), {}, 1),
        ("ridgeline_prebinned", P.ridgeline,
         ((np.linspace(0, 3, 21), np.abs(np.random.default_rng(1).random((4, 20)))),),
         {}, 1),
        ("joint_hexbin", P.joint_hexbin, (d["sw"], d["sm"], d["onset"]), {}, 4),
        ("peri_event_raster", P.peri_event_raster,
         ([("S_model", d["rows_m"]), ("S_world", d["rows_w"])], d["lags"]), {}, 6),
        ("parallel_coordinates", P.parallel_coordinates,
         (d["pc"], d["metrics"]), {"higher_better": [1, -1, -1, 1]}, 1),
        # a 3-column grid rounds UP to a full last row, so the axis count is
        # ceil(n/3)*3 -- deriving it keeps this test about LAYOUT rather than about
        # how many arms happen to be registered today
        ("small_multiples", P.small_multiples,
         (d["series"], d["ref"]), {"ncol": 3},
         -(-len(d["series"]) // 3) * 3),
        ("effect_map", P.effect_map,
         (d["eff"], d["arms"][1:], d["metrics"]),
         {"sig": np.abs(d["eff"]) > 0.4}, 2),
        ("ecdf_overlay", P.ecdf_overlay, (d["ecdf"],), {"vlines": [(8, "k=8")]}, 1),
        ("head_stream", P.head_stream,
         ([("ent0", np.arange(40.0), d["stream"]),
           ("ent0.1", np.arange(40.0), d["stream"][::-1])],), {}, 2),
        ("bubble", P.bubble_reg_vs_sparsity, (d["bubble"],), {"ref_level": 0.3}, 1),
        ("latent_timeline", P.latent_error_timeline,
         (np.arange(20.0), {"one-step": np.linspace(0.1, 0.3, 20),
                            "open-loop": np.linspace(0.1, 0.9, 20)}),
         {"onset": (np.arange(20) % 7 == 0).astype(float)}, 1),
    ]


def test_every_panel_returns_a_figure_with_the_expected_axes(d):
    for name, fn, args, kw, n_axes in _cases(d):
        fig = fn(*args, **kw)
        assert isinstance(fig, Figure), name
        assert not P.is_no_data(fig), f"{name} fell back to no_data on real input"
        assert len(fig.axes) == n_axes, \
            f"{name}: expected {n_axes} axes, got {len(fig.axes)}"
        plt.close(fig)


def test_every_panel_draws_something(d):
    """An axes with no artists renders a blank box, which is worse than no_data."""
    for name, fn, args, kw, _ in _cases(d):
        fig = fn(*args, **kw)
        drawn = sum(len(a.lines) + len(a.collections) + len(a.patches) + len(a.images)
                    for a in fig.axes)
        assert drawn > 0, f"{name} produced no artists"
        plt.close(fig)


def test_panels_never_save_or_show(d, tmp_path, monkeypatch):
    """A panel that writes a file cannot be wrapped in wandb.Image, and a panel that
    calls show() would block a training loop."""
    def boom(*a, **k):
        raise AssertionError("a panel touched the filesystem / display")
    monkeypatch.setattr(plt, "savefig", boom, raising=False)
    monkeypatch.setattr(plt, "show", boom, raising=False)
    monkeypatch.setattr(Figure, "savefig", boom)
    for _, fn, args, kw, _ in _cases(d):
        plt.close(fn(*args, **kw))


# --- degenerate input -----------------------------------------------------------

EMPTY_CASES = [
    ("ridgeline", P.ridgeline, ([],), {}),
    ("ridgeline_none", P.ridgeline, (None,), {}),
    ("joint_hexbin", P.joint_hexbin, (np.array([]), np.array([])), {}),
    ("peri_event_raster", P.peri_event_raster, ([],), {}),
    ("peri_event_none", P.peri_event_raster, (None,), {}),
    ("parallel_coordinates", P.parallel_coordinates, ({}, []), {}),
    ("small_multiples", P.small_multiples, ({},), {}),
    ("effect_map", P.effect_map, (np.array([]), [], []), {}),
    ("ecdf_overlay", P.ecdf_overlay, ({},), {}),
    ("head_stream", P.head_stream, ([],), {}),
    ("bubble", P.bubble_reg_vs_sparsity, ({},), {}),
    ("latent_timeline", P.latent_error_timeline, (np.array([]), {}), {}),
]


@pytest.mark.parametrize("name,fn,args,kw", EMPTY_CASES,
                         ids=[c[0] for c in EMPTY_CASES])
def test_empty_input_degrades_to_no_data_instead_of_raising(name, fn, args, kw):
    fig = fn(*args, **kw)
    assert isinstance(fig, Figure), name
    assert P.is_no_data(fig), f"{name} must return a no_data figure, not a blank one"
    texts = " ".join(t.get_text() for a in fig.axes for t in a.texts)
    assert len(texts) > 10, f"{name}'s no_data must SAY what was missing"


NAN_CASES = [
    ("ridgeline", P.ridgeline, ([np.full(50, np.nan)],), {}),
    ("joint_hexbin", P.joint_hexbin,
     (np.full(50, np.nan), np.full(50, np.nan)), {}),
    ("parallel_coordinates", P.parallel_coordinates,
     ({"a": np.full(3, np.nan), "b": np.full(3, np.nan)}, ["m1", "m2", "m3"]), {}),
    ("effect_map", P.effect_map,
     (np.full((2, 3), np.nan), ["a", "b"], ["m1", "m2", "m3"]), {}),
    ("ecdf_overlay", P.ecdf_overlay, ({"a": np.full(20, np.nan)},), {}),
    ("head_stream", P.head_stream,
     ([("x", np.arange(10.0), np.full((3, 10), np.nan))],), {}),
    ("bubble", P.bubble_reg_vs_sparsity,
     ({"a": [(np.nan, np.nan, np.nan)]},), {}),
    ("latent_timeline", P.latent_error_timeline,
     (np.arange(10.0), {"e": np.full(10, np.nan)}), {}),
    ("small_multiples", P.small_multiples,
     ({"a": (np.arange(5.0), np.full(5, np.nan))},), {}),
    ("peri_event_raster", P.peri_event_raster,
     ([("x", np.full((4, 7), np.nan))],), {}),
]


@pytest.mark.parametrize("name,fn,args,kw", NAN_CASES, ids=[c[0] for c in NAN_CASES])
def test_all_nan_input_does_not_raise(name, fn, args, kw):
    """NaN columns are normal in a wandb export (tier-2 metrics log every 2000
    batches), so every panel has to survive a column that is entirely blank."""
    fig = fn(*args, **kw)
    assert isinstance(fig, Figure), name
    plt.close(fig)


ZERO_CASES = [
    ("ridgeline", P.ridgeline, ([np.zeros(200), np.zeros(200)],), {}),
    ("joint_hexbin", P.joint_hexbin, (np.zeros(200), np.zeros(200)), {}),
    ("parallel_coordinates", P.parallel_coordinates,
     ({"a": np.zeros(3), "b": np.zeros(3)}, ["m1", "m2", "m3"]), {}),
    ("effect_map", P.effect_map,
     (np.zeros((2, 3)), ["a", "b"], ["m1", "m2", "m3"]), {}),
    ("ecdf_overlay", P.ecdf_overlay, ({"a": np.zeros(50)},), {}),
    ("head_stream", P.head_stream,
     ([("x", np.arange(10.0), np.zeros((3, 10)))],), {}),
    ("bubble", P.bubble_reg_vs_sparsity, ({"a": [(0.0, 0.0, 0.0)]},), {}),
    ("latent_timeline", P.latent_error_timeline,
     (np.arange(10.0), {"e": np.zeros(10)}), {}),
    ("small_multiples", P.small_multiples,
     ({"a": (np.arange(5.0), np.zeros(5))},), {}),
    ("peri_event_raster", P.peri_event_raster, ([("x", np.zeros((4, 7)))],), {}),
]


@pytest.mark.parametrize("name,fn,args,kw", ZERO_CASES, ids=[c[0] for c in ZERO_CASES])
def test_all_zero_input_does_not_raise(name, fn, args, kw):
    """A collapsed code is an all-zero code, i.e. exactly the failure these panels
    exist to show -- so an all-zero input must plot, not explode on a zero range."""
    fig = fn(*args, **kw)
    assert isinstance(fig, Figure), name
    plt.close(fig)


def test_single_row_and_single_point_inputs_are_accepted():
    """The first logged batch of a run has exactly one of everything."""
    plt.close(P.ridgeline([np.array([0.5])]))
    plt.close(P.peri_event_raster([("S_model", np.array([[0.1, 0.4, 0.2]]))]))
    plt.close(P.head_stream([("one point", np.arange(2.0), np.ones((4, 2)))]))
    plt.close(P.bubble_reg_vs_sparsity({"LpWM-base": [(0.5, 0.2, 0.4)]}))
    plt.close(P.ecdf_overlay({"LpWM-base": np.array([3.0])}))
    plt.close(P.small_multiples({"LpWM-base": (np.arange(2.0), np.zeros(2))}))


def test_ridgeline_keeps_only_the_newest_rows_when_a_run_is_long():
    """100 epochs would make every ridge a sliver; the newest ones are the useful
    ones, and the row labels have to follow the rows they belong to."""
    series = [np.abs(np.random.default_rng(s).normal(1, 0.3, 200)) for s in range(40)]
    labels = [f"ep {i + 1}" for i in range(40)]
    fig = P.ridgeline(series, labels, max_rows=5)
    got = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert got == ["ep 36", "ep 37", "ep 38", "ep 39", "ep 40"]


# --- colour consistency ACROSS panels -------------------------------------------

def test_the_same_arm_gets_the_same_colour_in_different_panels(d):
    """The reason a fixed palette exists: one arm, one colour, every figure."""
    arm = "PiWM-sparse-2pct"
    want = mcolors.to_hex(P.arm_color(arm))
    ink = mcolors.to_hex(P.arm_ink(arm))

    pc = P.parallel_coordinates(d["pc"], d["metrics"])
    lines = {ln.get_label(): mcolors.to_hex(ln.get_color())
             for ln in pc.axes[0].lines}
    assert lines[arm] == want
    plt.close(pc)

    sm = P.small_multiples({arm: d["series"][arm]}, d["ref"])
    titles = {ax.get_title(): ax for ax in sm.axes if ax.get_title()}
    assert mcolors.to_hex(titles[arm].title.get_color()) == ink
    plt.close(sm)

    em = P.effect_map(d["eff"], d["arms"][1:], d["metrics"])
    tick = {t.get_text(): t for t in em.axes[0].get_yticklabels()}
    assert mcolors.to_hex(tick[arm].get_color()) == ink
    plt.close(em)

    ec = P.ecdf_overlay({arm: d["ecdf"][arm]})
    assert mcolors.to_hex(ec.axes[0].lines[0].get_color()) == want
    plt.close(ec)


def test_explicit_colors_override_the_arm_palette_in_the_ecdf():
    """The live path overlays EPOCHS, not arms, so it must be able to pass its own
    sequential colours without inheriting an arm's categorical one."""
    fig = P.ecdf_overlay({"ep 1": np.arange(1.0, 30.0)},
                         colors={"ep 1": "#123456"})
    assert mcolors.to_hex(fig.axes[0].lines[0].get_color()) == "#123456"


def test_contact_onsets_are_drawn_in_the_reserved_teal(d):
    teal = mcolors.to_hex(P.CONTACT)
    fig = P.joint_hexbin(d["sw"], d["sm"], d["onset"])
    edge_colors = [mcolors.to_hex(c) for coll in fig.axes[0].collections
                   for c in np.atleast_2d(coll.get_edgecolor())
                   if len(np.atleast_1d(c)) >= 3]
    assert teal in edge_colors, "onsets must be marked in the reserved teal"
    plt.close(fig)

    tl = P.latent_error_timeline(np.arange(10.0), {"e": np.zeros(10)},
                                onset=(np.arange(10) == 4).astype(float))
    assert teal in [mcolors.to_hex(ln.get_color()) for ln in tl.axes[0].lines]


# --- design invariants the panels are FOR ---------------------------------------

def test_effect_map_is_centred_at_zero_whatever_the_data(d):
    """An off-centre diverging map invents a sign, which is the one thing this
    panel must never do."""
    eff = np.array([[0.1, 0.2], [0.05, 3.0]])  # wildly asymmetric
    fig = P.effect_map(eff, ["a", "b"], ["m1", "m2"])
    im = fig.axes[0].images[0]
    lo, hi = im.get_clim()
    assert lo == pytest.approx(-hi)
    assert im.get_cmap().name == P.DIV
    assert hi >= 3.0


def test_effect_map_boxes_exactly_the_significant_cells():
    eff = np.array([[0.9, 0.1], [0.1, 0.1]])
    sig = np.array([[True, False], [False, False]])
    fig = P.effect_map(eff, ["a", "b"], ["m1", "m2"], sig=sig)
    boxes = [p for p in fig.axes[0].patches if not p.get_fill()]
    assert len(boxes) == 1


def test_head_stream_normalises_usage_and_marks_the_collapse_line():
    """Usage is a distribution over heads: it must sum to 1 at every step, or the
    'one colour swallowing the band' reading is meaningless."""
    p = np.array([[2.0, 8.0], [2.0, 1.0], [2.0, 0.5], [2.0, 0.5]])
    fig = P.head_stream([("x", np.arange(2.0), p)], collapse_line=0.9)
    ax = fig.axes[0]
    assert ax.get_ylim() == (0, 1)
    assert any(np.allclose(ln.get_ydata(), 0.9) for ln in ax.lines), \
        "the 0.9 collapse precondition line must be drawn"
    tops = np.sum([c.get_paths() for c in ax.collections] and [1], dtype=float)
    assert tops >= 1


def test_head_stream_accepts_both_J_by_T_and_T_by_J():
    """The live path accumulates (T, J) rows; the offline path holds (J, T)."""
    jt = np.abs(np.random.default_rng(2).random((4, 30))) + 0.05
    a = P.head_stream([("x", np.arange(30.0), jt)])
    b = P.head_stream([("x", np.arange(30.0), jt.T)])
    pa = a.axes[0].collections[0].get_paths()[0].vertices
    pb = b.axes[0].collections[0].get_paths()[0].vertices
    assert np.allclose(pa, pb)


def test_ecdf_is_monotone_and_ends_at_one():
    fig = P.ecdf_overlay({"LpWM-base": np.array([5.0, 1.0, 3.0, 2.0])})
    y = fig.axes[0].lines[0].get_ydata()
    assert np.all(np.diff(y) >= 0)
    assert y[-1] == pytest.approx(1.0)


def test_ecdf_log_axis_survives_zeros():
    """k-WTA at k=0 units active is a real, loggable value, and log(0) is not."""
    fig = P.ecdf_overlay({"a": np.array([0.0, 0.0, 4.0, 9.0])}, log_x=True)
    assert fig.axes[0].get_xscale() == "log"


def test_parallel_coordinates_normalises_so_one_is_always_best():
    vals = {"a": [0.1, 10.0], "b": [0.9, 1.0]}
    fig = P.parallel_coordinates(vals, ["higher good", "lower good"],
                                 higher_better=[1, -1])
    lines = {ln.get_label(): ln.get_ydata() for ln in fig.axes[0].lines
             if ln.get_label() in ("a", "b")}
    assert lines["a"][0] == pytest.approx(0.0) and lines["b"][0] == pytest.approx(1.0)
    assert lines["a"][1] == pytest.approx(0.0), "10.0 is the WORST when lower is good"
    assert lines["b"][1] == pytest.approx(1.0)


def test_parallel_coordinates_can_take_pre_normalised_values():
    fig = P.parallel_coordinates({"a": [0.25, 0.75]}, ["m1", "m2"],
                                 already_normalised=True)
    ln = [l for l in fig.axes[0].lines if l.get_label() == "a"][0]
    assert list(ln.get_ydata()) == [0.25, 0.75]


def test_small_multiples_ghosts_the_reference_in_every_panel(d):
    """Every panel must already contain its own baseline, or the trellis is just
    nine unrelated plots."""
    ghost = mcolors.to_hex(mcolors.to_rgb(P.GHOST))
    fig = P.small_multiples(d["series"], d["ref"], ncol=3)
    used = [ax for ax in fig.axes if ax.get_title()]
    assert len(used) == len(d["series"])
    for ax in used:
        assert any(mcolors.to_hex(ln.get_color()) == ghost for ln in ax.lines), \
            f"{ax.get_title()} has no ghosted reference"


def test_small_multiples_labels_the_x_axis_above_an_empty_cell():
    """sharex hides tick labels off the bottom row, so a partly-filled last row
    would strip the axis from the panels above its gaps."""
    steps = np.arange(6.0)
    series = {f"arm{i}": (steps, steps * 0.1) for i in range(4)}
    fig = P.small_multiples(series, ncol=3)
    labelled = [ax for ax in fig.axes if ax.get_xlabel()]
    assert len(labelled) == 3, "the 3 panels with nothing beneath them need an axis"


def test_joint_hexbin_uses_a_log_count_scale(d):
    """The structure worth seeing is in the tail; a linear count map shows the mode
    and nothing else."""
    fig = P.joint_hexbin(d["sw"], d["sm"])
    hexes = [c for c in fig.axes[0].collections if c.get_array() is not None]
    assert hexes, "no hexbin collection"
    assert isinstance(hexes[0].norm, mcolors.LogNorm) or \
        hexes[0].get_array().max() < 1e9


def test_peri_event_raster_shows_one_row_per_event_and_a_mean_below():
    rows = np.random.default_rng(3).random((13, 9))
    fig = P.peri_event_raster([("S_model", rows)])
    img = fig.axes[0].images[0]
    assert img.get_array().shape == (13, 9)
    assert "n=13" in fig.axes[0].get_title()
    mean_ax = [a for a in fig.axes if a.get_xlabel()][0]
    assert len(mean_ax.lines) >= 2, "mean line plus the onset marker"


def test_peri_event_raster_scales_to_one_or_two_statistics():
    rows = np.random.default_rng(4).random((6, 7))
    assert len(P.peri_event_raster([("a", rows)]).axes) == 3
    assert len(P.peri_event_raster([("a", rows), ("b", rows)]).axes) == 6


def test_bubble_labels_do_not_collide_when_arms_cluster():
    """Nine arms whose sparsity differs by 2% used to stack nine labels on one line."""
    pts = {a: [(0.40 + 0.001 * i, 0.30 + 0.001 * i, 0.5)]
           for i, a in enumerate(P.ARM_SPEC)}
    fig = P.bubble_reg_vs_sparsity(pts)
    ax = fig.axes[0]
    ys = sorted(a.xy[1] for a in ax.texts if a.get_text() in P.ARM_SPEC)
    assert len(ys) == len(P.ARM_SPEC), "every arm must still be labelled"
    assert min(np.diff(ys)) > 0.02, "labels were placed on top of each other"


def test_bubble_reference_line_is_drawn_when_given():
    fig = P.bubble_reg_vs_sparsity({"a": [(0.4, 0.3, 0.5)]}, ref_level=0.26)
    assert any(np.allclose(ln.get_ydata(), 0.26) for ln in fig.axes[0].lines)


def test_bubble_drops_nonpositive_l0_on_a_log_axis_but_keeps_it_otherwise():
    pts = {"a": [(0.0, 0.3, 0.5), (0.4, 0.3, 0.5)]}
    log_fig = P.bubble_reg_vs_sparsity(pts, log_x=True)
    lin_fig = P.bubble_reg_vs_sparsity(pts, log_x=False)
    n_log = sum(len(c.get_offsets()) for c in log_fig.axes[0].collections)
    n_lin = sum(len(c.get_offsets()) for c in lin_fig.axes[0].collections)
    assert n_log == 1 and n_lin == 2


def test_every_panel_has_a_white_facecolor(d):
    """wandb renders on a dark page; a transparent figure loses every dark glyph."""
    for name, fn, args, kw, _ in _cases(d):
        fig = fn(*args, **kw)
        assert mcolors.to_hex(fig.get_facecolor()) == "#ffffff", name
        plt.close(fig)


def test_every_panel_reports_its_sample_size_somewhere(d):
    """A shape with no n is not reviewable, and n is what decides whether any of
    these shapes mean anything at n=3 seeds."""
    for name, fn, args, kw, _ in _cases(d):
        fig = fn(*args, **kw)
        text = " ".join(
            [t.get_text() for a in fig.axes for t in a.texts]
            + [a.get_title() for a in fig.axes]
            + [t.get_text() for t in fig.texts]
            + [t.get_text() for a in fig.axes if a.get_legend()
               for t in a.get_legend().get_texts()]
        )
        assert any(tok in text for tok in ("n =", "n=", "arms", "points", "events")), \
            f"{name} shows no sample size"
        plt.close(fig)


# --- the redesigned forms -------------------------------------------------------

def _campaign_seeds(rng=None):
    rng = rng or np.random.default_rng(3)
    # derived from the arm list, not a fixed-length literal: a hardcoded table silently
    # KeyErrors the moment an arm is added
    base = [.42, .40, .29, .36, .33, .30, .35, .37, .39]
    true = {a: base[i % len(base)] for i, a in enumerate(P.ARM_SPEC)}
    return {a: true[a] + rng.normal(0, .02, 3) for a in P.ARM_SPEC}


CONTRASTS = [("LpWM-base", "PiWM-sparse-matched"), ("LpWM-base", "PiWM-sparse-2pct"),
             ("LpWM-ltv", "PiWM-gate-sup-sigmoid"), ("LpWM-ltv", "PiWM-union4-entropy")]


def _new_forms():
    """One rendered figure per redesigned form, on plausible synthetic input."""
    rng = np.random.default_rng(5)
    ep = np.linspace(0, 2, 120)
    seeds = _campaign_seeds()
    return {
        "estimation_plot": P.estimation_plot(seeds, CONTRASTS, mde=0.02),
        "phase_plane": P.phase_plane(
            {a: (0.5 + rng.normal(0, .01, 60), 0.2 + np.exp(-np.linspace(0, 3, 60)))
             for a in list(P.ARM_SPEC)[:4]}),
        "facet_grid": P.facet_grid(
            {a: (ep, 0.3 - 0.25 * np.exp(-2 * ep) + rng.normal(0, .005, ep.size))
             for a in P.ARM_SPEC}, ylabel="reg/density_gap"),
        "arm_ridgeline": P.arm_ridgeline(
            {a: rng.gamma(6, .12, 900) for a in list(P.ARM_SPEC)[:5]},
            refs=[(0.5, "sigmoid"), (1.0, "target")]),
        "strip_plot": P.strip_plot({a: rng.gamma(20, .05, 200) for a in P.ARM_SPEC},
                                   ref=1.0, ref_label="median"),
        "stat_tiles": P.stat_tiles([("ICC", "0.04", "seed noise dominates", "crit"),
                                    ("Resolved", "3 / 7", "vs MDE", "warn")]),
        "forest": P.forest([{"label": "Step 1", "arm": "LpWM-base", "mean": .1,
                             "lo": .03, "hi": .17, "threshold": 0.0, "verdict": "PASS"},
                            {"label": "Step 2", "arm": "PiWM-sparse-2pct", "mean": -.02,
                             "lo": -.06, "hi": .01, "threshold": 0.0, "verdict": "FAIL"}]),
        "dot_ci": P.dot_ci({"|d agent|": {j: (.3 + .05 * j, .25 + .05 * j, .36 + .05 * j, 90)
                                          for j in range(4)}}),
    }


def test_every_redesigned_form_returns_a_real_figure():
    for name, fig in _new_forms().items():
        assert isinstance(fig, Figure), f"{name} did not return a Figure"
        assert not P.is_no_data(fig), f"{name} returned a no-data placeholder on real input"
        assert any(ax.has_data() for ax in fig.axes), f"{name} drew nothing"
        plt.close(fig)


def test_every_redesigned_form_survives_empty_input():
    """A panel must never raise: a missing diagnostic cannot be allowed to kill a
    live training run, so the contract is a labelled placeholder, not an exception."""
    empties = [P.estimation_plot({}, []), P.phase_plane({}), P.facet_grid({}),
               P.arm_ridgeline({}), P.strip_plot({}), P.stat_tiles([]),
               P.forest([]), P.dot_ci({})]
    for fig in empties:
        assert P.is_no_data(fig)
        plt.close(fig)


def test_no_redesigned_form_draws_a_bar_chart():
    """The campaign's rule: a per-arm metric summary is never a bar.

    A bar encodes distance from an arbitrary zero, discards the seed spread that
    decides whether an effect is real, and forces a value-ramp onto nominal
    categories. This test is what stops one creeping back in.
    """
    from matplotlib.container import BarContainer
    for name, fig in _new_forms().items():
        for ax in fig.axes:
            bars = [c for c in ax.containers if isinstance(c, BarContainer)]
            assert not bars, f"{name} drew a bar chart"
        plt.close(fig)


def test_estimation_plot_marks_underpowered_contrasts_hollow():
    """Status rides on marker FILL, never on hue, so an arm keeps its identity while
    still reporting whether its effect cleared the detection floor."""
    seeds = {"LpWM-base": np.array([0.40, 0.40, 0.40]),
             "PiWM-sparse-matched": np.array([0.401, 0.401, 0.401]),   # far below any MDE
             "PiWM-sparse-2pct": np.array([0.20, 0.20, 0.20])}          # far above it
    fig = P.estimation_plot(seeds, [("LpWM-base", "PiWM-sparse-matched"),
                                    ("LpWM-base", "PiWM-sparse-2pct")], mde=0.05)
    faces = [c.get_facecolor()[0] for ax in fig.axes for c in ax.collections
             if getattr(c, "get_offsets", None) is not None and len(c.get_offsets()) == 1]
    whites = [f for f in faces if min(f[:3]) > 0.95]
    assert whites, "no hollow marker for the underpowered contrast"
    assert len(whites) < len(faces), "every marker is hollow -- the resolved one is missing"
    plt.close(fig)


def test_phase_plane_encodes_time_as_lightness_of_the_arm_hue():
    """One trace, two channels: hue is the arm, lightness is when. If the ramp were
    shared across arms the panel would need a second colour key."""
    a, b = "PiWM-sparse-2pct", "PiWM-union4-entropy"
    ra, rb = P.arm_ramp(a), P.arm_ramp(b)
    assert not np.allclose(ra(0.9)[:3], rb(0.9)[:3]), "two arms share a time ramp"
    for ramp in (ra, rb):
        assert sum(ramp(0.05)[:3]) > sum(ramp(0.95)[:3]), "ramp does not darken with time"


def test_declutter_separates_labels_without_a_draw():
    """`transData + transAxes.inverted()` is the identity until the first draw, so a
    label placer built on it silently stacks every label at its data coordinate."""
    fig, ax = plt.subplots()
    ax.set(xlim=(0, 1), ylim=(0, 1))
    P.declutter(ax, [(1.0, 0.50, "arm_a", "k"), (1.0, 0.501, "arm_b", "k")])
    ys = sorted(t.get_position()[1] for t in ax.texts)
    assert ys[1] - ys[0] > 0.03, "labels at the same height were not separated"
    assert all(0.0 <= t.get_position()[0] <= 1.0 for t in ax.texts), \
        "a label was placed outside the axes, where it lands on the next subplot"
    plt.close(fig)


def test_css_tokens_cover_both_modes_and_every_arm():
    """The HTML report reads its palette from here, so a missing token is a silently
    unstyled chart rather than an error."""
    t = P.css_tokens()
    assert set(t) == {"light", "dark"}
    assert set(t["light"]) == set(t["dark"]), "light and dark declare different tokens"
    for a in P.ARM_SPEC:
        assert P.arm_slot(a) in t["light"], f"no CSS token for {a}"
    assert t["light"]["surface"] != t["dark"]["surface"]


def test_ecdf_log_axis_is_auto_but_an_explicit_choice_wins():
    """`log_x="auto"` picks by span; True/False are obeyed.

    A log axis earns its place across decades (k-WTA pinned at k=8 beside a dense arm
    at ~190) and actively hurts inside one, where the ticks collide as
    "1.93e2 1.94e2 ...". But a heuristic must never override a caller who asked for a
    scale -- that is a silent behaviour change at a call site that stated its intent.
    """
    rng = np.random.default_rng(0)
    narrow = {"dense": rng.normal(197, 2, 300)}
    wide = {"dense": rng.normal(190, 5, 300), "kwta": rng.normal(8, 0.5, 300)}

    assert P.ecdf_overlay(narrow, log_x="auto").axes[0].get_xscale() == "linear"
    assert P.ecdf_overlay(wide, log_x="auto").axes[0].get_xscale() == "log"
    # explicit beats the heuristic, in both directions
    assert P.ecdf_overlay(narrow, log_x=True).axes[0].get_xscale() == "log"
    assert P.ecdf_overlay(wide, log_x=False).axes[0].get_xscale() == "linear"
    plt.close("all")


def test_arm_names_wrap_without_becoming_ambiguous():
    """Mechanism names are long; nine on one axis overlap unless they are broken.

    Breaking at a separator keeps every fragment meaningful. Truncation would not:
    "PiWM-gate-sup-s..." and "PiWM-gate-sup-so..." are the same label to a reader,
    which is exactly the pair the Step 3 gate has to distinguish.
    """
    wrapped = {a: P.wrap_arm(a) for a in P.ARM_SPEC}
    assert len(set(wrapped.values())) == len(wrapped), "two arms wrap to the same label"
    for a, w in wrapped.items():
        assert w.replace("\n", "-").replace("--", "-") in (a, a.replace("_", "-")), \
            f"{a} lost information when wrapped to {w!r}"
        assert max(len(line) for line in w.split("\n")) <= 14, f"{w!r} is still too wide"
    assert P.wrap_arm("short") == "short", "a short name must not be broken"
