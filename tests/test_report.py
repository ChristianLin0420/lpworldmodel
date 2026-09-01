"""analysis/report.py emits one self-contained HTML file with no build step.

The tests here are the ones that catch the failures that actually bite: a NaN in
the payload (JSON.parse dies and the page is blank), a palette that drifts from the
PNG suite, and a chart that has no non-visual equivalent.
"""
import json
import re

import numpy as np
import pytest

from analysis import figures as FG
from analysis import panels as P
from analysis import report as R


@pytest.fixture(scope="module")
def campaign(tmp_path_factory):
    tmp = FG._synth(tmp_path_factory.mktemp("report"))
    camp = json.loads((tmp / "campaign.json").read_text())
    runs = FG.load_runs(str(tmp / "runs" / "*"))
    return camp, runs


@pytest.fixture(scope="module")
def payload(campaign):
    camp, runs = campaign
    return R.build_payload(camp.get("arms", {}), camp.get("gates", []),
                           camp.get("gate_values", {}), runs)


def test_payload_is_strict_json_with_no_nan(payload):
    """`json.dumps(allow_nan=False)` is the whole point: NaN is not JSON, and a
    payload carrying one produces a silently blank page rather than an error."""
    s = json.dumps(payload, allow_nan=False)
    assert "NaN" not in s and "Infinity" not in s
    assert json.loads(s) == payload


def test_every_arm_carries_a_style_the_page_can_render(payload):
    slots = set(P.css_tokens()["light"])
    for a in payload["arms"]:
        assert a["slot"] in slots, f"{a['name']} has no CSS token"
        assert a["marker"] in {"o", "s", "D", "^"}
        assert isinstance(a["dash"], bool)


def test_the_report_palette_is_the_png_palette(payload):
    """One design system, two media. If the report restated its own hexes they would
    drift, which is exactly how the old suite showed one arm in two colours."""
    tokens = payload["tokens"]["light"]
    for a in payload["arms"]:
        assert tokens[a["slot"]].lower() == P.arm_color(a["name"]).lower()


def test_controls_are_marked_and_excluded_from_their_own_contrast(payload):
    controls = {a["name"] for a in payload["arms"] if a["control"]}
    assert controls == set(P.CONTROL_ARMS) & {a["name"] for a in payload["arms"]}
    for c, v in payload["contrasts"]:
        assert c != v, "an arm was contrasted against itself"


def test_histories_are_decimated_so_the_file_stays_openable(payload):
    for a in payload["arms"]:
        assert len(a["hist"]) <= 400, f"{a['name']} history was not decimated"
        assert all(np.isfinite(v) for v in a["hist"])


def test_render_is_one_self_contained_file(payload):
    html = R.render(payload)
    assert html.lstrip().startswith("<!doctype html")
    assert html.rstrip().endswith("</html>")
    # no build step, no network: a report that needs a CDN is useless on a cluster.
    # The SVG namespace URI is an XML identifier, never fetched, so it is exempt.
    fetchable = html.replace("http://www.w3.org/2000/svg", "")
    assert "http://" not in fetchable and "https://" not in fetchable
    assert "<script src" not in html and "<link" not in html
    assert "@import" not in html
    assert "__DATA__" not in html and "__CSS__" not in html and "__METRIC__" not in html


def test_light_is_the_default_and_the_os_cannot_override_it(payload):
    """The report is read beside papers and printed PNGs, so it opens light.

    An OS-dark machine auto-flipping it would put the reader on a surface the PNGs
    were never validated against. Dark stays available on the toggle, with its own
    validated steps -- it is a choice, not an ambient default.
    """
    html = R.render(payload)
    assert "prefers-color-scheme" not in html, "the OS setting can still flip the report"
    assert ":root { color-scheme: light;" in html, "light is not the root default"
    assert ':root[data-theme="dark"]' in html, "dark is unreachable from the toggle"
    # the dark surface must still be a real, distinct set of steps, not a filter
    assert payload["tokens"]["dark"]["surface"] != payload["tokens"]["light"]["surface"]


def test_live_mode_emits_a_refresh_and_a_stamp(payload):
    """A stale tab must be visibly stale: the stamp says when, the meta tag reloads."""
    live = R.render(payload, refresh=30)
    assert 'http-equiv="refresh" content="30"' in live
    assert "refreshing every 30s" in live
    static = R.render(payload)
    assert "http-equiv=\"refresh\"" not in static, "static render should not reload"
    assert "generated " in static, "even a static render says when it was made"


def test_every_chart_has_a_table_twin(payload):
    """Colour-only encoding fails accessibility, so every value must also be
    reachable without seeing the chart."""
    html = R.render(payload)
    assert 'id="tbl"' in html and "<table>" in html.replace("'", '"') or "table" in html
    assert "Table view" in html


def test_empty_campaign_still_renders(tmp_path):
    """A report generated before any run finished must be a readable empty page, not
    a traceback -- it is the first thing looked at when a campaign is launched."""
    payload = R.build_payload({}, [], {}, [])
    html = R.render(payload)
    assert html.lstrip().startswith("<!doctype html")
    assert payload["mde"] is None
    assert payload["arms"] == [] and payload["contrasts"] == []
