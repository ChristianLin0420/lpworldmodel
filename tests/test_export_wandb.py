"""The wandb history exporter, tested without touching the network.

The exporter is what unblocks every run-level panel, so its parsing and matching
logic is tested against the exact shapes the live `PiWM-pushT` project
returns: histogram payloads mixed into scalar rows, all rows pinned inside epoch 1,
and run names of the form '<arm>/s<seed>'. A fake run object stands in for
wandb.Api so the suite stays offline.
"""
import csv
import json

import numpy as np
import pytest

from analysis import export_wandb as E


# --- scalars and histograms -----------------------------------------------------

def test_is_scalar_rejects_the_payloads_that_share_a_history_row():
    assert E.is_scalar(1) and E.is_scalar(0.5) and E.is_scalar(-3)
    assert not E.is_scalar(None)
    assert not E.is_scalar("running")
    assert not E.is_scalar({"_type": "histogram", "values": [1]})
    assert not E.is_scalar([1, 2])
    assert not E.is_scalar(True), "a bool in a metric column is a logging bug, not 1"


def test_decode_histogram_handles_the_packed_bins_wandb_actually_sends():
    v = {"_type": "histogram",
         "packedBins": {"min": 2.0, "size": 0.5, "count": 3},
         "values": [1, 2, 3]}
    h = E.decode_histogram(v)
    assert h["edges"] == [2.0, 2.5, 3.0, 3.5]
    assert h["counts"] == [1.0, 2.0, 3.0]


def test_decode_histogram_handles_explicit_bin_edges():
    h = E.decode_histogram({"_type": "histogram", "bins": [0, 1, 3],
                            "values": [4, 5]})
    assert h["edges"] == [0.0, 1.0, 3.0] and h["counts"] == [4.0, 5.0]


def test_decode_histogram_returns_none_for_anything_else():
    assert E.decode_histogram(0.5) is None
    assert E.decode_histogram({"_type": "image-file", "path": "x.png"}) is None
    assert E.decode_histogram({"_type": "histogram", "values": []}) is None
    assert E.decode_histogram({"_type": "histogram", "values": [1]}) is None


def test_collect_hists_subsamples_and_drops_gradient_watch_payloads():
    def hist(i):
        return {"_type": "histogram",
                "packedBins": {"min": 0.0, "size": 1.0, "count": 2},
                "values": [i, i + 1]}
    rows = [{"_step": i, "dist/z_l0_per_sample": hist(i),
             "gradients/module.embed.0.weight": hist(i), "train/loss": 0.1}
            for i in range(100)]
    got = E.collect_hists(rows, max_per_key=10)
    assert set(got) == {"dist/z_l0_per_sample"}, "wandb.watch payloads are dropped"
    assert len(got["dist/z_l0_per_sample"]) == 10
    steps = [h["step"] for h in got["dist/z_l0_per_sample"]]
    assert steps[0] == 0 and steps[-1] == 99, "the subsample must span the run"


# --- epoch bookkeeping ----------------------------------------------------------

def test_batches_per_epoch_recovers_the_real_campaign_value():
    """train.py logs step=(epoch-1)*N+i and epoch_frac=epoch-1+i/N, so step/frac
    is exactly N. The live campaign's value is 30,965."""
    rows = [{"_step": 0, "progress/epoch_frac": 0.0}]
    rows += [{"_step": s, "progress/epoch_frac": s / 30965.0}
             for s in range(50, 5000, 50)]
    assert E.batches_per_epoch(rows) == pytest.approx(30965.0)


def test_batches_per_epoch_is_none_without_progress_logging():
    assert E.batches_per_epoch([{"_step": 10, "train/loss": 0.2}]) is None
    assert E.batches_per_epoch([]) is None


# --- resume detection -----------------------------------------------------------

REAL_LOG = """\
[2026-08-30 23:26:06,593][__main__][WARNING] - Received signal 15; will checkpoint
[2026-08-30 23:29:37,075][__main__][INFO] - Resuming Wandb run 32td3p4t
[2026-08-30 23:29:47,684][__main__][INFO] - Resuming from epoch 1, batch 291: /x
[2026-08-31 00:19:34,650][__main__][WARNING] - Exiting after preemption checkpoint
[2026-08-31 00:25:00,000][__main__][INFO] - Resuming from epoch 2, batch 100: /x
"""


def test_resumes_from_log_reads_the_exact_marker_train_py_writes():
    marks = E.resumes_from_log(REAL_LOG, n_batches=30965)
    assert [(m["epoch"], m["batch"]) for m in marks] == [(1, 291), (2, 100)]
    assert marks[0]["epoch_frac"] == pytest.approx(291 / 30965.0)
    assert marks[1]["epoch_frac"] == pytest.approx(1 + 100 / 30965.0)
    assert marks[1]["step"] == 30965 + 100
    assert all(m["source"] == "train.log" for m in marks)


def test_resumes_from_log_is_quiet_on_a_run_that_never_restarted():
    assert E.resumes_from_log("no restarts here", 100) == []
    assert E.resumes_from_log("", None) == []


def test_resumes_from_log_without_batches_per_epoch_leaves_step_unset():
    m = E.resumes_from_log(REAL_LOG)[0]
    assert m["step"] is None and m["epoch"] == 1


def test_resumes_from_gaps_flags_a_queue_wait_and_not_a_slow_run():
    """The gap has to be both absolutely large and large relative to this run's own
    cadence: the absolute floor alone flags a slow run, the ratio alone flags a
    fast one's first rows."""
    rows = [{"_timestamp": 1000.0 + 40 * i, "_step": 50 * i,
             "progress/epoch_frac": i / 100.0, "epoch": 1} for i in range(20)]
    for r in rows[10:]:  # a 90 minute requeue
        r["_timestamp"] += 5400.0
    marks = E.resumes_from_gaps(rows)
    assert len(marks) == 1
    assert marks[0]["step"] == 500
    assert marks[0]["wall_gap_s"] == pytest.approx(5440.0)
    assert marks[0]["source"] == "timestamp_gap"

    steady = [{"_timestamp": 1000.0 + 600 * i, "_step": 50 * i} for i in range(20)]
    assert E.resumes_from_gaps(steady) == [], "a uniformly slow run is not a resume"
    assert E.resumes_from_gaps(rows[:2]) == []


def test_marker_units_collapses_markers_into_plottable_lists():
    marks = [{"epoch": 1, "epoch_frac": 0.5, "step": None},
             {"epoch": 2, "epoch_frac": 1.25, "step": 40000}]
    u = E.marker_units(marks)
    assert u["epoch"] == [1.0, 2.0]
    assert u["epoch_frac"] == [0.5, 1.25]
    assert u["step"] == [40000.0], "a None must not become a marker at 0"
    assert E.marker_units([]) == {}


# --- run matching ---------------------------------------------------------------

class FakeRun:
    """The subset of wandb's public Run that the exporter touches."""

    def __init__(self, id="abc123", name="PiWM-union4-entropy/s0", group="PiWM-union4-entropy",
                 state="running", rows=None, config=None, fail=False):
        self.id, self.name, self.group, self.state = id, name, group, state
        self.job_type = "step4-union"
        self.tags = ["J:4+ent0.1", "prec:bf16"]
        self.created_at = "2026-08-31T08:23:18Z"
        self.url = f"https://wandb.ai/e/p/runs/{id}"
        self.config = config or {"embed_dim": 384, "n_heads": 4, "kwta_k": None}
        self._rows = rows or []
        self._fail = fail

    def scan_history(self, page_size=1000, min_step=None):
        if self._fail:
            raise RuntimeError("network is down")
        return [r for r in self._rows
                if min_step is None or r.get("_step", 0) >= min_step]


def _history(n=20, heads=True, hist_every=8):
    rows = []
    for i in range(n):
        step = 50 * i
        r = {"_step": step, "_timestamp": 1.788e9 + 40 * i, "_runtime": 40.0 * i,
             "epoch": 1, "progress/epoch_frac": step / 30965.0,
             "train/loss": 0.5 - 0.01 * i, "sparsity/train_l0_frac": 0.5,
             "opt/lr": 5e-4}
        if heads:
            r["heads/train_head_usage_p0"] = 0.3
        if i % hist_every == 0:
            r["dist/z_l0_per_sample"] = {
                "_type": "histogram",
                "packedBins": {"min": 100.0, "size": 1.0, "count": 4},
                "values": [1, 2, 3, 4]}
        rows.append(r)
    return rows


def test_seed_of_parses_both_naming_conventions():
    assert E.seed_of("PiWM-union4-entropy/s0") == 0
    assert E.seed_of("LpWM-base_pd384_bf16_s2") == 2
    assert E.seed_of("probe") is None
    assert E.seed_of(None) is None


def test_local_run_dirs_indexes_by_the_id_train_py_persists(tmp_path):
    (tmp_path / "arm_s0").mkdir()
    (tmp_path / "arm_s0" / "hydra.yaml").write_text(
        "wandb_project: PiWM-pushT\nwandb_run_id: diwh6bdy\n")
    (tmp_path / "arm_s1").mkdir()
    (tmp_path / "arm_s1" / "hydra.yaml").write_text("wandb_run_id: null\n")
    (tmp_path / "no_cfg").mkdir()
    ids = E.local_run_dirs(tmp_path)
    assert set(ids) == {"diwh6bdy"}
    assert ids["diwh6bdy"].name == "arm_s0"


def test_match_dir_prefers_the_run_id_then_group_and_seed(tmp_path):
    d = tmp_path / "PiWM-union4-entropy_pd384_bf16_s0"
    d.mkdir()
    (d / "hydra.yaml").write_text("wandb_run_id: abc123\n")
    by_id = E.local_run_dirs(tmp_path)

    got, how = E.match_dir(FakeRun(id="abc123"), tmp_path, by_id)
    assert got == d and how == "wandb_run_id"

    got, how = E.match_dir(FakeRun(id="other"), tmp_path, by_id)
    assert got == d and how == "group+seed", "a missing hydra.yaml still matches"

    got, how = E.match_dir(FakeRun(id="x", name="brand_new/s9", group="brand_new"),
                           tmp_path, by_id)
    assert got.name == "brand_new_s9" and how == "synthesised"


# --- csv round trip -------------------------------------------------------------

def test_rows_to_csv_keeps_scalars_leads_with_step_and_writes_atomically(tmp_path):
    rows = _history(4)
    p = tmp_path / "wandb_history.csv"
    cols = E.rows_to_csv(rows, p)
    assert cols[:5] == ["_step", "_runtime", "_timestamp", "epoch",
                        "progress/epoch_frac"]
    assert "dist/z_l0_per_sample" not in cols, "histograms never enter the CSV"
    assert not (tmp_path / "wandb_history.csv.tmp").exists()
    with open(p, newline="") as f:
        got = list(csv.DictReader(f))
    assert len(got) == 4
    assert float(got[0]["train/loss"]) == pytest.approx(0.5)


def test_rows_to_csv_blanks_a_cell_a_step_never_logged(tmp_path):
    p = tmp_path / "h.csv"
    E.rows_to_csv([{"_step": 0, "a": 1.0}, {"_step": 1, "b": 2.0}], p)
    got = list(csv.DictReader(open(p, newline="")))
    assert got[0]["b"] == "" and got[1]["a"] == ""


def test_read_existing_csv_round_trips_by_step(tmp_path):
    p = tmp_path / "h.csv"
    E.rows_to_csv(_history(5), p)
    prev = E.read_existing_csv(p)
    assert sorted(prev) == [0, 50, 100, 150, 200]
    assert prev[100]["train/loss"] == pytest.approx(0.48)
    assert E.read_existing_csv(tmp_path / "nope.csv") == {}


# --- end to end, offline --------------------------------------------------------

def test_export_run_writes_every_artefact_the_panels_read(tmp_path):
    run = FakeRun(rows=_history(24))
    info = E.export_run(run, tmp_path, {})
    d = tmp_path / "PiWM-union4-entropy_s0"
    assert info["status"] == "ok" and info["n_rows"] == 24
    assert (d / "wandb_history.csv").exists()
    assert (d / "resume_steps.json").exists()

    meta = json.loads((d / "wandb_meta.json").read_text())
    assert meta["group"] == "PiWM-union4-entropy" and meta["seed"] == 0
    assert meta["batches_per_epoch"] == pytest.approx(30965.0)
    assert meta["config"]["n_heads"] == 4
    assert meta["state"] == "running"

    hists = json.loads((d / "wandb_hists.json").read_text())
    assert len(hists["dist/z_l0_per_sample"]) == 3


def test_exported_run_is_readable_by_the_figure_suite(tmp_path):
    """The whole point of the exporter: figures.py must be able to plot the result."""
    from analysis import figures as Fg
    E.export_run(FakeRun(rows=_history(30)), tmp_path, {})
    runs = Fg.load_runs(str(tmp_path / "*"))
    assert len(runs) == 1
    r = runs[0]
    assert r["arm"] == "PiWM-union4-entropy" and r["seed"] == 0
    assert "train_loss" in r["hist"] and "train_l0_frac" in r["hist"]
    assert Fg.x_axis(r["hist"])[2] == "epoch_frac"
    assert Fg.fig_training_curves(runs, tmp_path / "figs") is not None
    assert Fg.fig_l0_distribution(runs, tmp_path / "figs") is not None


def test_export_run_prefers_the_train_log_marker_over_the_timestamp_gap(tmp_path):
    d = tmp_path / "PiWM-union4-entropy_pd384_bf16_s0"
    d.mkdir()
    (d / "hydra.yaml").write_text("wandb_run_id: abc123\n")
    (d / "train.log").write_text(REAL_LOG)
    E.export_run(FakeRun(rows=_history(20)), tmp_path,
                 E.local_run_dirs(tmp_path))
    marks = json.loads((d / "resume_steps.json").read_text())
    assert [m["source"] for m in marks["markers"]] == ["train.log", "train.log"]
    assert marks["epoch"] == [1.0, 2.0]
    assert marks["step"] == [291.0, 31065.0]


def test_export_run_incremental_only_scans_new_steps_and_keeps_old_histograms(tmp_path):
    """A partial campaign is topped up repeatedly; histograms live only in the
    scanned rows, so an incremental pass must not lose the earlier snapshots."""
    run = FakeRun(rows=_history(10))
    E.export_run(run, tmp_path, {})
    d = tmp_path / "PiWM-union4-entropy_s0"
    first = json.loads((d / "wandb_hists.json").read_text())
    assert len(first["dist/z_l0_per_sample"]) == 2

    run._rows = _history(30)
    info = E.export_run(run, tmp_path, {}, incremental=True)
    assert info["n_rows"] == 30
    assert info["n_new_rows"] == 20, "already-exported steps are not re-scanned"
    merged = json.loads((d / "wandb_hists.json").read_text())
    steps = [h["step"] for h in merged["dist/z_l0_per_sample"]]
    assert steps == sorted(steps) and 0.0 in steps and 1200.0 in steps


def test_export_run_reports_an_api_failure_instead_of_raising(tmp_path):
    info = E.export_run(FakeRun(fail=True), tmp_path, {})
    assert info["status"] == "error" and "network is down" in info["error"]
    assert not (tmp_path / "PiWM-union4-entropy_s0" / "wandb_history.csv").exists()


def test_export_run_reports_a_run_that_has_logged_nothing(tmp_path):
    info = E.export_run(FakeRun(rows=[]), tmp_path, {})
    assert info["status"] == "empty" and info["n_rows"] == 0


def test_export_run_dry_run_touches_nothing(tmp_path):
    info = E.export_run(FakeRun(rows=_history(5)), tmp_path, {}, dry_run=True)
    assert info["status"] == "dry_run"
    assert info["matched_by"] == "synthesised"
    assert list(tmp_path.iterdir()) == []


def test_scan_rows_retries_then_gives_up(tmp_path):
    calls = {"n": 0}

    class Flaky(FakeRun):
        def scan_history(self, page_size=1000, min_step=None):
            calls["n"] += 1
            if calls["n"] < 2:
                raise RuntimeError("transient")
            return _history(3)

    rows = E.scan_rows(Flaky(), retries=3)
    assert len(rows) == 3 and calls["n"] == 2

    calls["n"] = -10
    with pytest.raises(RuntimeError):
        E.scan_rows(Flaky(), retries=2)


def test_scan_rows_sorts_by_step_and_honours_max_rows():
    run = FakeRun(rows=list(reversed(_history(10))))
    rows = E.scan_rows(run)
    assert [r["_step"] for r in rows] == sorted(r["_step"] for r in rows)
    assert len(E.scan_rows(run, max_rows=4)) == 4


def test_defaults_point_at_the_live_campaign_project():
    """The blocker was that nobody could remember where the histories live."""
    assert E.DEFAULT_ENTITY == "crlc112358"
    assert E.DEFAULT_PROJECT == "PiWM-pushT"
    assert "n_heads" in E.META_CONFIG_KEYS and "kwta_k" in E.META_CONFIG_KEYS


def test_batches_per_epoch_ignores_a_nonfinite_ratio():
    rows = [{"_step": 0, "progress/epoch_frac": 0.0},
            {"_step": np.nan, "progress/epoch_frac": 0.5},
            {"_step": 100, "progress/epoch_frac": 0.01}]
    assert E.batches_per_epoch(rows) == pytest.approx(10000.0)
