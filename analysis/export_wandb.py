"""Pull wandb run histories down to disk so analysis/figures.py can read them offline.

Why this exists: `figures.py::load_runs()` reads `runs/outputs/<run>/wandb_history.csv`,
and until this script nothing in the repo ever wrote one. Four panels (head usage,
engagement, training curves, training health) could therefore never render on real
runs. Everything here is read-only against wandb.

What it writes per run, into `<out_root>/<run_dir>/`:

  wandb_history.csv   every scalar column of the full history, one row per logged
                      step, sorted by _step, blanks where a key was not logged at
                      that step. Column names are the sectioned wandb keys
                      ('train/loss', 'sparsity/train_l0_frac'); figures.py
                      un-sections them.
  wandb_meta.json     id / name / group / job_type / state / tags / created_at,
                      the campaign-relevant config, row and column counts, and
                      batches_per_epoch recovered from _step vs progress/epoch_frac
                      (the panels need it to convert batch indices into epochs).
  wandb_hists.json    decoded wandb.Histogram columns (dist/*), subsampled. These
                      carry the per-sample L0 distribution, which no scalar does.
  resume_steps.json   preemption/resume boundaries in three units
                      {"epoch": [...], "epoch_frac": [...], "step": [...]} plus the
                      provenance of each marker. Primary source is the run dir's
                      train.log ("Resuming from epoch E, batch B"), which is exact;
                      the fallback is a wall-clock gap in the wandb timestamps.

Run matching. A wandb run is tied to its local run dir by the `wandb_run_id` that
train.py persists into `hydra.yaml`, which is exact even after renames. Failing
that we match `<group>_*_s<seed>` against the directory names, and failing that we
create `<out_root>/<group>_s<seed>`.

Usage:
    set -a && . ./.env && set +a
    python analysis/export_wandb.py                      # whole project
    python analysis/export_wandb.py --group PiWM-union4-entropy    # one arm
    python analysis/export_wandb.py --incremental        # only new steps
    python analysis/export_wandb.py --dry_run            # match runs, write nothing
"""
import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

DEFAULT_ENTITY = "crlc112358"
DEFAULT_PROJECT = "PiWM-pushT"

# config keys worth carrying into wandb_meta.json: the campaign knobs the figure
# suite groups and annotates by. The full config is already in hydra.yaml.
META_CONFIG_KEYS = (
    "embed_dim", "precision", "reg_weight", "target_p", "mu", "kwta_k",
    "gate_input", "gate_norm", "n_heads", "head_entropy_coef", "agg",
)

RESUME_RE = re.compile(r"Resuming from epoch (\d+), batch (\d+)")
RUN_ID_RE = re.compile(r"^wandb_run_id:\s*(\S+)\s*$", re.M)


# --- pure helpers (unit tested) -------------------------------------------------

def is_scalar(v):
    """True for values that belong in a CSV cell.

    wandb history rows mix floats with `wandb.Histogram` / `wandb.Image` payloads,
    which arrive as dicts and would otherwise be str()-ed into the CSV and then
    silently become nan in every downstream panel. bool is excluded from the
    numeric path deliberately: `False` in a metric column is a logging bug, not a 0.
    """
    if isinstance(v, bool) or v is None:
        return False
    return isinstance(v, (int, float))


def decode_histogram(v):
    """wandb.Histogram JSON -> {'edges': [...], 'counts': [...]} or None.

    wandb ships histograms in two shapes: `packedBins` (min/size/count, uniform
    width) and an explicit `bins` edge list. Both appear in this project's history,
    so both are handled rather than assumed.
    """
    if not isinstance(v, dict) or v.get("_type") != "histogram":
        return None
    counts = v.get("values")
    if not counts:
        return None
    packed = v.get("packedBins")
    if packed:
        lo, size, n = packed["min"], packed["size"], int(packed["count"])
        edges = [lo + size * i for i in range(n + 1)]
    elif v.get("bins"):
        edges = list(v["bins"])
    else:
        return None
    return {"edges": [float(e) for e in edges], "counts": [float(c) for c in counts]}


def batches_per_epoch(rows):
    """Recover the number of train batches in one epoch from the history.

    train.py logs at `step = (epoch-1)*n_batches + i` and logs
    `progress/epoch_frac = epoch-1 + i/n_batches`, so `step = epoch_frac*n_batches`
    exactly and the ratio is constant. The median over rows shrugs off the row at
    epoch_frac == 0. Returns None when epoch_frac was never logged, which is the
    case for pre-`progress/` exports.
    """
    ratios = []
    for r in rows:
        f, s = r.get("progress/epoch_frac"), r.get("_step")
        if is_scalar(f) and is_scalar(s) and f > 1e-9 and s > 0:
            ratios.append(s / f)
    if not ratios:
        return None
    ratios.sort()
    return float(round(ratios[len(ratios) // 2]))


def resumes_from_log(text, n_batches=None):
    """Parse train.log for exact resume points.

    Each restart inside a 4h window logs 'Resuming from epoch E, batch B' before it
    skips forward, so this is the authoritative marker; the wandb timestamp gap is
    only a proxy for it. Returns a list of marker dicts.
    """
    out = []
    for m in RESUME_RE.finditer(text or ""):
        epoch, batch = int(m.group(1)), int(m.group(2))
        frac = (epoch - 1) + (batch / n_batches if n_batches else 0.0)
        out.append({
            "epoch": epoch,
            "batch": batch,
            "epoch_frac": frac,
            "step": (epoch - 1) * n_batches + batch if n_batches else None,
            "source": "train.log",
        })
    return out


def resumes_from_gaps(rows, min_gap_s=180.0, gap_factor=8.0):
    """Infer resume points from wall-clock holes in the wandb timestamps.

    Consecutive history rows are ~log_every_x_batch apart in time. A requeue leaves
    a hole of at least the queue wait, so a gap that is both absolutely large and
    far above this run's own median gap is a restart. Both conditions are required:
    the absolute floor alone would flag a slow run, the relative factor alone would
    flag the very first rows of a fast one.
    """
    ts = [(r.get("_timestamp"), r.get("_step"), r.get("progress/epoch_frac"),
           r.get("epoch")) for r in rows]
    ts = [t for t in ts if is_scalar(t[0])]
    if len(ts) < 3:
        return []
    gaps = sorted(ts[i + 1][0] - ts[i][0] for i in range(len(ts) - 1))
    med = gaps[len(gaps) // 2]
    thresh = max(min_gap_s, gap_factor * med if med > 0 else min_gap_s)
    out = []
    for i in range(len(ts) - 1):
        gap = ts[i + 1][0] - ts[i][0]
        if gap < thresh:
            continue
        _, step, frac, epoch = ts[i + 1]
        out.append({
            "epoch": epoch if is_scalar(epoch) else None,
            "epoch_frac": frac if is_scalar(frac) else None,
            "step": step if is_scalar(step) else None,
            "wall_gap_s": float(gap),
            "source": "timestamp_gap",
        })
    return out


def marker_units(markers):
    """Collapse marker dicts into the {unit: [x, ...]} form figures.py plots."""
    out = {}
    for unit in ("epoch", "epoch_frac", "step"):
        xs = sorted({round(float(m[unit]), 6) for m in markers
                     if m.get(unit) is not None})
        if xs:
            out[unit] = xs
    return out


def local_run_dirs(out_root):
    """{wandb_run_id: dir} for every run dir that already has a hydra.yaml.

    Reading hydra.yaml with a regex rather than OmegaConf keeps this importable
    without hydra and safe against a config half-written by a live run.
    """
    ids = {}
    for cfg in sorted(Path(out_root).glob("*/hydra.yaml")):
        try:
            m = RUN_ID_RE.search(cfg.read_text(errors="replace"))
        except OSError:
            continue
        if m and m.group(1) not in ("null", "~", "None"):
            ids[m.group(1)] = cfg.parent
    return ids


def seed_of(name):
    """Seed from a wandb run name like 'PiWM-union4-entropy/s0'."""
    m = re.search(r"/s(\d+)$|_s(\d+)$", name or "")
    if not m:
        return None
    return int(m.group(1) or m.group(2))


def match_dir(run, out_root, by_id):
    """Local run dir for a wandb run, most reliable strategy first.

    Returns (dir, how). The id match survives a renamed arm; the group+seed match
    covers a run dir whose hydra.yaml was never written (a job killed during
    init); the synthesised name is the last resort so a run is never dropped.
    """
    out_root = Path(out_root)
    if run.id in by_id:
        return by_id[run.id], "wandb_run_id"
    group, seed = run.group or "", seed_of(run.name)
    if group and seed is not None:
        cands = [d for d in sorted(out_root.glob(f"{group}_*s{seed}")) if d.is_dir()]
        if len(cands) == 1:
            return cands[0], "group+seed"
        if cands:
            return cands[0], "group+seed (ambiguous)"
    stem = (run.name or run.id).replace("/", "_")
    return out_root / stem, "synthesised"


def rows_to_csv(rows, path):
    """Write history rows to CSV atomically, union of scalar columns, _step first.

    Atomic because figures.py may be reading the previous export while a long
    campaign export is running; a half-written CSV would parse as a truncated run.
    """
    cols, seen = [], set()
    for r in rows:
        for k, v in r.items():
            if k not in seen and is_scalar(v):
                seen.add(k)
                cols.append(k)
    lead = [c for c in ("_step", "_runtime", "_timestamp", "epoch",
                        "progress/epoch_frac") if c in seen]
    cols = lead + sorted(c for c in cols if c not in lead)
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow(["" if not is_scalar(r.get(c)) else r[c] for c in cols])
    os.replace(tmp, path)
    return cols


def read_existing_csv(path):
    """Previously exported rows keyed by _step, for --incremental."""
    path = Path(path)
    if not path.exists():
        return {}
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            row = {}
            for k, v in r.items():
                if v in ("", None):
                    continue
                try:
                    row[k] = float(v)
                except ValueError:
                    continue
            if "_step" in row:
                out[int(row["_step"])] = row
    return out


# --- wandb side -----------------------------------------------------------------

def scan_rows(run, page_size=2000, min_step=None, max_rows=None, retries=3):
    """Full history for one run via scan_history, with retries.

    scan_history pages server-side and, unlike run.history(), does not sample or
    truncate -- which matters here because a finished campaign run logs tens of
    thousands of steps and run.history() would silently return 500 of them.
    """
    for attempt in range(retries):
        try:
            it = run.scan_history(page_size=page_size, min_step=min_step)
            rows = []
            for row in it:
                rows.append({k: v for k, v in row.items() if v is not None})
                if max_rows and len(rows) >= max_rows:
                    break
            rows.sort(key=lambda r: r.get("_step", 0))
            return rows
        except Exception as e:  # network/API flake: retry, then give up on this run
            if attempt == retries - 1:
                raise
            print(f"    scan_history failed ({type(e).__name__}: {e}); retrying")
            time.sleep(2 ** attempt)
    return []


def collect_hists(rows, max_per_key=40, skip_prefixes=("gradients/",)):
    """{key: [{'step', 'edges', 'counts'}, ...]} for histogram columns, subsampled.

    Evenly subsampled rather than truncated so the exported set still spans the
    whole run; these become the L0-distribution-over-training panel. `gradients/*`
    is dropped by default: wandb.watch emits one histogram per parameter tensor
    per step, which is ~90% of the payload and nothing here plots it.
    """
    per_key = {}
    for r in rows:
        for k, v in r.items():
            if k.startswith(tuple(skip_prefixes)):
                continue
            h = decode_histogram(v)
            if h is None:
                continue
            h["step"] = r.get("_step")
            per_key.setdefault(k, []).append(h)
    out = {}
    for k, hs in per_key.items():
        if len(hs) > max_per_key:
            idx = [round(i * (len(hs) - 1) / (max_per_key - 1))
                   for i in range(max_per_key)]
            hs = [hs[i] for i in sorted(set(idx))]
        out[k] = hs
    return out


def export_run(run, out_root, by_id, incremental=False, page_size=2000,
               max_rows=None, dry_run=False, min_rows=1):
    """Export one wandb run. Returns a status dict; never raises on a single run."""
    d, how = match_dir(run, out_root, by_id)
    info = {"id": run.id, "name": run.name, "group": run.group, "state": run.state,
            "dir": str(d), "matched_by": how, "n_rows": 0, "status": "ok"}
    if dry_run:
        info["status"] = "dry_run"
        return info

    hist_path = d / "wandb_history.csv"
    prev = read_existing_csv(hist_path) if incremental else {}
    min_step = (max(prev) + 1) if prev else None
    try:
        rows = scan_rows(run, page_size=page_size, min_step=min_step,
                         max_rows=max_rows)
    except Exception as e:
        info.update(status="error", error=f"{type(e).__name__}: {e}")
        return info

    if prev:
        merged = dict(prev)
        for r in rows:
            if is_scalar(r.get("_step")):
                merged[int(r["_step"])] = {**merged.get(int(r["_step"]), {}), **r}
        rows_out = [merged[s] for s in sorted(merged)]
        info["n_new_rows"] = len(rows)
    else:
        rows_out = rows

    if len(rows_out) < min_rows:
        info.update(status="empty", n_rows=len(rows_out))
        return info

    d.mkdir(parents=True, exist_ok=True)
    cols = rows_to_csv(rows_out, hist_path)
    n_batches = batches_per_epoch(rows_out)

    cfg = dict(run.config or {})
    meta = {
        "id": run.id, "name": run.name, "group": run.group,
        "job_type": getattr(run, "job_type", None), "state": run.state,
        "tags": list(run.tags or []), "created_at": str(run.created_at),
        "url": run.url, "seed": seed_of(run.name),
        "n_rows": len(rows_out), "n_columns": len(cols),
        "batches_per_epoch": n_batches,
        "last_step": rows_out[-1].get("_step"),
        "runtime_hours": (rows_out[-1].get("_runtime") or 0) / 3600.0,
        "config": {k: cfg.get(k) for k in META_CONFIG_KEYS if k in cfg},
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (d / "wandb_meta.json").write_text(json.dumps(meta, indent=2, default=str))

    # histograms live only in the scanned rows, never in the CSV, so an incremental
    # pass would otherwise drop every snapshot taken before its min_step
    hists = collect_hists(rows_out)
    hist_path = d / "wandb_hists.json"
    if prev and hist_path.exists():
        try:
            old = json.loads(hist_path.read_text())
        except (OSError, ValueError):
            old = {}
        for k, hs in old.items():
            keep = {h.get("step") for h in hists.get(k, [])}
            hists[k] = sorted(hists.get(k, []) + [h for h in hs
                                                  if h.get("step") not in keep],
                              key=lambda h: h.get("step") or 0)
    if hists:
        hist_path.write_text(json.dumps(hists, default=float))

    log_path = d / "train.log"
    marks = resumes_from_log(
        log_path.read_text(errors="replace") if log_path.exists() else "", n_batches)
    if not marks:
        marks = resumes_from_gaps(rows_out)
    (d / "resume_steps.json").write_text(json.dumps(
        {**marker_units(marks), "markers": marks}, indent=2, default=float))

    info.update(n_rows=len(rows_out), n_columns=len(cols),
                batches_per_epoch=n_batches, n_resumes=len(marks),
                n_hist_keys=len(hists))
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", DEFAULT_ENTITY))
    ap.add_argument("--project", default=os.environ.get("WANDB_PROJECT", DEFAULT_PROJECT))
    ap.add_argument("--out_root", default="runs/outputs")
    ap.add_argument("--group", action="append", default=None,
                    help="restrict to these wandb groups (repeatable)")
    ap.add_argument("--name_re", default=None, help="regex filter on run name")
    ap.add_argument("--state", default=None,
                    help="restrict to runs in this state (running/finished/crashed)")
    ap.add_argument("--incremental", action="store_true",
                    help="only scan steps newer than the existing CSV")
    ap.add_argument("--page_size", type=int, default=2000)
    ap.add_argument("--max_rows", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="cap number of runs")
    ap.add_argument("--dry_run", action="store_true",
                    help="report run->dir matching and exit")
    args = ap.parse_args(argv)

    import wandb  # imported late so --help works without a wandb install

    if not os.environ.get("WANDB_API_KEY"):
        print("warning: WANDB_API_KEY unset; relying on ~/.netrc "
              "(`set -a && . ./.env && set +a` sets it)")
    path = f"{args.entity}/{args.project}"
    api = wandb.Api(timeout=60)
    try:
        runs = list(api.runs(path))
    except Exception as e:
        print(f"could not list {path}: {type(e).__name__}: {e}")
        return 2
    if not runs:
        print(f"no runs in {path} -- check --entity/--project and WANDB_API_KEY")
        return 1

    if args.group:
        runs = [r for r in runs if r.group in set(args.group)]
    if args.name_re:
        rx = re.compile(args.name_re)
        runs = [r for r in runs if rx.search(r.name or "")]
    if args.state:
        runs = [r for r in runs if r.state == args.state]
    runs.sort(key=lambda r: (r.group or "", r.name or ""))
    if args.limit:
        runs = runs[: args.limit]
    if not runs:
        print("no runs left after filtering")
        return 1

    by_id = local_run_dirs(args.out_root)
    print(f"{path}: {len(runs)} runs; {len(by_id)} local dirs carry a wandb_run_id")
    results = []
    for r in runs:
        print(f"  {r.group or '-':<18} {r.name:<22} {r.state:<9}", flush=True)
        info = export_run(r, args.out_root, by_id, incremental=args.incremental,
                          page_size=args.page_size, max_rows=args.max_rows,
                          dry_run=args.dry_run)
        results.append(info)
        extra = f"  ({info['matched_by']})"
        if info["status"] == "ok":
            print(f"    -> {info['n_rows']} rows x {info['n_columns']} cols, "
                  f"{info['n_resumes']} resume marks, "
                  f"batches/epoch={info['batches_per_epoch']} "
                  f"{info['dir']}{extra}")
        else:
            print(f"    -> {info['status']}: {info.get('error', '')} "
                  f"{info['dir']}{extra}")

    ok = [r for r in results if r["status"] == "ok"]
    bad = [r for r in results if r["status"] in ("error", "empty")]
    running = [r for r in ok if r["state"] == "running"]
    print(f"\nexported {len(ok)}/{len(results)} runs "
          f"({sum(r['n_rows'] for r in ok)} rows total)")
    if running:
        print(f"{len(running)} still running -- histories are partial, re-run with "
              "--incremental to top them up")
    for r in bad:
        print(f"FAILED {r['name']}: {r['status']} {r.get('error', '')}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
