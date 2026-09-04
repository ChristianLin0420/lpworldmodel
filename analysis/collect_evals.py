"""Turn the planner's scattered per-run logs into the campaign.json the figures read.

There was no step between the two. `plan.py` writes one `plan_outputs/<stamp>_<run>_gH<h>/
logs.json` per RUN, and `analysis/figures.py` wants one `campaign.json` holding
`{"arms": {arm: {seed: success}}}` plus the pre-registered gates -- so the estimation
plot, the gate scorecard and the power curve had no way to be rendered from a real
campaign at all. This is that step.

Two things it is careful about, both of which silently corrupt every contrast if got
wrong:

  * `logs.json` is JSON-LINES, not a JSON document. Each MPC re-planning iteration
    appends an object, so `json.load` raises and a naive reader that catches the
    exception ends up with nothing.
  * the file carries BOTH `mpc/success_rate` (the planner improving WITHIN an
    episode, one row per iteration) and `final_eval/success_rate` (the number the
    gates are registered on). They differ, and the mpc rows are the last ones
    written -- so "read the last line" picks the wrong metric.

Usage:
    python analysis/collect_evals.py --out campaign.json
    python analysis/collect_evals.py --plan-outputs plan_outputs --out campaign.json
    python analysis/figures.py --campaign campaign.json --runs 'runs/export/*' --out figures/
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis import figures as FG  # noqa: E402
from analysis import panels as P  # noqa: E402

#: plan_outputs dir name: "<YYYYmmddHHMMSS>_<RUN_NAME>_gH<goal_H>"
_DIR = re.compile(r"^(?P<stamp>\d{8,})_(?P<run>.+)_gH(?P<gh>\d+)$")

#: The metric the gates are defined on. NOT "mpc/success_rate" -- see module docstring.
SUCCESS_KEY = "final_eval/success_rate"

#: plan.py:134 used to build eval_seed as [seed*n + 1 for n in range(n_evals)], which
#: DEGENERATES at seed 0 to [1]*n_evals -- all 50 "episodes" were one initial condition,
#: so a seed-0 number measured planner stochasticity on a single task instance rather
#: than a 50-episode success rate. It also made the episode SETS differ per seed
#: ([1..50] for seed 1, [1,3,5,...] for seed 2), so eval noise was not common-mode and
#: did not cancel in a paired difference. Fixed in 49a3e55 to disjoint blocks.
#:
#: Evals from before that commit are therefore on a DIFFERENT measurement instrument
#: and must not be pooled with later ones.
#:
#: GROUND TRUTH is the `eval_seed: [...]` line every planning job prints, recovered
#: from slurm_logs/: post-fix runs start at seed*n_evals+1 (s3 -> 151, s10 -> 501),
#: pre-fix runs always start at 1. Two weaker rules were tried and both misclassify:
#:   * the <YYYYmmddHHMMSS> in the plan_outputs dir name is written in a different
#:     timezone from git and stat (a dir stamped 151403 has mtime 22:14:04, a 7h
#:     offset), so comparing it to a commit time marks EVERY run as pre-fix;
#:   * logs.json mtime marks 6 runs wrong -- sparse-matched s0-2 and LeWM-ltv s0-2
#:     STARTED before the fix and FINISHED after it, so they wrote a post-fix mtime
#:     while actually running the degenerate eval_seed.
#: mtime is kept only as a fallback for runs whose slurm log has been rotated away.
EVAL_SCHEME_FIX_EPOCH = 1788275140.0   # 49a3e55, 2026-09-01 16:05:40 local
_SLURM_TRUTH = None


def _slurm_scheme_table(slurm_logs="slurm_logs"):
    """{run_name: 'fixed'|'buggy'} from the eval_seed line each job prints."""
    global _SLURM_TRUTH
    if _SLURM_TRUTH is not None:
        return _SLURM_TRUTH
    table = {}
    for f in Path(slurm_logs).glob("eval_*.out"):
        run = re.sub(r"_\d+\.out$", "", f.name)[len("eval_"):]
        try:
            seed = int(run.rsplit("_s", 1)[-1])
        except ValueError:
            continue
        try:
            line = next(l for l in f.open() if l.startswith("eval_seed:"))
            seeds = json.loads(line.split(":", 1)[1].strip())
        except Exception:
            continue
        # a run whose 2nd episode seed is seed*50+2 used the disjoint-block scheme
        if len(seeds) > 1:
            table[run] = "fixed" if seeds[1] == seed * len(seeds) + 2 else "buggy"
    _SLURM_TRUTH = table
    return table


def eval_scheme(run, log_path):
    """'fixed' if this eval ran on the repaired eval_seed, else 'buggy'."""
    hit = _slurm_scheme_table().get(run)
    if hit is not None:
        return hit
    try:
        return "fixed" if Path(log_path).stat().st_mtime >= EVAL_SCHEME_FIX_EPOCH else "buggy"
    except OSError:
        return "buggy"


def read_logs(path):
    """Every record in a JSON-lines log, skipping partial trailing writes.

    A run still in flight can have a half-written last line; dropping it is right,
    because the alternative is aborting on a file that is otherwise complete.
    """
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # partial trailing write from a live job
    return out


def final_success(records, key=SUCCESS_KEY):
    """The final-eval success rate, or None if this run has not reached it yet."""
    for rec in reversed(records):          # last write wins if it was logged twice
        if key in rec:
            return float(rec[key])
    return None


def collect(plan_outputs="plan_outputs", key=SUCCESS_KEY, scheme="fixed"):
    """{arm: {seed: success}} plus the per-run detail, newest eval per run winning.

    `scheme` selects the eval instrument: "fixed" (default, the only one valid for a
    paired test), "buggy" (historical, seed 0 degenerate), or "all" (do NOT use for
    inference -- it pools two different measurement instruments).
    """
    arms, detail, pending = {}, {}, []
    for d in sorted(Path(plan_outputs).glob("*")):
        m = _DIR.match(d.name)
        if not (d.is_dir() and m):
            continue
        run = m.group("run")
        log = d / "logs.json"
        if not log.exists():
            pending.append(run)
            continue
        sch = eval_scheme(run, log)
        if scheme != "all" and sch != scheme:
            continue
        recs = read_logs(log)
        s = final_success(recs, key)
        if s is None:
            pending.append(run)
            continue
        arm, seed = FG.run_arm(run), FG.run_seed(run)
        if seed is None:
            continue
        # sorted() puts the newest timestamp last, so a re-run overwrites an older one
        arms.setdefault(arm, {})[str(seed)] = s
        detail[run] = {"success": s, "goal_H": int(m.group("gh")), "dir": str(d),
                       "n_records": len(recs), "eval_scheme": sch}
    return arms, detail, sorted(set(pending))


class ArmNameError(KeyError):
    """An arm name that does not resolve. Deliberately an exception, not a None."""


def resolve_arm(arms, name):
    """Map a bare arm name to its key in `arms`, accounting for the feature tag.

    A run is named `${arm}_pd${D}${ftag}_${prec}_s${seed}` where ftag is `_patch` for a
    patch-feature arm and empty for `cls`. collect() strips the pd/precision/seed parts but
    NOT the feature tag, so the patch arms are keyed "PiWM-patchdecode_patch" while every
    cls arm is keyed by its bare name.

    THIS HAS BITTEN THREE TIMES, always silently, because the natural spelling is
    `arms.get("PiWM-patchdecode", {})` and an empty dict is indistinguishable from an arm
    that has not been evaluated yet:

      * wave23_autopilot.sh reported the T2 contrast as n=0 with n=5 and n=8 on disk.
      * r6_watch.sh did the same AND gated its ROUND5-COMPLETE marker on every pair
        reaching n>=8, so the marker became unreachable.
      * analysis modules taking a caller-supplied arm name have the same exposure.

    So this raises rather than returning None: an unresolvable name is a bug in the caller,
    never "no data". The trailing underscore in the prefix test matters -- without it
    "PiWM-vp" would swallow "PiWM-vp-mc".
    """
    if name in arms:
        return name
    cand = [k for k in arms if k.startswith(name + "_")]
    if len(cand) == 1:
        return cand[0]
    if not cand:
        raise ArmNameError(f"no arm matches {name!r} (have {len(arms)} arms)")
    raise ArmNameError(f"{name!r} is ambiguous: {sorted(cand)}")


def arm_seeds(arms, name, default=None):
    """resolve_arm + lookup, with an explicit opt-in to 'missing is empty'."""
    try:
        return arms[resolve_arm(arms, name)]
    except ArmNameError:
        if default is None:
            raise
        return default


def build_gates(arms, groups=None, threshold=0.0):
    """Pre-registered gates as paired effects, reusing figures.py's own statistics.

    Deliberately NOT a fresh implementation: `paired_effect` drops seeds present in
    only one arm rather than mean-imputing them, and uses the t critical value for
    n-1 df. Re-deriving that here would eventually disagree with the panels.
    """
    gates = []
    for gate, members in FG.group_arms(arms, groups).items():
        ctrl = members[0]
        for arm in members[1:]:
            e = FG.paired_effect(arms, ctrl, arm)
            if not e["n"]:
                continue
            gates.append({
                "name": f"{gate}: {arm} vs {ctrl}",
                "arm": arm,
                "observed": e["mean"],
                "lo": e["lo"] if np.isfinite(e["lo"]) else e["mean"],
                "hi": e["hi"] if np.isfinite(e["hi"]) else e["mean"],
                "threshold": threshold,
                "direction": "above",
                "n": e["n"],
            })
    return gates


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan-outputs", default="plan_outputs")
    ap.add_argument("--out", default="campaign.json")
    ap.add_argument("--key", default=SUCCESS_KEY,
                    help="metric the gates are defined on (default: %(default)s)")
    ap.add_argument("--threshold", type=float, default=0.0,
                    help="pre-registered gate threshold on the paired effect")
    ap.add_argument("--scheme", default="fixed", choices=("fixed", "buggy", "all"),
                    help="which eval instrument to report (default: %(default)s). "
                         "'all' pools pre- and post-49a3e55 evals and is not valid "
                         "for inference.")
    a = ap.parse_args(argv)

    arms, detail, pending = collect(a.plan_outputs, a.key, a.scheme)
    if a.scheme == "all":
        print("WARNING --scheme=all pools two different eval instruments "
              "(see EVAL_SCHEME_FIX); paired tests over it are not valid.\n")
    if not arms:
        print(f"no completed evals under {a.plan_outputs}/ "
              f"(looked for '{a.key}' in each logs.json)")
        if pending:
            print(f"  {len(pending)} eval(s) started but not finished: "
                  + ", ".join(pending[:6]) + (" ..." if len(pending) > 6 else ""))
        return 1

    gates = build_gates(arms)
    payload = {"arms": arms, "gates": gates}
    Path(a.out).write_text(json.dumps(payload, indent=2))

    print(f"{'arm':<26} {'n':>2}  seeds -> success")
    for arm in sorted(arms):
        v = arms[arm]
        marker = "  (control)" if P.is_control(arm) else ""
        print(f"  {arm:<24} {len(v):>2}  "
              + ", ".join(f"s{k}={v[k]:.3f}" for k in sorted(v)) + marker)
    if gates:
        print(f"\n{'contrast':<44} {'effect':>8} {'95% CI':>18} {'n':>2}")
        for g in gates:
            print(f"  {g['name'][:42]:<42} {g['observed']:>+8.4f} "
                  f"[{g['lo']:>+7.4f},{g['hi']:>+7.4f}] {g['n']:>2}")
    if pending:
        print(f"\n{len(pending)} eval(s) still running: "
              + ", ".join(pending[:6]) + (" ..." if len(pending) > 6 else ""))
    print(f"\nwrote {a.out}  ({len(arms)} arms, {sum(map(len, arms.values()))} runs, "
          f"{len(gates)} contrasts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
