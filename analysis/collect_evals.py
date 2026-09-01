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
_DIR = re.compile(r"^\d{8,}_(?P<run>.+)_gH(?P<gh>\d+)$")

#: The metric the gates are defined on. NOT "mpc/success_rate" -- see module docstring.
SUCCESS_KEY = "final_eval/success_rate"


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


def collect(plan_outputs="plan_outputs", key=SUCCESS_KEY):
    """{arm: {seed: success}} plus the per-run detail, newest eval per run winning."""
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
                       "n_records": len(recs)}
    return arms, detail, sorted(set(pending))


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
    a = ap.parse_args(argv)

    arms, detail, pending = collect(a.plan_outputs, a.key)
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
