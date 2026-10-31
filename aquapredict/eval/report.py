"""Plain-text/markdown evaluation report. Contains only measured numbers + the mandatory caveat."""
from __future__ import annotations
import pandas as pd
from . import metrics as M

CAVEAT = ("All results come from an internally developed simulator. They establish algorithmic feasibility, "
          "NOT real-world accuracy. Prediction intervals are not guarantees. Filter-wear prediction does not "
          "assess water safety.")


def _md(df: pd.DataFrame, nd=3):
    d = df.reset_index()
    head = "| " + " | ".join(map(str, d.columns)) + " |\n|" + "---|" * len(d.columns) + "\n"
    body = ""
    for _, r in d.iterrows():
        body += "| " + " | ".join(f"{v:.{nd}f}" if isinstance(v, float) else str(v) for v in r) + " |\n"
    return head + body


def render_markdown(result, success=None, model_version="n/a", data_note=""):
    o, cfg = result.opps, result.cfg
    cyc = M.cycle_outcomes(o, cfg)
    s = ["# AquaPredict evaluation report", f"*Model version:* {model_version}  ",
         f"*Held-out households:* {result.n_households}  *Forecast opportunities:* {len(o):,}  "
         f"*Filter lives scored:* {cyc.groupby(['household_id','stage','cycle']).ngroups:,} "
         f"(*excluded as too short to forecast with >= {cfg.min_notice_days:.0f} days notice, all methods:* {result.n_short_excluded})", data_note, "",
         "## Definitions (fixed beforehand)", f"```\n{M.__doc__.strip()}\n```", f"Config: `{cfg.to_dict()}`", "",
         "## Window error", _md(M.window_summary(o, cfg)), "## Late / premature warnings",
         _md(M.late_false_summary(cyc)), "## Window error by stage",
         _md(M.window_summary(o, cfg, by=("method", "stage"))), "## Late warnings by scenario",
         _md(M.late_false_summary(cyc, by=("method", "scenario")))]
    iv = M.interval_summary(o, cfg)
    if len(iv):
        s += ["## Prediction intervals", _md(iv)]
    if success:
        s.append("## Success criteria")
        for k, v in success.items():
            if k != "all_passed":
                s.append(f"- **{k}**: {'PASS' if v['passed'] else 'FAIL'} - {v}")
        s.append(f"\n**All criteria met: {success['all_passed']}**")
    s += ["", f"> {CAVEAT}"]
    return "\n".join(s)
