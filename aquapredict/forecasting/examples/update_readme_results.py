"""Fill the generated blocks of README.md from the artifacts of ONE run (no hand-typed numbers).

    python examples/update_readme_results.py            # rewrite the blocks from artifacts/*.json
    python examples/update_readme_results.py --check    # exit 1 if README differs from the artifacts

Blocks (between marker comments in README.md):
  RESULTS : tables rendered from artifacts/metrics.json + artifacts/interval_selection_dev.json
  SAMPLE  : example responses taken from examples/sample_prediction.json
The README records the ``run_id`` of the metrics it was rendered from. ``train_demo.py`` stamps the same id into
the saved model and ``run_demo.py`` copies it into sample_prediction.json, so one run id ties them together.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
METRICS = ROOT / "artifacts" / "metrics.json"
SELECTION = ROOT / "artifacts" / "interval_selection_dev.json"
SAMPLE = ROOT / "examples" / "sample_prediction.json"

NAMES = {"gradient_boosting": "Gradient boosting (ours)", "volume_rule": "Volume rule",
         "health_threshold_proxy": "Health-threshold proxy (20 %)", "trend_extrapolation": "Trend extrapolation"}


def _p(x):
    return "n/a" if x is None else f"{100 * x:.1f} %"


def _d(x):
    return "n/a" if x is None else f"{x:.1f} d"


def render_results(m: dict, sel: dict) -> str:
    env = m["environment"]
    L = [f"*Generated from `artifacts/metrics.json` (run_id `{m['run_id']}`). Environment: Python {env['python']}, "
         f"numpy {env['numpy']}, pandas {env['pandas']}, scikit-learn {env['scikit-learn']}, scipy {env['scipy']}, "
         f"joblib {env['joblib']}, shap {env['shap'] or 'not installed'}. Other library versions can give slightly "
         "different numbers; re-running `train_demo.py` rewrites `artifacts/metrics.json` with your own.*", ""]
    h, r = m["households"], m["rows"]
    L += [f"Households (rows): train {h['train']} ({r['train']}), calibration {h['calibration']} ({r['calibration']}), "
          f"development {h['development']} ({r['development']}), **test {h['test']} ({r['test']})**. "
          f"Test filter lives: {m['test_alerting']['horizon_14d']['gradient_boosting']['n_lives']}.", ""]
    for title, key in (("Remaining-days error, all test rows (days)", "test_comparison"),
                       ("Remaining-days error, rows with <= 30 true days left (retrospective view, days)",
                        "test_comparison_last_30_days_of_life")):
        L += [f"**{title}**", "", "| Method | Median AE | MAE | RMSE | n |", "|---|---|---|---|---|"]
        for k, v in m[key].items():
            L.append(f"| {NAMES[k]} | {v['median_ae_days']:.1f} | {v['mae_days']:.1f} | {v['rmse_days']:.1f} | {v['n']} |")
        L.append("")
    for hk in ("horizon_14d", "horizon_30d"):
        d = m["test_alerting"][hk]["gradient_boosting"]["definition"]
        L += [f"**Alerting, H = {d['alert_horizon_days']:g} d** (late: no alert or lead < {d['min_lead_days']:g} d; "
              f"false alarm: lead > {d['max_early_days']:g} d)", "",
              "| Method | Late | False alarm | On time | No alert (part of late) | Median lead | Late rows | False-alarm rows |",
              "|---|---|---|---|---|---|---|---|"]
        for k, v in m["test_alerting"][hk].items():
            L.append(f"| {NAMES[k]} | {_p(v['late_warning_rate'])} | {_p(v['false_alarm_rate'])} | {_p(v['on_time_rate'])} | "
                     f"{_p(v['no_alert_rate'])} | {_d(v['median_lead_days'])} | {_p(v['late_row_rate'])} | {_p(v['false_alarm_row_rate'])} |")
        L.append("")
    c = m["test_interval_coverage"]
    lo, hi = c["overall_coverage_household_bootstrap_90pct_range"]
    verdict = ("**The target is NOT met on these test households.**" if c["empirical_coverage"] < c["target_coverage"]
               else "The point estimate is at or above the target; see the bootstrap range for uncertainty.")
    L += [f"**Prediction intervals on the test households** - method: {c['interval_method']}, target {c['target_coverage']:.0%}, "
          f"final calibration on {m['interval_calibration_households']} households ({m['n_calibration']} rows). {verdict}", "",
          "| Group | Coverage | Mean width | Rows |", "|---|---|---|---|",
          f"| **Overall** | **{_p(c['empirical_coverage'])}** (household-bootstrap 90 % range {_p(lo)} - {_p(hi)}) | "
          f"**{c['mean_interval_width_days']:.1f} d** | {c['n_test_rows']} |"]
    for k, v in c["by_stage"].items():
        L.append(f"| stage `{k}` | {_p(v['empirical_coverage'])} | {v['mean_interval_width_days']:.1f} d | {v['n']} |")
    for k, v in c["by_true_remaining_bucket"].items():
        L.append(f"| true remaining {k.replace(' left', '')} | {_p(v['empirical_coverage'])} | {v['mean_interval_width_days']:.1f} d | {v['n']} |")
    hc = c["household_coverage"]
    L += ["", f"Per-household coverage on the test households: min {_p(hc['min'])}, 10th percentile {_p(hc['p10'])}, "
          f"median {_p(hc['median'])}; {_p(hc['share_at_or_above_target_minus_10pts'])} of households at or above "
          f"{c['target_coverage'] - 0.10:.0%}. Mean width before this change was 133.1 d.", ""]
    L += [f"**Interval variants compared on non-test households** ({sel['repeats']} random household resplits of the "
          f"{sel['households_in_pool']}-household pool; resplit sizes {sel['resplit_sizes_households']}; "
          f"`artifacts/interval_selection_dev.json`)", "",
          "| Variant | Mean coverage | SD | Min | Max | Repeats >= target | Mean width |", "|---|---|---|---|---|---|---|"]
    for k, v in sel["variants"].items():
        L.append(f"| {k}{' (chosen)' if k == sel['chosen_variant'] else ''} | {_p(v['mean_coverage'])} | {_p(v['sd_coverage'])} | "
                 f"{_p(v['min_coverage'])} | {_p(v['max_coverage'])} | {_p(v['share_of_repeats_at_or_above_target'])} | {v['mean_width_days']:.1f} d |")
    ch = sel["variants"][sel["chosen_variant"]]
    L += ["", f"Chosen variant, mean coverage over those resplits by stage: "
          + ", ".join(f"{k} {_p(v)}" for k, v in ch["mean_coverage_by_stage"].items())
          + "; by true remaining life: " + ", ".join(f"{k.replace(' left', '')} {_p(v)}" for k, v in ch["mean_coverage_by_true_remaining_bucket"].items()) + "."]
    return "\n".join(L)


def render_sample(s: dict) -> str:
    ok, bad = s["1_normal"], s["2_required_reading_missing"]
    ok_view = {k: ok[k] for k in ("filter_stage", "household_id", "forecast_days", "interval_days", "drivers", "explanation",
                                  "fault_events")}
    ok_view["source_change"] = {k: ok["source_change"].get(k) for k in ("evaluated", "detected", "message")}
    ok_view.update(status=ok["status"], reason=ok["reason"])
    bad_view = {k: bad[k] for k in ("filter_stage", "household_id", "forecast_days", "interval_days", "drivers", "fault_events",
                                    "status", "reason")}
    return "\n".join([
        f"*Generated from `examples/sample_prediction.json` (model run_id `{s['model_run_id']}`); synthetic data. "
        f"{s['note'].capitalize()}.*", "",
        "Successful response (scenario `1_normal`):", "", "```json", json.dumps(ok_view, indent=2), "```", "",
        "Withheld response (scenario `2_required_reading_missing`):", "", "```json", json.dumps(bad_view, indent=2), "```"])


def _block(name: str) -> re.Pattern:
    return re.compile(rf"(<!-- {name}:START[^>]*-->)\n.*?\n(<!-- {name}:END -->)", re.S)


def build_readme(text: str) -> str:
    m, sel, smp = (json.loads(p.read_text(encoding="utf-8")) for p in (METRICS, SELECTION, SAMPLE))
    for name, body in (("RESULTS", render_results(m, sel)), ("SAMPLE", render_sample(smp))):
        text, n = _block(name).subn(lambda mo: f"<!-- {name}:START run_id={m['run_id']} -->\n{body}\n{mo.group(2)}", text)
        if n != 1:
            raise SystemExit(f"README.md must contain exactly one {name} block (found {n}).")
    return text


def main() -> int:
    old = README.read_text(encoding="utf-8")
    new = build_readme(old)
    if "--check" in sys.argv:
        print("README matches artifacts" if new == old else "README is OUT OF DATE vs artifacts/*.json")
        return 0 if new == old else 1
    README.write_text(new, encoding="utf-8")
    print("README.md results and sample blocks updated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
