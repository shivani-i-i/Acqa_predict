import numpy as np
import pandas as pd
from aquapredict.eval import metrics as M
from aquapredict.eval.multiseed import run_multiseed
from aquapredict.eval import figures as F
from aquapredict.simulator.generate import generate_household


def test_bootstrap_ci_brackets_estimate_and_resamples_households():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"method": "m", "household_id": np.repeat([f"h{i}" for i in range(30)], 20),
                       "x": rng.normal(10, 2, 600)})
    t = M.household_bootstrap_ci(df, "x", ("method",), n_boot=200, stat="mean")
    r = t.iloc[0]
    assert r.ci_lo <= r.estimate <= r.ci_hi and r.n_households == 30 and r.n_rows == 600
    assert abs(r.estimate - 10) < 1 and (r.ci_hi - r.ci_lo) < 2


def test_bootstrap_ci_wider_with_fewer_households():
    rng = np.random.default_rng(1)
    def ci(nh):
        df = pd.DataFrame({"method": "m", "household_id": np.repeat(range(nh), 10), "x": rng.normal(0, 1, nh * 10)})
        r = M.household_bootstrap_ci(df, "x", ("method",), n_boot=300).iloc[0]
        return r.ci_hi - r.ci_lo
    assert ci(8) > ci(80)


def test_fallback_summary_counts_trend_fallbacks():
    o = pd.DataFrame({"method": ["trend"] * 4, "stage": ["carbon"] * 2 + ["ro"] * 2, "fallback": [True, True, True, False]})
    s = M.fallback_summary(o)["fallback_rate"]
    assert s[("trend", "carbon")] == 1.0 and s[("trend", "ro")] == 0.5


def test_multiseed_pools_unique_households_and_returns_cis():
    opps, cyc, tb, excl = run_multiseed(seeds=(0, 1), n=30, n_boot=50)
    assert opps.household_id.str.startswith("s").all()
    assert opps.groupby("household_id").size().shape[0] > 10
    for k in ("median_abs_err_ci", "late_rate_ci", "premature_rate_ci", "late_rate_by_scenario_ci", "fallback"):
        assert len(tb[k]) > 0
    e = tb["median_abs_err_ci"].set_index("method")
    assert (e.ci_lo <= e.estimate).all() and (e.estimate <= e.ci_hi).all()


def test_figures_are_written(tmp_path):
    df, _ = generate_household(1, "source_change", seed=1)
    p = F.fig_household(df, str(tmp_path / "h.png"))
    assert (tmp_path / "h.png").stat().st_size > 10_000 and p.endswith("h.png")
