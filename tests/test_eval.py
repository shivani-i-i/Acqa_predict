import numpy as np
import pandas as pd
import pytest
from aquapredict.schema import STAGES, VISIBLE_COLS
from aquapredict.simulator.generate import generate_dataset
from aquapredict.baselines.baselines import BASELINES, volume_rule
from aquapredict.eval.config import EvalConfig
from aquapredict.eval.harness import evaluate
from aquapredict.eval.splits import split_households, assert_disjoint, subset
from aquapredict.eval import metrics as M
from aquapredict.eval.criteria import check_success
from aquapredict.planner.cost import policy_table, sensitivity

CFG = EvalConfig()


@pytest.fixture(scope="module")
def data():
    df, _ = generate_dataset(24, mix="normal", seed=21)
    return df


def make_oracle(df, width=5.0, bias=0.0, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    truth = {(r.household_id, r.day): {s: getattr(r, f"true_dtc_{s}") for s in STAGES}
             for r in df[["household_id", "day"] + [f"true_dtc_{s}" for s in STAGES]].itertuples()}

    def f(hist, stage):
        t = truth[(hist.household_id.iloc[-1], int(hist.day.iloc[-1]))][stage] + bias + rng.normal(0, noise)
        return {"point": t, "lo": max(0.0, t - width), "hi": t + width, "withheld": False}
    return f


def test_split_is_disjoint_and_by_household(data):
    sp = split_households(data.household_id.unique(), seed=1)
    assert_disjoint(sp)
    assert sum(len(v) for v in sp.values()) == data.household_id.nunique()
    with pytest.raises(AssertionError):
        assert_disjoint({"a": ["x"], "b": ["x"]})
    tr, te = subset(data, sp["train"]), subset(data, sp["test"])
    assert not set(tr.household_id) & set(te.household_id)


def test_forecaster_sees_only_past_visible_columns(data):
    seen = []

    def spy(hist, stage):
        assert list(hist.columns) == VISIBLE_COLS                # no truth columns
        assert len(hist) == int(hist.day.iloc[-1]) + 1            # history ends at the forecast day, nothing later
        assert (hist.day.to_numpy() == np.arange(len(hist))).all()
        seen.append(len(hist))
        return {"point": 10.0}
    evaluate(data[data.household_id == data.household_id.iloc[0]], {"spy": spy}, cfg=CFG)
    assert len(seen) > 10


def test_perfect_oracle_has_zero_error_full_coverage_no_late(data):
    res = evaluate(data, {"oracle": make_oracle(data)}, cfg=CFG)
    w = M.window_summary(res.opps, CFG)
    assert w.loc["oracle", "median_abs_err_primary"] == 0
    iv = M.interval_summary(res.opps, CFG)
    assert iv.loc["oracle", "coverage"] == 1.0
    assert iv.loc["oracle", "median_width_primary"] <= 10.0
    lf = M.late_false_summary(M.cycle_outcomes(res.opps, CFG))
    assert lf.loc["oracle", "late_rate"] == 0 and lf.loc["oracle", "premature_rate"] == 0


def test_cycle_outcome_definitions_by_hand():
    rows = []
    def cyc(k, crossing, alert_day):
        for d in range(0, crossing + 1):
            av = 10.0 if (alert_day is not None and d >= alert_day) else 99.0
            rows.append({"method": "m", "household_id": "H", "scenario": "s", "stage": "sediment", "cycle": k,
                         "cycle_start": 0, "crossing": crossing, "day": d, "true_dtc": crossing - d,
                         "point": av, "lo": np.nan, "hi": np.nan, "withheld": False, "alert_value": av})
    cyc(0, 100, 85)    # lead 15 -> ok
    cyc(1, 100, 98)    # lead 2  -> late (< 7)
    cyc(2, 100, 40)    # lead 60 -> premature (> 45)
    cyc(3, 100, None)  # never alerted -> late
    out = M.cycle_outcomes(pd.DataFrame(rows), CFG).sort_values("cycle")
    assert out.late.tolist() == [False, True, False, True]
    assert out.premature.tolist() == [False, False, True, False]
    assert out.lead.iloc[0] == 15


def test_withheld_never_alerts_and_is_not_scored():
    o = pd.DataFrame([{"method": "m", "household_id": "H", "scenario": "s", "stage": "ro", "cycle": 0, "cycle_start": 0,
                       "crossing": 50, "day": d, "true_dtc": 50 - d, "point": 5.0, "lo": 0.0, "hi": 9.0,
                       "withheld": True, "alert_value": np.nan} for d in range(0, 51, 5)])
    assert M.issued(o).empty
    assert M.cycle_outcomes(o, CFG).late.all()


def test_interval_coverage_and_width_formulas():
    o = pd.DataFrame({"method": "m", "household_id": "H", "scenario": "s", "stage": "ro", "cycle": 0,
                      "cycle_start": 0, "crossing": 100, "day": [0, 1, 2, 3], "true_dtc": [10.0, 20.0, 30.0, 40.0],
                      "point": 25.0, "lo": [5.0, 25.0, 25.0, 25.0], "hi": [15.0, 35.0, 35.0, 35.0],
                      "withheld": False, "alert_value": 25.0})
    iv = M.interval_summary(o, CFG)
    assert iv.loc["m", "coverage"] == pytest.approx(0.5)      # covers 10 and 30 only
    assert iv.loc["m", "median_width_all"] == 10.0


def test_baselines_return_finite_nonnegative_points(data):
    res = evaluate(data, BASELINES, cfg=CFG)
    assert set(res.opps.method) == set(BASELINES)
    assert np.isfinite(res.opps.point).all() and (res.opps.point >= 0).all()
    assert len(res.opps) > 500


def test_trend_beats_volume_rule_on_sediment_normal_households(data):
    res = evaluate(data, BASELINES, stages=("sediment",), cfg=CFG)
    w = M.window_summary(res.opps, CFG)
    assert w.loc["trend", "median_abs_err_primary"] < w.loc["volume_rule", "median_abs_err_primary"]


def test_insufficient_history_gives_no_opportunities():
    df, _ = generate_dataset(3, mix="insufficient_history", seed=1)
    assert evaluate(df, BASELINES, cfg=CFG).opps.empty


def test_short_lives_are_excluded_for_every_method(data):
    res = evaluate(data, {"oracle": make_oracle(data)}, cfg=CFG)
    assert res.n_short_excluded >= 0
    cyc = M.cycle_outcomes(res.opps, CFG)
    assert (cyc.crossing - cyc.cycle_start >= CFG.min_age_days + CFG.min_notice_days).all()


def test_success_criteria_logic_with_good_and_bad_models(data):
    good = make_oracle(data, width=8.2, noise=5.0)        # ~90% interval for sigma=5
    res = evaluate(data, {**BASELINES, "AquaPredict": good}, cfg=CFG)
    ok = check_success(res.opps, CFG)
    assert ok["1_window_error"]["passed"] and ok["2_interval_coverage"]["passed"] and ok["3_late_warnings"]["passed"]
    assert ok["all_passed"]
    bad = make_oracle(data, width=1.0, bias=40.0)          # overconfident and biased
    res = evaluate(data, {**BASELINES, "AquaPredict": bad}, cfg=CFG)
    assert not check_success(res.opps, CFG)["all_passed"]
    wide = make_oracle(data, width=60.0, noise=5.0)         # 100% coverage but very wide -> must not pass criterion 2
    res = evaluate(data, {**BASELINES, "AquaPredict": wide}, cfg=CFG)
    assert not check_success(res.opps, CFG)["2_interval_coverage"]["passed"]
    with pytest.raises(KeyError):
        check_success(res.opps, CFG, model="missing")


def test_cost_oracle_has_no_overdue_and_lowest_cost(data):
    res = evaluate(data, BASELINES, cfg=CFG)
    t = policy_table(M.cycle_outcomes(res.opps, CFG))
    assert t.loc["oracle", "overdue_days_per_cycle"] == 0
    assert t.loc["oracle", "annual_cost_per_purifier"] == t.annual_cost_per_purifier.min()
    s = sensitivity(M.cycle_outcomes(res.opps, CFG))
    assert {"vs_volume_rule", "annual_cost"} <= set(s.columns) and len(s) > 10


def test_volume_rule_unit_example():
    hist = pd.DataFrame({"usage_L": [40.0] * 11, "age_sediment_d": np.arange(11.0),
                         "cum_vol_sediment_L": np.cumsum([40.0] * 11)})
    # rated 6000 L, used 440 L, 40 L/day -> (6000-440)/40 = 139 days
    assert volume_rule(hist, "sediment")["point"] == pytest.approx(139.0)
