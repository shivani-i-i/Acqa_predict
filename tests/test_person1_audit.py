import json
import numpy as np
import pandas as pd
import pytest
from dataclasses import fields
from aquapredict.schema import STAGES
from aquapredict.simulator.config import SimConfig
from aquapredict.simulator.generate import generate_dataset
from aquapredict.simulator.scenarios import SCENARIOS, DEFAULT_MIX
from aquapredict.simulator.sanity import sanity_report
from aquapredict.validate import validate_telemetry, assert_valid
from aquapredict.baselines.baselines import BASELINES
from aquapredict.eval.config import EvalConfig
from aquapredict.eval.criteria import check_success
from aquapredict.eval.harness import evaluate
from aquapredict.eval.stress import run_stress_suite
from aquapredict.eval.report import render_markdown, CAVEAT
from aquapredict.eval.run_baselines import run as run_baselines
from aquapredict.export import export_scenario, export_results, save_table

CFG = EvalConfig()


# ---------------- regression / reproducibility ----------------
def test_default_dataset_unchanged_golden_statistics():
    """Guards the data behind every reported result. If this fails the simulator DEFAULT changed."""
    df, _ = generate_dataset(20, seed=1)
    assert len(df) == 19010
    assert df.usage_L.mean() == pytest.approx(45.7032, abs=1e-3)
    assert df.dp_sediment_bar.mean() == pytest.approx(0.179004, abs=1e-5)
    assert int((df.true_dtc_sediment == 0).sum()) == 223


def test_every_config_field_is_documented():
    doc = open("docs/SIMULATOR.md").read()
    missing = [f.name for f in fields(SimConfig) if f"`{f.name}`" not in doc]
    assert not missing, f"undocumented SimConfig fields: {missing}"


def test_scenarios_default_mix_and_extra_scenario():
    assert len(SCENARIOS) == 11 and "late_replacement" in SCENARIOS
    assert set(DEFAULT_MIX) <= set(SCENARIOS) and "late_replacement" not in DEFAULT_MIX
    assert abs(sum(DEFAULT_MIX.values()) - 1.0) < 1e-9


# ---------------- late replacement ----------------
def test_late_replacement_labels_and_delay():
    df, _ = generate_dataset(25, mix="late_replacement", seed=5)
    delays, negatives = [], 0
    for _, g in df.groupby("household_id"):
        g = g.reset_index(drop=True)
        for st in STAGES:
            d = g[f"true_dtc_{st}"].to_numpy()
            rep = g[f"replaced_{st}"].to_numpy()
            starts = np.r_[0, np.flatnonzero(rep)[np.flatnonzero(rep) > 0]]
            ends = np.r_[starts[1:], len(g)]
            for s, e in zip(starts, ends):
                seg = d[s:e]
                if np.isnan(seg).all():
                    continue
                assert (seg == 0).sum() == 1                       # exactly one crossing day per cycle
                assert (np.diff(seg) == -1).all()                  # overdue days continue counting down (negative)
                negatives += int((seg < 0).sum())
                if e < len(g):
                    delays.append(e - 1 - (s + int(np.flatnonzero(seg == 0)[0])))
    assert negatives > 0 and 15 < np.mean(delays) < 40            # configured mean 25 days


def test_default_has_no_overdue_rows():
    df, _ = generate_dataset(10, mix="normal", seed=5)
    assert not (df[[f"true_dtc_{s}" for s in STAGES]] < 0).any().any()


def test_overdue_sensors_keep_evolving_past_criterion():
    df, _ = generate_dataset(20, mix="late_replacement", seed=6)
    seen = False
    for _, g in df.groupby("household_id"):
        od = g[g.true_dtc_sediment < 0]
        if len(od):
            assert (od.true_dp_ratio >= 1.8 - 1e-9).all()          # still beyond the criterion while overdue
            seen = True
    assert seen


# ---------------- calibrated preset ----------------
def _median_life(cfg, scen):
    df, _ = generate_dataset(30, mix=scen, seed=11, cfg=cfg)
    life = np.concatenate([g.age_sediment_d.to_numpy()[np.flatnonzero(g.true_dtc_sediment.to_numpy() == 0)]
                           for _, g in df.groupby("household_id")])
    return np.median(life), (life < 21).mean()


def test_calibrated_preset_fixes_pathological_short_lives_only_where_intended():
    m0, s0 = _median_life(None, "gradual_pressure_rise")
    m1, s1 = _median_life(SimConfig.calibrated(), "gradual_pressure_rise")
    assert m1 > 2 * m0 and s1 < s0
    assert SimConfig().partic_ramp_cap > 1e6 and SimConfig.calibrated().partic_ramp_cap == 2.0
    assert SimConfig.calibrated(partic_ramp_cap=1.5).partic_ramp_cap == 1.5


# ---------------- validation ----------------
def test_validate_accepts_generated_and_rejects_broken():
    df, _ = generate_dataset(6, mix="normal", seed=2)
    assert validate_telemetry(df, require_truth=True) == []
    assert "missing columns" in validate_telemetry(df.drop(columns=["flow_Lpm"]))[0]
    assert any("duplicate" in p for p in validate_telemetry(pd.concat([df, df.iloc[:3]])))
    bad = df.copy(); bad.loc[3, "temp_C"] = 500
    assert any("temp_C" in p for p in validate_telemetry(bad))
    bad = df.copy(); bad.loc[bad.index[7], "op_mode"] = "weird"
    assert any("op_mode" in p for p in validate_telemetry(bad))
    gap = df[~((df.household_id == df.household_id.iloc[0]) & (df.day == 10))]
    assert any("contiguous" in p for p in validate_telemetry(gap))
    assert validate_telemetry(df.iloc[0:0]) == ["table is empty"]
    with pytest.raises(ValueError):
        assert_valid(df.drop(columns=["usage_L"]))


def test_harness_refuses_malformed_input_and_generator_rejects_zero():
    df, _ = generate_dataset(3, mix="normal", seed=2)
    with pytest.raises(ValueError):
        evaluate(df.drop(columns=["dp_sediment_bar"]), BASELINES, cfg=CFG)
    with pytest.raises(ValueError):
        generate_dataset(0)


# ---------------- criteria: paired scoring and preset threshold ----------------
def _opps(method, err, withheld, alert=None, n_h=8):
    rows = []
    for h in range(n_h):
        for day in range(40, 100):
            ok = not withheld(day)
            t = 100 - day
            rows.append({"method": method, "household_id": f"h{h}", "scenario": "s", "stage": "sediment", "cycle": 0,
                         "cycle_start": 0, "crossing": 100, "day": day, "true_dtc": float(t),
                         "point": t + err(day) if ok else np.nan, "lo": np.nan, "hi": np.nan,
                         "withheld": not ok, "fallback": False,
                         "alert_value": (alert(day) if alert else (t + err(day))) if ok else np.nan})
    return pd.DataFrame(rows)


def test_cherry_picking_by_withholding_does_not_win_criterion_1():
    base_err = lambda d: 1.0 if d % 2 == 0 else 20.0          # baseline: accurate on even days, bad on odd days
    parts = [_opps(b, base_err, lambda d: False) for b in ("volume_rule", "health_20pct", "trend")]
    model = _opps("M", lambda d: 5.0, lambda d: d % 2 == 1)    # model: mediocre (5) but only answers the easy days
    res = check_success(pd.concat(parts + [model], ignore_index=True), CFG, model="M")
    assert res["1_window_error"]["baselines_unpaired"]["trend"] > 5          # unpaired scoring would have flattered it
    assert res["1_window_error"]["baselines"]["trend"] == pytest.approx(1.0)  # paired: baseline wins on the same days
    assert not res["1_window_error"]["passed"]


def test_criterion_3_uses_preset_threshold_not_best_threshold():
    vol = _opps("volume_rule", lambda d: 0.0, lambda d: False, alert=lambda d: 99.0 if d < 95 else 5.0)   # alerts only at day 95: late
    others = [_opps(b, lambda d: 0.0, lambda d: False, alert=lambda d: 99.0 if d < 95 else 5.0) for b in ("health_20pct", "trend")]
    eager = _opps("M", lambda d: 0.0, lambda d: False, alert=lambda d: 10.0)       # alerts on the very first check: never late, always premature
    res = check_success(pd.concat([vol] + others + [eager], ignore_index=True), CFG, model="M")["3_late_warnings"]
    assert res["model_late"] == 0 and res["model_premature"] == 1.0 and res["volume_rule_late"] == 1.0
    assert not res["passed"]                                   # much better late rate, but unacceptable false-alarm rate


# ---------------- previously untested modules ----------------
def test_stress_suite_runs_and_volume_rule_ignores_missing_data():
    st = run_stress_suite(BASELINES, n_households=6, seed=3, only=["baseline_lawA", "missing_50pct"])
    assert set(st.variant) == {"baseline_lawA", "missing_50pct"}
    assert {"median_abs_err_primary", "late_rate", "premature_rate", "coverage"} <= set(st.columns)
    v = st[st.method == "volume_rule"].set_index("variant").median_abs_err_primary
    assert v["baseline_lawA"] == pytest.approx(v["missing_50pct"])


def test_report_contains_definitions_caveat_and_exclusions():
    df, _ = generate_dataset(10, mix="normal", seed=4)
    res = evaluate(df, BASELINES, cfg=CFG)
    md = render_markdown(res, None, model_version="x")
    assert CAVEAT in md and "Definitions" in md and "excluded as too short" in md and "volume_rule" in md


def test_exports(tmp_path):
    p = export_scenario(SCENARIOS["stuck_sensor"], str(tmp_path / "s.json"))
    d = json.load(open(p))
    assert d["name"] == "stuck_sensor" and d["data_source"] == "synthetic" and d["fault_templates"]
    out = export_results({"t": pd.DataFrame({"a": [1, 2]}).rename_axis("i")}, str(tmp_path / "r"))
    assert "SYNTHETIC" in (tmp_path / "r" / "NOTICE.txt").read_text() and pd.read_csv(out["t"]).shape[0] == 2
    assert save_table(pd.DataFrame({"a": [1]}), str(tmp_path / "x" / "t")).endswith((".csv", ".parquet"))


def test_run_baselines_smoke_and_calibrated_flag(tmp_path):
    res, tables, success = run_baselines(n=24, seed=0, out=str(tmp_path), verbose=False, cfg_sim=SimConfig.calibrated())
    assert {"window_summary", "late_false_summary", "cost_policies", "operating_curve"} <= set(tables)
    assert (tmp_path / "report.md").exists() and success is None
    assert tables["window_summary"].loc["trend", "median_abs_err_primary"] < tables["window_summary"].loc["volume_rule", "median_abs_err_primary"]


def test_sanity_report_structure():
    df, meta = generate_dataset(30, seed=3, cfg=SimConfig.calibrated())
    r = sanity_report(df, meta)
    assert r["households"] == 30 and set(r["stages"]) == set(STAGES)
    assert abs(sum(r["sediment_cycle_share_by_scenario"].values()) - 1.0) < 0.01
    assert r["stages"]["sediment"]["completed_cycles"] > 20


def test_evaluate_batched_matches_evaluate_and_reports_progress(tmp_path):
    from aquapredict.eval.harness import evaluate_batched
    df, _ = generate_dataset(7, mix="normal", seed=12)
    full = evaluate(df, BASELINES, cfg=CFG).opps
    calls = []
    res = evaluate_batched(df, BASELINES, cfg=CFG, batch_households=3, progress=lambda d, t, m: calls.append((d, t)),
                           checkpoint=str(tmp_path / "ck.pkl"))
    key = ["method", "household_id", "stage", "cycle", "day"]
    a = full.sort_values(key).reset_index(drop=True)[key + ["point"]]
    b = res.opps.sort_values(key).reset_index(drop=True)[key + ["point"]]
    pd.testing.assert_frame_equal(a, b)
    assert calls[-1] == (7, 7) and len(calls) == 3 and (tmp_path / "ck.pkl").exists()
    assert res.n_households == 7
