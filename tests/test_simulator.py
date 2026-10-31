import numpy as np
import pandas as pd
import pytest
from aquapredict.schema import STAGES, ALL_COLS, VISIBLE_COLS, TRUTH_COLS, FAULTABLE_SENSORS
from aquapredict.simulator.generate import generate_dataset, generate_household
from aquapredict.simulator.scenarios import SCENARIOS, custom_scenario
from aquapredict.simulator.config import SimConfig
from aquapredict.export import write_dataset


def test_reproducible_same_seed_and_differs_across_seeds():
    a, _ = generate_dataset(6, seed=5)
    b, _ = generate_dataset(6, seed=5)
    c, _ = generate_dataset(6, seed=6)
    assert a.equals(b)
    assert not a.equals(c)


def test_household_independent_of_dataset_size():
    a, _ = generate_dataset(4, seed=2)
    b, _ = generate_dataset(9, seed=2)
    first = b[b.household_id.isin(a.household_id.unique())].reset_index(drop=True)
    assert a.equals(first)


def test_schema_columns_and_visible_truth_disjoint():
    df, _ = generate_dataset(3, mix="normal", seed=1)
    assert list(df.columns) == ALL_COLS
    assert not set(VISIBLE_COLS) & set(TRUTH_COLS)
    assert (df.data_source == "synthetic").all()


@pytest.mark.parametrize("law", ["A", "B"])
def test_truth_labels_consistent(law):
    df, _ = generate_dataset(15, mix="normal", seed=4, law=law)
    n_cross = 0
    for _, g in df.groupby("household_id"):
        g = g.reset_index(drop=True)
        for stage in STAGES:
            dtc = g[f"true_dtc_{stage}"].to_numpy()
            rep = g[f"replaced_{stage}"].to_numpy()
            starts = np.r_[0, np.flatnonzero(rep)]
            ends = np.r_[starts[1:], len(g)]
            for s, e in zip(starts, ends):
                seg = dtc[s:e]
                if np.isnan(seg).all():
                    continue                                   # censored last cycle
                assert seg[-1] == 0                            # cycle ends exactly at the crossing day
                assert (np.diff(seg) == -1).all()              # counts down by one day
                n_cross += 1
                if stage == "sediment":
                    assert g.true_dp_ratio[s + len(seg) - 1] >= 1.8 - 1e-9
                    assert (g.true_dp_ratio[s:s + len(seg) - 1] < 1.8).all()
                if stage == "ro":
                    f, r = g.true_flux_rel, g.true_rejection
                    c = s + len(seg) - 1
                    assert f[c] <= 0.7 + 1e-9 or r[c] <= 0.85 + 1e-9
                    assert ((f[s:c] > 0.7) & (r[s:c] > 0.85)).all()
    assert n_cross > 20


def test_wear_signals_rise_monotonically_within_cycle():
    df, _ = generate_dataset(5, mix="normal", seed=8)
    for _, g in df.groupby("household_id"):
        g = g.reset_index(drop=True)
        starts = np.r_[0, np.flatnonzero(g.replaced_sediment)]
        ends = np.r_[starts[1:], len(g)]
        for s, e in zip(starts, ends):
            assert (np.diff(g.true_dp_ratio[s:e]) >= -1e-12).all()


def test_no_out_tds_above_in_tds_without_fault():
    df, _ = generate_dataset(20, mix="normal", seed=3)
    ok = df[(~df.fault_active) & df.out_tds_ppm.notna() & df.feed_tds_ppm.notna()]
    assert (ok.out_tds_ppm < ok.feed_tds_ppm).all()


def test_idle_days_have_no_sensor_readings_and_zero_usage():
    df, _ = generate_dataset(30, mix="normal", seed=11)
    idle = df[df.op_mode == "idle"]
    assert len(idle) > 0
    assert idle[FAULTABLE_SENSORS].isna().all().all()
    assert (idle.usage_L == 0).all()


def test_all_ten_scenarios_generate():
    assert len(SCENARIOS) >= 10 and len([n for n in SCENARIOS if n != "late_replacement"]) == 10
    for name in SCENARIOS:
        df, meta = generate_household(0, name, seed=1)
        assert list(df.columns) == ALL_COLS and len(df) > 0
        assert df.scenario.iloc[0] == name


def test_insufficient_history_scenario_is_short():
    df, _ = generate_household(0, "insufficient_history", seed=1)
    assert len(df) == 10


def test_stuck_sensor_values_are_flat():
    df, _ = generate_dataset(10, mix="stuck_sensor", seed=2)
    f = df[df.fault_type == "stuck"]
    assert len(f) > 0
    for (_, sensor), g in f.groupby(["household_id", "fault_sensor"]):
        vals = g[sensor].dropna()
        assert vals.nunique() == 1


def test_dropout_and_drift_and_inversion_labels():
    d, _ = generate_dataset(8, mix="missing_readings", seed=2)
    f = d[d.fault_type == "dropout"]
    assert len(f) > 0 and all(f.loc[i, f.loc[i, "fault_sensor"]] != f.loc[i, f.loc[i, "fault_sensor"]] for i in f.index[:50])
    d, _ = generate_dataset(8, mix="sudden_sensor_drift", seed=2)
    assert (d.fault_type == "drift").any()
    inv = [{"kind": "tds_inversion", "prob": 1.0, "duration_range": (10, 20), "magnitude_range": (1.1, 1.3), "min_start": 40}]
    d, _ = generate_dataset(6, mix="normal", seed=2, extra_faults=inv)
    g = d[d.fault_type == "tds_inversion"].dropna(subset=["out_tds_ppm", "feed_tds_ppm"])
    assert len(g) > 0 and (g.out_tds_ppm > g.feed_tds_ppm).all()


def test_source_change_scenario_flags_changes():
    df, meta = generate_dataset(10, mix="source_change", seed=2)
    assert (meta.n_source_changes >= 1).all()
    assert df.source_change_flag.sum() >= 10


def test_low_inlet_mode_present_in_unusual_conditions():
    df, _ = generate_dataset(8, mix="unusual_operating_conditions", seed=2)
    assert (df.op_mode == "low_inlet").any()


def test_laws_differ_and_custom_scenario_works():
    a, _ = generate_dataset(5, mix="normal", seed=1, law="A")
    b, _ = generate_dataset(5, mix="normal", seed=1, law="B")
    assert not np.allclose(a.true_flux_rel, b.true_flux_rel)
    sc = custom_scenario("heavy", {"usage_median_L": 120.0})
    df, _ = generate_dataset(4, mix={sc: 1.0}, seed=1)
    assert (df.scenario == "heavy").all()


def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        generate_household(0, "normal", law="Z")
    with pytest.raises(KeyError):
        generate_household(0, "no_such_scenario")


def test_write_dataset_files_and_labels(tmp_path):
    df, meta = generate_dataset(3, mix="normal", seed=1)
    paths = write_dataset(df, meta, SimConfig(), str(tmp_path))
    assert all(__import__("os").path.exists(p) for p in paths.values())
    txt = (tmp_path / "DATA_README.txt").read_text()
    assert "SYNTHETIC" in txt and "litres/day" in txt
    assert SimConfig.from_dict(__import__("json").load(open(tmp_path / "sim_config.json"))) == SimConfig()
