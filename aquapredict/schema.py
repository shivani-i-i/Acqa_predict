"""Shared data contract for AquaPredict.

DRAFT by Person 1. All three people must agree before changing anything here.
One row = one household-day. Every number is SYNTHETIC (data_source = "synthetic").

Forecaster contract (Person 2 implements, the evaluation harness calls):

    forecaster(hist: pandas.DataFrame, stage: str) -> dict
        hist  : all rows of ONE household up to and including the forecast day,
                VISIBLE columns only (no truth/labels ever reach a forecaster).
        stage : "sediment" | "carbon" | "ro"
        returns {"point": float days-to-criterion,
                 "lo": float|None, "hi": float|None,   # prediction interval in days
                 "withheld": bool, "reason": str}      # withheld=True -> no forecast issued
"""
from __future__ import annotations

STAGES = ("sediment", "carbon", "ro")
OP_MODES = ("normal", "low_inlet", "idle")
DATA_SOURCE_SYNTHETIC = "synthetic"

MEASURED_SENSORS = ["usage_L", "feed_tds_ppm", "out_tds_ppm", "temp_C",
                    "inlet_bar", "flow_Lpm", "dp_sediment_bar"]
FAULTABLE_SENSORS = [s for s in MEASURED_SENSORS if s != "usage_L"]

UNITS = {
    "usage_L": "litres/day", "feed_tds_ppm": "ppm (mg/L)", "out_tds_ppm": "ppm (mg/L)",
    "temp_C": "degC", "inlet_bar": "bar", "flow_Lpm": "litres/min (permeate)",
    "dp_sediment_bar": "bar (pressure drop across sediment stage)",
    "age_<stage>_d": "days since this stage's filter was installed",
    "cum_vol_<stage>_L": "litres through this stage since installation",
    "replaced_<stage>": "bool, filter replaced at start of this day (maintenance record)",
    "true_dtc_<stage>": "days until the replacement criterion is reached (TRUTH). 0 = crossing day, NEGATIVE = filter is overdue (criterion already crossed, not yet replaced); NaN = censored (never crossed)",
}

# ---- replacement criteria (documented simulator assumptions, configurable) ----
SEDIMENT_DP_RATIO_LIMIT = 1.8    # (dp/flow) relative to clean value
RO_FLUX_LIMIT = 0.70             # temperature-corrected flux relative to clean value
RO_REJECTION_LIMIT = 0.85        # 1 - TDS_out/TDS_in
CARBON_CAPACITY_RATIO_LIMIT = 1.0  # cumulative load / capacity (no direct sensor)

# ---- simulator reference load: litres at reference conditions after which a median-wear household
# crosses the criterion. ILLUSTRATIVE. ----
SIM_REF_VOLUME_L = {"sediment": 18000.0, "carbon": 12000.0, "ro": 20000.0}

# ---- "rated life" used by the volume / 20% baselines. ILLUSTRATIVE, NOT manufacturer data.
# Chosen near the median simulated life so the volume rule is a fair (not strawman) baseline;
# the team should state this in the paper. Replace with real rated values when known. ----
DEFAULT_RATED = {
    "sediment": {"volume_L": 6000.0, "days": 365.0},
    "carbon":   {"volume_L": 8000.0, "days": 365.0},
    "ro":       {"volume_L": 12000.0, "days": 730.0},
}

def age_col(s): return f"age_{s}_d"
def vol_col(s): return f"cum_vol_{s}_L"
def repl_col(s): return f"replaced_{s}"
def dtc_col(s): return f"true_dtc_{s}"

META_COLS = ["scenario", "data_source"]
VISIBLE_COLS = (["household_id", "day", "date"] + MEASURED_SENSORS + ["op_mode"]
                + [age_col(s) for s in STAGES] + [vol_col(s) for s in STAGES]
                + [repl_col(s) for s in STAGES])
TRUTH_COLS = ([dtc_col(s) for s in STAGES]
              + ["true_dp_ratio", "true_flux_rel", "true_rejection", "true_carbon_ratio",
                 "fault_type", "fault_sensor", "fault_active", "missing_injected",
                 "source_change_flag", "source_regime"])
ALL_COLS = ["household_id", "scenario", "data_source", "day", "date"] + VISIBLE_COLS[3:] + TRUTH_COLS
