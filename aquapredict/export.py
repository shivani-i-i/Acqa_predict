"""Exports. Every file carries units, data-source label, seed and config so exported values match the app."""
from __future__ import annotations
import json
import os
import pandas as pd
from .schema import UNITS, DATA_SOURCE_SYNTHETIC

NOTICE = ("ALL DATA IN THIS FOLDER ARE SYNTHETIC (internally developed simulator). They are NOT Atomberg "
          "telemetry. Results on them show algorithmic feasibility, not real-world accuracy.")


def save_table(df: pd.DataFrame, path_no_ext: str) -> str:
    """Parquet if pyarrow is installed, otherwise CSV."""
    os.makedirs(os.path.dirname(path_no_ext) or ".", exist_ok=True)
    try:
        import pyarrow  # noqa: F401
        p = path_no_ext + ".parquet"; df.to_parquet(p, index=False)
    except ImportError:
        p = path_no_ext + ".csv"; df.to_csv(p, index=False)
    return p


def write_dataset(df, meta, cfg, out_dir, law="A", seed=0):
    os.makedirs(out_dir, exist_ok=True)
    paths = {
        "telemetry": save_table(df, os.path.join(out_dir, "telemetry")),
        "households": save_table(meta, os.path.join(out_dir, "households_meta")),
    }
    cfg.to_json(os.path.join(out_dir, "sim_config.json"))
    with open(os.path.join(out_dir, "DATA_README.txt"), "w") as f:
        f.write(NOTICE + f"\n\ndata_source={DATA_SOURCE_SYNTHETIC} law={law} seed={seed} "
                f"households={df.household_id.nunique()} rows={len(df)}\n\nUnits:\n")
        for k, v in UNITS.items():
            f.write(f"  {k}: {v}\n")
        f.write("\nColumns starting with true_, fault_, source_ and missing_injected are GROUND TRUTH labels, "
                "never to be given to a forecaster.\n")
    return paths


def export_scenario(scenario, out_path):
    d = {"name": scenario.name, "description": scenario.description, "overrides": scenario.overrides,
         "fault_templates": list(scenario.fault_templates), "data_source": DATA_SOURCE_SYNTHETIC}
    with open(out_path, "w") as f:
        json.dump(d, f, indent=2)
    return out_path


def export_results(result_tables: dict, out_dir):
    """result_tables: name -> DataFrame. Writes CSVs plus a notice."""
    os.makedirs(out_dir, exist_ok=True)
    out = {}
    for name, t in result_tables.items():
        p = os.path.join(out_dir, f"{name}.csv"); t.to_csv(p); out[name] = p
    with open(os.path.join(out_dir, "NOTICE.txt"), "w") as f:
        f.write(NOTICE + "\n")
    return out
