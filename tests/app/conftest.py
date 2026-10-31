import numpy as np
import pytest
from aquapredict.app.service import AquaService
from aquapredict.simulator.config import SimConfig
from aquapredict.simulator.generate import generate_household


@pytest.fixture
def svc(tmp_path):
    return AquaService(str(tmp_path / "t.db"), str(tmp_path / "art"))


@pytest.fixture(scope="session")
def trained_dir(tmp_path_factory):
    """Both learned models trained once at tiny scale (about 25 s) and shared by the tests that need them."""
    d = tmp_path_factory.mktemp("trained")
    s = AquaService(str(d / "x.db"), str(d / "art"))
    s.registry.train(n=36, seed=77, max_iter=30, stride=8, log=lambda *_: None)
    return str(d / "art")


def uploaded_rows(index=5, seed=2, days=120, replacements=True):
    """Daily rows as a real device would send them: no ground truth, no derived columns."""
    df, _ = generate_household(index, "normal", SimConfig.calibrated(), "A", seed)
    df = df.iloc[:days]
    cols = ["day", "usage_L", "dp_sediment_bar", "flow_Lpm", "out_tds_ppm", "feed_tds_ppm", "temp_C", "inlet_bar"]
    rows = [{k: (None if (isinstance(v, float) and np.isnan(v)) else (int(v) if k == "day" else float(v))) for k, v in r.items()}
            for r in df[cols].to_dict("records")]
    repl = {s: [int(d) for d in df["day"][df[f"replaced_{s}"]]] for s in ("sediment", "carbon", "ro")} if replacements else {}
    return rows, repl
