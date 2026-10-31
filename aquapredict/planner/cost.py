"""Replacement-policy cost comparison on SIMULATED usage. Not a savings claim.

Policy replacement day = alert day + notice days (time to schedule); if a policy never alerts the user
replaces reactively `reactive_delay_days` after the criterion is crossed. 'oracle' = hindsight lower bound.
PRICES ARE ILLUSTRATIVE PLACEHOLDERS (only RO = Rs 1,100 comes from the abstract's ref [1]; verify it).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class CostConfig:
    replacement_cost: dict = field(default_factory=lambda: {"sediment": 500.0, "carbon": 1000.0, "ro": 1100.0})
    service_visit_cost: float = 800.0
    notice_days: float = 7.0
    reactive_delay_days: float = 7.0
    status: str = ("ILLUSTRATIVE placeholders; only the RO membrane price (Rs 1,100) is from the abstract's "
                   "reference [1] and must be re-verified. Do not present as current manufacturer prices.")


def policy_table(cycles: pd.DataFrame, cc: CostConfig | None = None, include_oracle=True) -> pd.DataFrame:
    cc = cc or CostConfig()
    c = cycles.copy()
    if include_oracle:
        o = c[c["method"] == c["method"].iloc[0]].copy()
        o["method"] = "oracle"; o["alert_day"] = o["crossing"] - cc.notice_days
        c = pd.concat([c, o], ignore_index=True)
    has = c["alert_day"].notna()
    c["replace_day"] = np.where(has, c["alert_day"] + cc.notice_days, c["crossing"] + cc.reactive_delay_days)
    c["used"] = c["replace_day"] - c["cycle_start"]
    c["wasted"] = (c["crossing"] - c["replace_day"]).clip(lower=0)
    c["overdue"] = (c["replace_day"] - c["crossing"]).clip(lower=0)
    c["life"] = c["crossing"] - c["cycle_start"]
    c["cost"] = c["stage"].map(cc.replacement_cost) + cc.service_visit_cost
    rows = []
    for m, g in c.groupby("method"):
        annual = sum(s["cost"].sum() / s["used"].sum() * 365.0 for _, s in g.groupby("stage"))
        rows.append({"method": m, "n_cycles": len(g), "annual_cost_per_purifier": annual,
                     "wasted_life_pct": 100 * g["wasted"].sum() / g["life"].sum(),
                     "overdue_days_per_cycle": g["overdue"].mean(),
                     "late_share": (g["overdue"] > 0).mean()})
    return pd.DataFrame(rows).set_index("method")


def sensitivity(cycles, cost_mults=(0.5, 1.0, 2.0), service_mults=(0.0, 1.0, 3.0), base: CostConfig | None = None):
    """How the annual-cost gap vs the volume rule changes with prices (assumption sensitivity)."""
    base = base or CostConfig()
    out = []
    for cm in cost_mults:
        for sm in service_mults:
            cc = CostConfig({k: v * cm for k, v in base.replacement_cost.items()}, base.service_visit_cost * sm,
                            base.notice_days, base.reactive_delay_days)
            t = policy_table(cycles, cc)
            for m in t.index:
                out.append({"replacement_cost_x": cm, "service_cost_x": sm, "method": m,
                            "annual_cost": t.loc[m, "annual_cost_per_purifier"],
                            "vs_volume_rule": t.loc[m, "annual_cost_per_purifier"] - t.loc["volume_rule", "annual_cost_per_purifier"]
                            if "volume_rule" in t.index else np.nan,
                            "overdue_days_per_cycle": t.loc[m, "overdue_days_per_cycle"]})
    return pd.DataFrame(out)
