"""Figures for the paper/slides. Every plot is drawn from actual simulator/evaluation output and labelled SYNTHETIC."""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TAG = "SYNTHETIC DATA (internal simulator) - not Atomberg telemetry"
COL = {"volume_rule": "#9aa0a6", "health_20pct": "#e0a030", "trend": "#2a7ab9", "AquaPredict": "#c0392b", "oracle": "#444"}


def fig_household(df, out_path, stage="sediment"):
    """One household: what the sensors see, the replacement cycles, injected faults and source changes."""
    g = df.sort_values("day")
    d = g["day"].to_numpy()
    fig, ax = plt.subplots(4, 1, figsize=(10, 8.5), sharex=True)
    ratio = (g["dp_sediment_bar"] / g["flow_Lpm"])
    ax[0].plot(d, ratio / np.nanmedian(ratio.iloc[:10]), ".", ms=3, color="#2a7ab9", label="measured (dp/flow), normalised")
    ax[0].plot(d, g["true_dp_ratio"], "-", color="k", lw=1, label="true (hidden)")
    ax[0].axhline(1.8, ls="--", color="r", lw=.8, label="replacement criterion 1.8x")
    ax[1].plot(d, g["flow_Lpm"], ".", ms=3, color="#2a9d8f", label="flow (L/min)")
    ax[1].set_ylabel("flow"); ax2 = ax[1].twinx(); ax2.plot(d, g["temp_C"], "-", color="#e76f51", lw=.8, alpha=.7); ax2.set_ylabel("temp C")
    ax[2].plot(d, g["feed_tds_ppm"], ".", ms=3, color="#6a4c93", label="TDS in")
    ax[2].plot(d, g["out_tds_ppm"], ".", ms=3, color="#f4a261", label="TDS out")
    ax[3].plot(d, g[f"true_dtc_{stage}"], color="k"); ax[3].set_ylabel(f"true days to\ncriterion ({stage})")
    for a in ax[:3]:
        for r in d[g[f"replaced_{stage}"].to_numpy()]:
            a.axvline(r, color="#2a9d8f", lw=.8, alpha=.6)
        for r in d[g["source_change_flag"].to_numpy()]:
            a.axvline(r, color="#6a4c93", lw=1.2, ls=":")
        f = g["fault_active"].to_numpy()
        if f.any():
            a.fill_between(d, 0, 1, where=f, transform=a.get_xaxis_transform(), color="red", alpha=.12)
    ax[0].set_ylabel("sediment dp/flow"); ax[2].set_ylabel("TDS (ppm)"); ax[3].set_xlabel("day")
    ax[0].legend(loc="upper left", fontsize=7); ax[2].legend(loc="upper left", fontsize=7)
    sc = g["scenario"].iloc[0]
    fig.suptitle(f"{g['household_id'].iloc[0]} - scenario '{sc}'  (green: filter replaced, purple: source change, red band: injected fault)\n{TAG}", fontsize=9)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    return out_path


def fig_comparison(ci_tables, out_path):
    e, l, p = ci_tables["median_abs_err_ci"], ci_tables["late_rate_ci"], ci_tables["premature_rate_ci"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for a, t, title, scale in ((ax[0], e, "Median |window error| (days)\nforecasts within 60 d of criterion", 1),
                               (ax[1], None, "Late vs premature warnings (%)", 100)):
        if t is not None:
            x = np.arange(len(t))
            a.bar(x, t["estimate"], color=[COL.get(m, "#888") for m in t["method"]],
                  yerr=[t["estimate"] - t["ci_lo"], t["ci_hi"] - t["estimate"]], capsize=4)
            a.set_xticks(x); a.set_xticklabels(t["method"], rotation=15)
        else:
            x = np.arange(len(l)); w = .38
            a.bar(x - w / 2, 100 * l["estimate"], w, label="late", color="#c0392b",
                  yerr=[100 * (l["estimate"] - l["ci_lo"]), 100 * (l["ci_hi"] - l["estimate"])], capsize=3)
            a.bar(x + w / 2, 100 * p["estimate"], w, label="premature (false alarm)", color="#e0a030",
                  yerr=[100 * (p["estimate"] - p["ci_lo"]), 100 * (p["ci_hi"] - p["estimate"])], capsize=3)
            a.set_xticks(x); a.set_xticklabels(l["method"], rotation=15); a.legend(fontsize=8)
        a.set_title(title, fontsize=9)
    fig.suptitle("Baselines on held-out households, pooled over seeds, 95% household-bootstrap CI\n" + TAG, fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    return out_path


def fig_operating_curve(curve, out_path):
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for m, g in curve.groupby("method"):
        g = g.sort_values("lead_target")
        ax.plot(100 * g["premature_rate"], 100 * g["late_rate"], "-o", ms=3, label=m, color=COL.get(m, None))
    ax.set_xlabel("premature (false-alarm) rate %"); ax.set_ylabel("late-warning rate %")
    ax.set_title("Operating curves: lower-left is better\n" + TAG, fontsize=8); ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    return out_path
