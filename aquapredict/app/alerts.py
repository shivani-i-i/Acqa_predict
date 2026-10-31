"""Alert rules. Pure functions: current evidence -> candidate alerts. The service handles persistence, de-duplication and
the open / acknowledged / resolved / dismissed lifecycle. Model-generated alerts are never confused with user actions."""
from __future__ import annotations

LEAD_DAYS = 21          # a maintenance alert fires when the (lower bound of the) window is within this many days
URGENT_DAYS = 7
SAFETY_NOTE = " (This is about filter wear, not water safety.)"


def _alert(kind, severity, title, message, action, key, details=None):
    return {"kind": kind, "severity": severity, "title": title, "message": message, "recommended_action": action,
            "dedupe_key": key, "details": details or {}}


def candidates(stage: str, fc: dict, events: list[dict], indicator: dict, lead: int = LEAD_DAYS) -> list[dict]:
    out = []
    st = fc["status"]
    if st == "withheld":
        out.append(_alert("forecast_withheld", "medium", f"{stage}: forecast withheld", f"Reason: {fc['reason']}",
                          "Fix the data problem (check the flagged sensors) before relying on any replacement estimate.",
                          f"{stage}|forecast_withheld"))
    else:
        iv = fc["interval_days"]
        basis = iv[0] if iv else fc["point_days"]
        label = f"lower bound {iv[0]:.0f} d" if iv else f"point estimate {fc['point_days']:.0f} d, no calibrated interval"
        if basis <= lead:
            sev = "high" if basis <= URGENT_DAYS else "medium"
            out.append(_alert("maintenance_due", sev, f"{stage}: replacement window approaching",
                              f"Estimated time to the replacement criterion: {label}. Predicted, not confirmed." + SAFETY_NOTE,
                              "Plan the replacement inside the predicted window; confirm it in the planner.",
                              f"{stage}|maintenance_due", {"basis_days": float(basis), "point_days": fc["point_days"], "interval": iv}))
    for e in events:
        if e["kind"] in ("stuck", "stale", "missing", "tds_inversion") and e.get("blocks_forecast", True):
            sev = "medium" if e["kind"] == "missing" else "high"
            out.append(_alert("sensor_fault", sev, f"{stage}: suspected sensor problem ({e['sensor']})",
                              f"{e['reason']}. Suspected, not confirmed; it can make this stage's forecast unreliable.",
                              "Inspect the sensor / wiring; the forecast is withheld while the problem persists." if e["kind"] != "missing"
                              else "Check connectivity; repeated gaps reduce forecast reliability.",
                              f"{stage}|sensor_fault|{e['sensor']}|{e['kind']}", {"sensor": e["sensor"], "type": e["kind"]}))
    sc = fc.get("source_change") or {}
    if sc.get("detected"):
        out.append(_alert("source_change", "low", f"{stage}: possible water-source or operating-condition change",
                          f"{sc.get('message', '')} A statistical signal only; it is not proof of a new source.",
                          "Check whether the household's water source changed; wear may now be faster or slower.",
                          f"{stage}|source_change"))
    if indicator.get("usage_based") and indicator.get("progress") is not None and indicator["progress"] >= 1.0:
        out.append(_alert("rated_usage_reached", "medium", f"{stage}: illustrative rated usage reached",
                          f"{indicator['text']} No sensor tracks this stage, so this is a usage count, not a measurement of wear.",
                          "Consider planning a replacement; the rating is an illustrative assumption, not manufacturer data.",
                          f"{stage}|rated_usage_reached"))
    elif indicator.get("available") and indicator.get("progress") is not None and indicator["progress"] >= 1.0:
        out.append(_alert("indicator_past_criterion", "high", f"{stage}: wear indicator has passed the assumed criterion",
                          f"{indicator['text']} This is the sensor indicator passing the assumed threshold; it is NOT a confirmed failure."
                          + SAFETY_NOTE, "Replace the filter or verify the reading.", f"{stage}|indicator_past_criterion"))
    return out
