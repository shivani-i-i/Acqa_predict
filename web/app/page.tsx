"use client";

import { useEffect, useMemo, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const stages = ["sediment", "carbon", "ro"] as const;
type Stage = (typeof stages)[number];
type Household = { id: string; name: string; source: string; now_day: number };
type Forecast = {
  stage_label: string;
  status: string;
  point_days: number | null;
  interval_days: number[] | null;
  reliability: string;
  reason: string | null;
  recommendation: string;
};
type Dashboard = {
  household: Household;
  stages: Record<Stage, { label: string; condition: { progress: number; text: string }; sensor_health: string; forecast: Forecast }>;
  open_alerts: number;
  upcoming_maintenance: Array<{ stage: Stage; status: string; window_lo: number | null; window_hi: number | null }>;
  disclaimer: string;
};

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API}${path}`, { ...init, headers: { "Content-Type": "application/json", ...(init?.headers || {}) } });
  if (!response.ok) throw new Error((await response.json().catch(() => ({}))).detail || `Request failed (${response.status})`);
  return response.json();
}

function Purifier({ dashboard, focus, onFocus }: { dashboard: Dashboard | null; focus: Stage; onFocus: (stage: Stage) => void }) {
  return (
    <div className="scene" aria-label="Interactive procedural purifier view">
      <div className="scene-glow" />
      <div className="purifier">
        <div className="purifier-cap top-cap" />
        {stages.map((stage, index) => {
          const progress = dashboard?.stages[stage]?.condition.progress || 0;
          const color = progress > 0.8 ? "critical" : progress > 0.55 ? "warn" : "ok";
          return (
            <button className={`cartridge ${color} ${focus === stage ? "focused" : ""}`} key={stage} onClick={() => onFocus(stage)}>
              <span className="cartridge-core" />
              <span className="cartridge-label">{dashboard?.stages[stage]?.label || stage}</span>
              <span className="cartridge-wear">{Math.round(progress * 100)}% wear signal</span>
              {dashboard?.stages[stage]?.sensor_health !== "ok" && <span className="alert-orb" />}
            </button>
          );
        })}
        <div className="purifier-cap bottom-cap" />
      </div>
      <div className="water-stream"><i /><i /><i /><i /><i /></div>
    </div>
  );
}

export default function Home() {
  const [households, setHouseholds] = useState<Household[]>([]);
  const [hid, setHid] = useState("");
  const [dashboard, setDashboard] = useState<Dashboard | null>(null);
  const [focus, setFocus] = useState<Stage>("sediment");
  const [tab, setTab] = useState("Dashboard");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [theme, setTheme] = useState<"dark" | "light">("dark");

  const load = async (id = hid) => {
    if (!id) return;
    setDashboard(await api<Dashboard>(`/households/${id}/dashboard`));
  };
  useEffect(() => { api<Household[]>("/households").then((items) => { setHouseholds(items); if (items[0]) { setHid(items[0].id); load(items[0].id); } }).catch((e) => setMessage(e.message)); }, []);

  const createDemo = async () => {
    setBusy(true);
    try {
      const item = await api<Household>("/households", { method: "POST", body: JSON.stringify({ name: "My demo home", scenario: "normal", law: "A", seed: 7, now_day: 60, calibrated: true, overrides: {}, user_faults: [] }) });
      setHouseholds([item]); setHid(item.id); await load(item.id); setMessage("Synthetic demo household created."); setTab("Dashboard");
    } catch (e) { setMessage(e instanceof Error ? e.message : "Could not create household"); } finally { setBusy(false); }
  };

  const current = dashboard?.stages[focus]?.forecast;
  const features = ["Dashboard", "Filter detail", "Alerts", "Planner & costs", "Validation lab", "Models", "Data & export"];
  const title = useMemo(() => tab === "Filter detail" ? `${dashboard?.stages[focus]?.label || "Filter"} detail` : tab, [tab, dashboard, focus]);

  return (
    <main className={theme}>
      <header className="topbar"><a className="brand" href="#"><span className="brand-drop">◆</span><span>AquaPredict<small>purifier care</small></span></a><nav>{features.map((item) => <button className={tab === item ? "active" : ""} key={item} onClick={() => setTab(item)}>{item}</button>)}</nav><button className="theme" onClick={() => setTheme(theme === "dark" ? "light" : "dark")}>{theme === "dark" ? "☼" : "☾"}</button></header>
      <section className="disclaimer">SYNTHETIC DATA PROTOTYPE · Forecasts describe filter wear only and do not assess water safety.</section>
      {message && <div className="toast" onClick={() => setMessage("")}>{message}</div>}
      {!dashboard ? (
        <section className="welcome"><div><p className="eyebrow">CONNECTED PURIFIER INTELLIGENCE</p><h1>See the health of every filter at a glance.</h1><p className="lede">AquaPredict combines telemetry, sensor-quality checks, forecasts, alerts, planning, and validation in one care centre.</p><button className="primary" onClick={createDemo} disabled={busy}>{busy ? "Creating..." : "Create synthetic demo household"}</button><p className="muted">All demo results are synthetic and reproducible.</p></div><Purifier dashboard={null} focus={focus} onFocus={setFocus} /></section>
      ) : (
        <section className="workspace">
          <div className="workspace-head"><div><p className="eyebrow">LIVE PURIFIER WORKSPACE</p><h1>{title}</h1><p className="muted">{dashboard.household.name} · {dashboard.household.source} telemetry · day {dashboard.household.now_day}</p></div><select value={hid} onChange={(e) => { setHid(e.target.value); load(e.target.value); }}>{households.map((h) => <option value={h.id} key={h.id}>{h.name}</option>)}</select></div>
          <div className="dashboard-grid"><div className="panel scene-panel"><Purifier dashboard={dashboard} focus={focus} onFocus={(stage) => { setFocus(stage); setTab("Filter detail"); }} /><p className="scene-hint">Select a cartridge to inspect its forecast, sensor health, and maintenance recommendation.</p></div><div className="panel insight-panel"><div className="panel-heading"><span>FORECAST READOUT</span><b>{dashboard.open_alerts} open alerts</b></div><div className="forecast-hero"><strong>{current?.point_days == null ? "—" : `${Math.round(current.point_days)}d`}</strong><span>estimated {dashboard.stages[focus].label.toLowerCase()} life remaining</span></div><div className={`status ${current?.status}`}>{current?.status || "loading"}</div><p>{current?.recommendation}</p>{current?.interval_days && <div className="interval"><span>Calibrated interval</span><b>{Math.round(current.interval_days[0])}–{Math.round(current.interval_days[1])} days</b></div>}<p className="muted">{current?.reason || current?.reliability}</p></div></div>
          <div className="cards">{stages.map((stage) => { const item = dashboard.stages[stage]; const fc = item.forecast; return <button className={`filter-card ${focus === stage ? "selected" : ""}`} key={stage} onClick={() => setFocus(stage)}><span className="card-kicker">{item.label}</span><strong>{fc.point_days == null ? "Withheld" : `${Math.round(fc.point_days)} days`}</strong><span className={`status ${fc.status}`}>{fc.status}</span><small>{item.condition.text}</small></button>; })}</div>
          {tab === "Alerts" && <div className="panel lower"><h2>Alerts and sensor events</h2><p>Alerts are severity ordered and can be acknowledged, resolved, or dismissed through the API.</p><button className="secondary" onClick={() => load()}>Refresh monitoring</button></div>}
          {tab === "Planner & costs" && <div className="panel lower"><h2>Maintenance planner</h2>{dashboard.upcoming_maintenance.length ? dashboard.upcoming_maintenance.map((item) => <p key={item.stage}>{dashboard.stages[item.stage].label}: <b>{item.status}</b> · {item.window_lo ?? "—"}–{item.window_hi ?? "—"} days</p>) : <p className="muted">No upcoming maintenance plan yet.</p>}</div>}
          {tab === "Validation lab" && <div className="panel lower"><h2>Validation lab</h2><p>Run strict evaluation on fresh synthetic households from the Streamlit app or `POST /validation/run`. Test households are never used for tuning.</p></div>}
          {tab === "Models" && <div className="panel lower"><h2>Model registry</h2><p>Compare transparent baselines, the reference model, and the integrated forecasting pipeline. Artifact integrity is verified with SHA-256 sidecars.</p></div>}
          {tab === "Data & export" && <div className="panel lower"><h2>Data boundary</h2><p>Upload household telemetry through the API and use USGS only as environmental context. USGS data never become filter-life labels.</p></div>}
        </section>
      )}
      <footer>{dashboard?.disclaimer || "AquaPredict · Synthetic data prototype · Not a water-safety assessment"}</footer>
    </main>
  );
}
