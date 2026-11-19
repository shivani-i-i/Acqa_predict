"use client";

import dynamic from "next/dynamic";
import { useCallback, useEffect, useMemo, useState } from "react";
import { Activity, AlertTriangle, BarChart3, Database, Droplets, Gauge, Moon, RefreshCw, Sun, Wrench } from "lucide-react";
import { api, type Alert, type Dashboard, type Household, type Stage } from "../lib/api";
import { ConfidenceBand } from "../components/ConfidenceBand";

const PurifierScene = dynamic(() => import("../components/PurifierScene"), {
  ssr: false,
  loading: () => <div className="scene-loading">Loading the digital twin…</div>,
});

const stages: Stage[] = ["sediment", "carbon", "ro"];
const tabs = ["Dashboard", "Filter detail", "Alerts", "Planner", "Validation", "Models", "Data"];

function statusLabel(status: string) {
  return status.replace("_", " ");
}

export default function Home() {
  const [households, setHouseholds] = useState<Household[]>([]);
  const [householdId, setHouseholdId] = useState("");
  const [dashboard, setDashboard] = useState<Dashboard | null>(null);
  const [selected, setSelected] = useState<Stage>("sediment");
  const [tab, setTab] = useState("Dashboard");
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [theme, setTheme] = useState<"dark" | "mist">("dark");
  const [loading, setLoading] = useState(true);
  const [working, setWorking] = useState(false);
  const [error, setError] = useState("");
  const [validation, setValidation] = useState<Record<string, unknown> | null>(null);

  const refresh = useCallback(async (id = householdId) => {
    if (!id) return;
    setLoading(true);
    try {
      const [nextDashboard, nextAlerts] = await Promise.all([api.households.dashboard(id), api.alerts.list(id)]);
      setDashboard(nextDashboard);
      setAlerts(nextAlerts);
      setError("");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not load purifier data.");
    } finally {
      setLoading(false);
    }
  }, [householdId]);

  useEffect(() => {
    api.households.list().then((items) => {
      setHouseholds(items);
      if (items[0]) {
        setHouseholdId(items[0].id);
        refresh(items[0].id);
      } else setLoading(false);
    }).catch((reason) => {
      setError(reason instanceof Error ? reason.message : "Could not connect to the API.");
      setLoading(false);
    });
  }, [refresh]);

  const createDemo = async () => {
    setWorking(true);
    try {
      const household = await api.households.createDemo();
      setHouseholds([household]);
      setHouseholdId(household.id);
      await refresh(household.id);
      setTab("Dashboard");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not create the demo household.");
    } finally {
      setWorking(false);
    }
  };

  const focusForecast = dashboard?.stages[selected].forecast;
  const pageTitle = tab === "Filter detail" ? `${dashboard?.stages[selected].label || "Filter"} detail` : tab;

  return (
    <main className={theme}>
      <header className="topbar">
        <a className="brand" href="#"><span className="brand-mark"><Droplets size={18} /></span><span>AquaPredict<small>purifier care centre</small></span></a>
        <nav aria-label="Primary navigation">{tabs.map((item) => <button className={tab === item ? "active" : ""} key={item} onClick={() => setTab(item)}>{item}</button>)}</nav>
        <button className="theme-toggle" aria-label="Toggle theme" onClick={() => setTheme(theme === "dark" ? "mist" : "dark")}>{theme === "dark" ? <Sun size={17} /> : <Moon size={17} />}</button>
      </header>
      <div className="notice"><span>SYNTHETIC DATA PROTOTYPE</span> Forecasts describe filter wear only and do not assess water safety.</div>
      {error && <div className="error-banner" role="alert"><AlertTriangle size={16} />{error}<button onClick={() => setError("")}>Dismiss</button></div>}
      {!dashboard ? (
        <section className="landing">
          <div className="landing-copy"><p className="eyebrow">CONNECTED PURIFIER INTELLIGENCE</p><h1>A calmer way to care for every filter.</h1><p className="lede">Explore a digital twin of your purifier, understand sensor health, and plan maintenance using transparent synthetic forecasts.</p><button className="primary" onClick={createDemo} disabled={working}>{working ? "Creating demo…" : "Create synthetic demo household"}</button><p className="muted">No real-world accuracy is implied. Your demo is reproducible and clearly labelled.</p></div>
          <div className="hero-scene"><PurifierScene dashboard={null} selected={selected} onSelect={setSelected} /></div>
        </section>
      ) : (
        <section className="workspace">
          <div className="workspace-head"><div><p className="eyebrow">LIVE PURIFIER WORKSPACE</p><h1>{pageTitle}</h1><p className="muted">{dashboard.household.name} · <span className="source-badge">{dashboard.household.source}</span> telemetry · day {dashboard.household.now_day}</p></div><div className="header-actions"><select value={householdId} onChange={(event) => { setHouseholdId(event.target.value); refresh(event.target.value); }}>{households.map((household) => <option value={household.id} key={household.id}>{household.name}</option>)}</select><button className="icon-button" onClick={() => refresh()} aria-label="Refresh"><RefreshCw size={17} /></button></div></div>
          {tab === "Dashboard" || tab === "Filter detail" ? <DashboardView dashboard={dashboard} selected={selected} setSelected={(stage) => { setSelected(stage); setTab("Filter detail"); }} forecast={focusForecast} /> : <FeatureView tab={tab} dashboard={dashboard} alerts={alerts} setAlerts={setAlerts} validation={validation} setValidation={setValidation} working={working} setWorking={setWorking} refresh={refresh} />}
        </section>
      )}
      <footer>{dashboard?.disclaimer || "AquaPredict · Synthetic data prototype · Not a water-safety assessment"}</footer>
    </main>
  );
}

function DashboardView({ dashboard, selected, setSelected, forecast }: { dashboard: Dashboard; selected: Stage; setSelected: (stage: Stage) => void; forecast?: Dashboard["stages"][Stage]["forecast"] }) {
  const selectedStage = dashboard.stages[selected];
  if (!selectedStage || !forecast) return <div className="panel empty-state"><p>No forecast is available for this stage.</p></div>;
  return <><div className="dashboard-grid"><div className="panel scene-panel"><div className="panel-kicker">DIGITAL TWIN / SELECT A STAGE</div><PurifierScene dashboard={dashboard} selected={selected} onSelect={setSelected} /><p className="scene-hint">Click a cartridge to focus its forecast and maintenance evidence.</p></div><div className="panel insight-panel"><div className="panel-heading"><span>FORECAST READOUT</span><b>{dashboard.open_alerts} open alerts</b></div><div className="forecast-hero"><strong>{forecast.point_days == null ? "—" : `${Math.round(forecast.point_days)}d`}</strong><span>{selectedStage.label.toLowerCase()} life remaining</span></div><span className={`status ${forecast.status}`}>{statusLabel(forecast.status)}</span><p>{forecast.recommendation}</p><ConfidenceBand forecast={forecast} /><p className="muted">{forecast.reason || forecast.reliability}</p></div></div><div className="cards">{stages.map((stage) => <FilterCard key={stage} stage={stage} item={dashboard.stages[stage]} selected={selected === stage} onClick={() => setSelected(stage)} />)}</div></>;
}

function FilterCard({ stage, item, selected, onClick }: { stage: Stage; item: Dashboard["stages"][Stage]; selected: boolean; onClick: () => void }) {
  const forecast = item.forecast;
  return <button className={`filter-card ${selected ? "selected" : ""}`} onClick={onClick}><span className="card-kicker">{item.label}</span><strong>{forecast.point_days == null ? "Withheld" : `${Math.round(forecast.point_days)} days`}</strong><span className={`status ${forecast.status}`}>{statusLabel(forecast.status)}</span><small>{item.condition.text}</small><span className={`health-dot ${item.sensor_health}`} />{stage === "carbon" && <em>usage rule</em>}</button>;
}

function FeatureView({ tab, dashboard, alerts, setAlerts, validation, setValidation, working, setWorking, refresh }: { tab: string; dashboard: Dashboard; alerts: Alert[]; setAlerts: (alerts: Alert[]) => void; validation: Record<string, unknown> | null; setValidation: (value: Record<string, unknown> | null) => void; working: boolean; setWorking: (value: boolean) => void; refresh: () => Promise<void> }) {
  if (tab === "Alerts") return <section className="feature-grid"><div className="panel feature-copy"><p className="eyebrow">SENSOR HEALTH</p><h2>Alerts that explain themselves.</h2><p>Fault markers are signals to investigate, not proof that a sensor is broken. Acknowledge, resolve, or dismiss an alert with a clear audit trail.</p></div><div className="panel list-panel">{alerts.length ? alerts.map((alert) => <div className="alert-row" key={alert.id}><span className={`severity ${alert.severity}`}>{alert.severity}</span><div><b>{alert.title}</b><small>{alert.stage} · day {alert.day} · {alert.status}</small></div><button onClick={async () => { await api.alerts.action(alert.id, "acknowledge"); setAlerts(alerts.map((item) => item.id === alert.id ? { ...item, status: "acknowledged" } : item)); }}>Acknowledge</button></div>) : <EmptyState icon={<Activity />} text="No active alerts for this household." />}</div></section>;
  if (tab === "Planner") return <section className="feature-grid"><div className="panel feature-copy"><p className="eyebrow">MAINTENANCE PLANNER</p><h2>Plan before urgency.</h2><p>Maintenance states stay separate: predicted, recommended, confirmed, and completed. Prices are illustrative only.</p></div><div className="panel list-panel">{dashboard.upcoming_maintenance.length ? dashboard.upcoming_maintenance.map((item) => <div className="plan-row" key={item.stage}><Wrench size={18} /><div><b>{dashboard.stages[item.stage].label}</b><small>{item.status} · {item.window_lo ?? "—"}–{item.window_hi ?? "—"} days</small></div></div>) : <EmptyState icon={<Wrench />} text="No upcoming maintenance plan." />}</div></section>;
  if (tab === "Validation") return <section className="feature-grid"><div className="panel feature-copy"><p className="eyebrow">VALIDATION LAB</p><h2>Measure on fresh synthetic homes.</h2><p>Compare models against transparent baselines. Results are synthetic and never claim real-world accuracy.</p><button className="primary" disabled={working} onClick={async () => { setWorking(true); try { setValidation(await api.validation.run()); } finally { setWorking(false); } }}>{working ? "Running…" : "Run validation"}</button></div><div className="panel result-panel">{validation ? <pre>{JSON.stringify(validation, null, 2)}</pre> : <EmptyState icon={<BarChart3 />} text="Run the lab to see model criteria and interval metrics." />}</div></section>;
  if (tab === "Models") return <section className="feature-grid"><div className="panel feature-copy"><p className="eyebrow">MODEL REGISTRY</p><h2>Transparent about limits.</h2><p>Model cards, synthetic benchmark context, training state, and artifact-integrity checks belong together.</p></div><div className="panel result-panel"><p><Gauge size={18} /> Reference comparison model</p><p><Activity size={18} /> Integrated forecasting model</p><p className="muted">Train and manage artifacts from the Streamlit Models page or API.</p></div></section>;
  return <section className="feature-grid"><div className="panel feature-copy"><p className="eyebrow">DATA BOUNDARY</p><h2>Context without confusion.</h2><p>Uploaded telemetry remains separate from synthetic truth. USGS observations provide environmental context only and never become filter-life labels.</p></div><div className="panel result-panel"><p><Database size={18} /> CSV exports and household configuration</p><p><Droplets size={18} /> Sensor readings remain source-labelled</p><p className="muted">Use the full API for upload and export operations.</p><button className="secondary" onClick={() => refresh()}>Refresh data</button></div></section>;
}

function EmptyState({ icon, text }: { icon: React.ReactNode; text: string }) {
  return <div className="empty-state">{icon}<p>{text}</p></div>;
}
