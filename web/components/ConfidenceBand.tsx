import type { Forecast } from "../lib/api";

export function ConfidenceBand({ forecast }: { forecast: Forecast }) {
  if (!forecast.interval_days) {
    return <div className="honesty-note">No calibrated interval is available for this forecast.</div>;
  }
  const [low, high] = forecast.interval_days;
  const point = forecast.point_days ?? high;
  const max = Math.max(high, point, 1);
  return (
    <div className="confidence">
      <div className="confidence-scale"><span>today</span><span>{Math.round(low)}d</span><span>{Math.round(point)}d</span><span>{Math.round(high)}d</span></div>
      <div className="confidence-track"><i style={{ left: `${(low / max) * 100}%`, width: `${((high - low) / max) * 100}%` }} /><b style={{ left: `${(point / max) * 100}%` }} /></div>
      <small>Separate-household conformal interval; not a guarantee.</small>
    </div>
  );
}
