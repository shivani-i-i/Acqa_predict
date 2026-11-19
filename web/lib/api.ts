export const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export type Stage = "sediment" | "carbon" | "ro";
export type ForecastStatus = "ok" | "degraded" | "withheld" | "point_only" | "interval";

export interface Household {
  id: string;
  name: string;
  source: "synthetic" | "uploaded";
  now_day: number;
}

export interface Forecast {
  stage_label: string;
  status: ForecastStatus;
  point_days: number | null;
  interval_days: number[] | null;
  reliability: string;
  reason: string | null;
  recommendation: string;
  drivers?: Array<Record<string, string | number>>;
  sensor_events?: Array<Record<string, string | number>>;
  interval_status?: string;
}

export interface DashboardStage {
  label: string;
  condition: { progress: number; text: string; available?: boolean };
  sensor_health: string;
  forecast: Forecast;
}

export interface Dashboard {
  household: Household;
  stages: Record<Stage, DashboardStage>;
  open_alerts: number;
  upcoming_maintenance: Array<{ stage: Stage; status: string; window_lo: number | null; window_hi: number | null }>;
  disclaimer: string;
}

interface DashboardResponse {
  household: Household;
  stages: Record<Stage, {
    label: string;
    condition: DashboardStage["condition"];
    sensor_health: string;
    reliability: string;
    status: ForecastStatus;
    point_days: number | null;
    interval_days: number[] | null;
    reason: string | null;
    recommendation: string;
  }>;
  open_alerts: number;
  upcoming_maintenance: Dashboard["upcoming_maintenance"];
  disclaimer: string;
}

export interface Alert {
  id: number;
  stage: Stage;
  severity: string;
  title: string;
  status: string;
  day: number;
  details?: Record<string, unknown>;
}

interface AlertResponse extends Omit<Alert, "day"> {
  last_day?: number;
}

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const apiKey = process.env.NEXT_PUBLIC_API_KEY;
  const response = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(apiKey ? { "x-api-key": apiKey } : {}),
      ...(init?.headers || {}),
    },
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `AquaPredict request failed (${response.status})`);
  }
  return response.json();
}

export const api = {
  households: {
    list: () => request<Household[]>("/households"),
    dashboard: async (id: string): Promise<Dashboard> => {
      const response = await request<DashboardResponse>(`/households/${id}/dashboard`);
      return {
        ...response,
        stages: Object.fromEntries(Object.entries(response.stages).map(([stage, item]) => [
          stage,
          {
            label: item.label,
            condition: item.condition,
            sensor_health: item.sensor_health,
            forecast: {
              stage_label: item.label,
              status: item.status,
              point_days: item.point_days,
              interval_days: item.interval_days,
              reliability: item.reliability,
              reason: item.reason,
              recommendation: item.recommendation,
            },
          },
        ])) as Record<Stage, DashboardStage>,
      };
    },
    forecast: (id: string, stage: Stage) => request<Forecast>(`/households/${id}/forecast?stage=${stage}&explain=true`),
    createDemo: () => request<Household>("/households", {
      method: "POST",
      body: JSON.stringify({ name: "My demo home", scenario: "normal", law: "A", seed: 7, now_day: 60, calibrated: true, overrides: {}, user_faults: [] }),
    }),
  },
  alerts: {
    list: async (id: string): Promise<Alert[]> => (await request<AlertResponse[]>(`/alerts?household_id=${id}`)).map((alert) => ({
      ...alert,
      day: alert.last_day ?? 0,
    })),
    action: (id: number, action: "acknowledge" | "resolve" | "dismiss") =>
      request<Alert>(`/alerts/${id}/${action}`, { method: "POST", body: JSON.stringify({ note: null }) }),
  },
  validation: {
    run: (n = 4) => request<Record<string, unknown>>("/validation/run", {
      method: "POST",
      body: JSON.stringify({ n_households: n, seed: Date.now() % 100000, methods: ["trend", "volume_rule", "health_20pct"], calibrated: true }),
    }),
  },
};
