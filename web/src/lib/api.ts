// Typed client for the FastAPI service (src/api/main.py).
// Every shape here mirrors a real endpoint; nothing is mocked.

const BASE = (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, "") ?? "/api";

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, {
      ...init,
      headers: init?.body ? { "Content-Type": "application/json" } : undefined,
    });
  } catch {
    throw new ApiError(0, `Can't reach the Zenith API at ${BASE}. Start it with "uvicorn src.api.main:app" or "docker compose up".`);
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, `${path} failed (${res.status}): ${detail}`);
  }
  return res.json() as Promise<T>;
}

const post = <T>(path: string, body: unknown) =>
  request<T>(path, { method: "POST", body: JSON.stringify(body) });

// ── Shapes ─────────────────────────────────────────────────────

export interface Health {
  status: string;
  model_loaded: boolean;
  timestamp: string;
  version: string;
}

export interface Plant {
  plant_id: string;
  name: string;
  location: {
    name: string;
    state: string;
    latitude: number;
    longitude: number;
    elevation_m: number;
    timezone: string;
  };
  ac_capacity_mw: number;
  regulatory: {
    dsm_ruleset_id: string | null;
    seller_category: string | null;
    contract_rate_rs_per_kwh: number | null;
    transaction_type: string | null;
  };
  simulated: boolean;
}

export interface WeatherRow {
  timestamp: string;
  shortwave_radiation: number;
  cloud_cover: number;
  temperature_2m: number;
  relative_humidity_2m: number;
  wind_speed_10m: number;
}

export interface Forecast {
  forecast_hours: number;
  timestamps: string[];
  predictions_mw: number[];
  peak_output_mw: number;
  peak_index: number;
  peak_timestamp: string;
  total_generation_mwh: number;
  interval_hours: number;
  generated_at: string;
  predictions_p10_mw: number[];
  predictions_p50_mw: number[];
  predictions_p90_mw: number[];
}

export interface OptimizeRow {
  solar_mw: number;
  declared_schedule_mw: number;
  surplus_mw: number;
  charge_mw: number;
  discharge_mw: number;
  battery_level_mwh: number;
  grid_balance_mw: number;
  deviation_mwh: number;
  action: "CHARGE" | "DISCHARGE" | "HOLD";
  dsm_rs?: number;
  block_start?: string;
}

export interface OptimizeResult {
  status: string;
  blocks: number;
  schedule: OptimizeRow[];
  summary: {
    total_charged_mwh: number;
    total_discharged_mwh: number;
    total_deviation_mwh: number;
    total_dsm_rs?: number;
    blocks_charging: number;
    blocks_discharging: number;
    blocks_hold: number;
  };
}

export interface BatteryParams {
  battery_capacity_mwh: number;
  charge_rate_mw: number;
  discharge_rate_mw: number;
  initial_charge_mwh: number;
  round_trip_efficiency: number;
}

export const NO_BATTERY: BatteryParams = {
  battery_capacity_mwh: 0,
  charge_rate_mw: 0,
  discharge_rate_mw: 0,
  initial_charge_mwh: 0,
  round_trip_efficiency: 0.9,
};

export interface GateClosures {
  timestamp: string;
  plant_id: string;
  transaction_type: string;
  bilateral_revision: { allowed: boolean; effective_timestamp: string | null };
  real_time_market: { delivery_window_start: string; bid_window_open: string; gate_closure: string };
}

export interface RevisionResult {
  applied_schedule_mw: number[];
  effective_timestamp: string | null;
  revision_allowed: boolean;
  locked_block_count: number;
}

// The model card is a free-form JSON document; only the fields the Model
// page reads are typed.
export interface ModelCard {
  model_path: string;
  plant_id: string;
  target: string;
  git_sha?: string;
  feature_list: string[];
  data_window: { start: string; days: number; source: string };
  rolling_origin_backtest?: {
    n_origins: number;
    horizon_hours: number;
    averaged_nmae_pct: Record<string, number>;
    averaged_nrmse_pct: Record<string, number>;
  };
  final_holdout?: {
    period: { start: string; days: number };
    model_skill_nmae_pct: Record<string, number>;
    deliverable_skill_nmae_pct: Record<string, number>;
  };
  quantile_model?: Record<string, unknown>;
}

// GET /plants/{id} returns the plant's full YAML config (the same dict
// every API endpoint reads); the console only needs this flattened view.
export interface PlantConfig {
  plant_id: string;
  name: string;
  location: Plant["location"];
  capacity: { ac_capacity_mw: number };
  regulatory?: {
    dsm_ruleset_id?: string | null;
    seller_category?: string | null;
    contract_rate_rs_per_kwh?: number | null;
    revision_windows?: { transaction_type?: string | null } | null;
  } | null;
  data_provenance?: { simulated?: boolean } | null;
}

export function toPlant(cfg: PlantConfig): Plant {
  const reg = cfg.regulatory ?? {};
  return {
    plant_id: cfg.plant_id,
    name: cfg.name,
    location: cfg.location,
    ac_capacity_mw: cfg.capacity.ac_capacity_mw,
    regulatory: {
      dsm_ruleset_id: reg.dsm_ruleset_id ?? null,
      seller_category: reg.seller_category ?? null,
      contract_rate_rs_per_kwh: reg.contract_rate_rs_per_kwh ?? null,
      transaction_type: reg.revision_windows?.transaction_type ?? null,
    },
    // Prefer the config's explicit flag; fall back to the "(simulated)" name
    // convention for plants onboarded without a data_provenance block.
    simulated: cfg.data_provenance?.simulated ?? /simulated/i.test(cfg.name),
  };
}

// ── Endpoints ──────────────────────────────────────────────────

export const api = {
  base: BASE,
  health: () => request<Health>("/health"),
  plants: () => request<{ plant_ids: string[] }>("/plants"),
  plant: (id: string) => request<PlantConfig>(`/plants/${encodeURIComponent(id)}`).then(toPlant),
  modelCard: () => request<ModelCard>("/model-card"),
  forecast: (plantId: string, hours: WeatherRow[]) =>
    post<Forecast>("/forecast", { plant_id: plantId, hours }),
  optimize: (args: {
    plantId: string;
    date: string;
    solarMw: number[];
    declaredMw: number[];
    battery: BatteryParams;
  }) =>
    post<OptimizeResult>("/optimize", {
      plant_id: args.plantId,
      date: args.date,
      solar_forecast_mw: args.solarMw,
      declared_schedule_mw: args.declaredMw,
      ...args.battery,
    }),
  gateClosures: (plantId: string, timestamp: string) =>
    request<GateClosures>(
      `/schedule/gate-closures?plant_id=${encodeURIComponent(plantId)}&timestamp=${encodeURIComponent(timestamp)}`,
    ),
  revise: (args: { plantId: string; date: string; locked: number[]; proposed: number[]; requestTimestamp: string }) =>
    post<RevisionResult>("/schedule/revise", {
      plant_id: args.plantId,
      date: args.date,
      locked_schedule_mw: args.locked,
      proposed_schedule_mw: args.proposed,
      request_timestamp: args.requestTimestamp,
    }),
};
