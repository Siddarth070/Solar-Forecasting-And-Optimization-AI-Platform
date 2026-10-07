import { useEffect, useState } from "react";
import { api, type Forecast, type Plant } from "./api";
import { clearSkyDay } from "./blocks";
import { fetchDayWeather, type DayWeather } from "./weather";

export type WeatherSource = "open-meteo" | "demo";

export interface DayForecastState {
  status: "idle" | "loading" | "ready" | "error";
  plant: Plant | null;
  weather: DayWeather | null;
  forecast: Forecast | null;
  error: string | null;
  step: string;
}

const initial: DayForecastState = { status: "idle", plant: null, weather: null, forecast: null, error: null, step: "" };

/**
 * Plant config → weather for the date → POST /forecast, as one unit.
 * `reload` bumps a counter so the caller can retry after an error.
 */
export function useDayForecast(plantId: string | null, date: string, source: WeatherSource, reload = 0) {
  const [state, setState] = useState<DayForecastState>(initial);

  useEffect(() => {
    if (!plantId) return;
    let cancelled = false;
    const set = (s: Partial<DayForecastState>) => !cancelled && setState((prev) => ({ ...prev, ...s }));

    (async () => {
      set({ status: "loading", error: null, step: "Loading plant" });
      try {
        const plant = await api.plant(plantId);
        set({ plant, step: source === "demo" ? "Building clear-sky demo weather" : "Fetching weather from Open-Meteo" });
        const weather: DayWeather =
          source === "demo"
            ? {
                rows: clearSkyDay(date, plant.location.latitude, plant.location.longitude),
                quality: { source: "clear-sky demo", missingFilled: 0, clipped: 0, notes: [] },
              }
            : await fetchDayWeather(date, plant.location.latitude, plant.location.longitude);
        set({ weather, step: "Running the forecast model" });
        const forecast = await api.forecast(plantId, weather.rows);
        set({ forecast, status: "ready", step: "" });
      } catch (e) {
        set({ status: "error", error: e instanceof Error ? e.message : String(e), forecast: null, step: "" });
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [plantId, date, source, reload]);

  return state;
}
