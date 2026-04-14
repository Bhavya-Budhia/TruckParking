from typing import Dict, Tuple

import numpy as np
import pandas as pd

from model_engine_v2 import model_engine_func, minmax_score

TIME_WINDOWS = {
    "morning": "08:00:00",
    "afternoon": "14:00:00",
    "evening": "20:00:00",
}


def jitter_point(lat: float, lon: float, rng: np.random.Generator, max_shift_deg: float) -> Tuple[float, float]:
    return (
        float(lat + rng.uniform(-max_shift_deg, max_shift_deg)),
        float(lon + rng.uniform(-max_shift_deg, max_shift_deg)),
    )


def build_start_time(base_date: str, time_name: str) -> str:
    date_part = str(pd.Timestamp(base_date).date())
    return f"{date_part} {TIME_WINDOWS[time_name]}"


def sample_hos(base_hos: float, rng: np.random.Generator, hos_jitter_hr: float) -> float:
    hos = base_hos + rng.uniform(-hos_jitter_hr, hos_jitter_hr)
    return max(0.5, float(round(hos, 2)))


def sample_amenity_weight(base_weight: float, rng: np.random.Generator, amenity_jitter: float) -> float:
    wt = base_weight + rng.uniform(-amenity_jitter, amenity_jitter)
    return min(0.60, max(0.05, float(round(wt, 3))))


def run_simulation(
        driver_lat: float,
        driver_lon: float,
        dest_lat: float,
        dest_lon: float,
        num_runs: int,
        base_hos_left_hr: float,
        freeflow_mph: float,
        base_date: str,
        base_amenity_weight: float = 0.20,
        location_jitter_deg: float = 0.20,
        hos_jitter_hr: float = 1.50,
        amenity_jitter: float = 0.10,
        seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    scenario_frames = []

    for run_id in range(1, int(num_runs) + 1):
        run_driver_lat, run_driver_lon = jitter_point(driver_lat, driver_lon, rng, location_jitter_deg)
        run_dest_lat, run_dest_lon = jitter_point(dest_lat, dest_lon, rng, location_jitter_deg)
        run_hos = sample_hos(base_hos_left_hr, rng, hos_jitter_hr)
        run_amenity_weight = sample_amenity_weight(base_amenity_weight, rng, amenity_jitter)

        for time_name in ["morning", "afternoon", "evening"]:
            scenario_id = f"run_{run_id}_{time_name}"
            start_time = build_start_time(base_date, time_name)

            df = model_engine_func(
                driver_lat=run_driver_lat,
                driver_lon=run_driver_lon,
                dest_lat=run_dest_lat,
                dest_lon=run_dest_lon,
                hos_left_hr=run_hos,
                freeflow_mph=freeflow_mph,
                start_time=start_time,
                amenity_weight=run_amenity_weight,
                scenario_id=scenario_id,
            ).copy()

            df["run_id"] = run_id
            df["time_window"] = time_name
            df["sim_driver_lat"] = run_driver_lat
            df["sim_driver_lon"] = run_driver_lon
            df["sim_dest_lat"] = run_dest_lat
            df["sim_dest_lon"] = run_dest_lon
            df["sim_hos_left_hr"] = run_hos
            df["sim_amenity_weight"] = run_amenity_weight
            scenario_frames.append(df)

    scenario_df = pd.concat(scenario_frames, ignore_index=True)

    scenario_df["is_top_10"] = (scenario_df["utility_rank"] <= 10).astype(int)
    scenario_df["scenario_score_for_agg"] = np.where(
        scenario_df["feasible_stop"] == 1,
        scenario_df["scenario_utility"],
        0.0,
    )

    summary = (
        scenario_df.groupby(["pin id", "pinname", "lat", "lng"], as_index=False)
        .agg(
            avg_scenario_utility=("scenario_score_for_agg", "mean"),
            p10_scenario_utility=("scenario_score_for_agg", lambda s: float(np.quantile(s, 0.10))),
            feasible_rate=("feasible_stop", "mean"),
            top_10_rate=("is_top_10", "mean"),
            avg_p_available=("p_available", "mean"),
            avg_detour_mi=("detour_mi", "mean"),
            avg_truck_stop_mi=("truck_stop_mi", "mean"),
            avg_stop_dest_mi=("stop_dest_mi", "mean"),
            avg_amenities_score=("amenities_score", "mean"),
            avg_capacity=("truckParkingSpotCount", "mean"),
            scenario_count=("scenario_id", "nunique"),
        )
    )

    summary["score_avg_utility"] = minmax_score(summary["avg_scenario_utility"])
    summary["score_worst_case"] = minmax_score(summary["p10_scenario_utility"])
    summary["score_feasible_rate"] = summary["feasible_rate"].fillna(0)
    summary["score_top_10_rate"] = summary["top_10_rate"].fillna(0)

    summary["combined_utility"] = (
            0.50 * summary["score_avg_utility"]
            + 0.20 * summary["score_feasible_rate"]
            + 0.15 * summary["score_top_10_rate"]
            + 0.15 * summary["score_worst_case"]
    )

    summary = summary.sort_values(
        ["combined_utility", "feasible_rate", "avg_p_available"],
        ascending=False,
    ).reset_index(drop=True)
    summary["simulation_rank"] = np.arange(1, len(summary) + 1)

    return summary, scenario_df
