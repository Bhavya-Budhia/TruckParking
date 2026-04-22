import math
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

import h3
import joblib
import numpy as np
import pandas as pd
from scgraph.geographs.us_freeway import us_freeway_geograph

path = r"C:\Users\bhavy\Massachusetts Institute of Technology\Truck Parking Capstone - General\Truck Stop Finder 🚚⛽\\"

DEFAULT_UTILITY_WEIGHTS: Dict[str, float] = {
    "parking": 0.30,
    "amenities": 0.20,
    "capacity": 0.15,
    "detour": 0.20,
    "traffic": 0.10,
    "remaining_route": 0.05,
}


@lru_cache(maxsize=1)
def load_reference_data():
    """Load stable reference files once per Python process."""
    cong_4_df = pd.read_csv("Congestion_speed_r_4.csv")
    cong_3_df = pd.read_csv("Congestion_speed_r_3.csv")
    cong_2_df = pd.read_csv("Congestion_speed_r_2.csv")

    truck_stop_df = pd.read_csv("stop_tab.csv")
    truck_stop_df = truck_stop_df[
        [
            "pin id", "pinname", "link_id", "lat", "lng",
            "truckParkingSpotCount", "amenities_score", "f_system", "route_num"
        ]
    ].copy()

    artifact_dir = Path("output_excel")
    model_bundle = joblib.load(artifact_dir / "parking_availability_model.joblib")
    obs_sorted = pd.read_parquet(artifact_dir / "parking_obs_sorted.parquet")

    return {
        "cong_4_df": cong_4_df,
        "cong_3_df": cong_3_df,
        "cong_2_df": cong_2_df,
        "truck_stop_df": truck_stop_df,
        "model_bundle": model_bundle,
        "obs_sorted": obs_sorted,
    }


@lru_cache(maxsize=20000)
def cached_shortest_path(origin_lat: float, origin_lon: float, dest_lat: float, dest_lon: float):
    return us_freeway_geograph.get_shortest_path(
        origin_node={"latitude": float(origin_lat), "longitude": float(origin_lon)},
        destination_node={"latitude": float(dest_lat), "longitude": float(dest_lon)},
        output_units="mi",
        cache=True,
    )


def attach_last_obs_before_inference(query_df: pd.DataFrame, obs_sorted: pd.DataFrame, time_col: str = "query_ts"):
    q = query_df.copy()
    q[time_col] = pd.to_datetime(q[time_col], utc=True, errors="coerce").astype("datetime64[ns, UTC]")
    q = q.dropna(subset=[time_col]).sort_values([time_col, "pin id"]).reset_index(drop=True)

    obs_sorted = obs_sorted.copy()
    obs_sorted["ts_utc"] = pd.to_datetime(obs_sorted["ts_utc"], utc=True, errors="coerce").astype("datetime64[ns, UTC]")
    obs_sorted = obs_sorted.dropna(subset=["ts_utc"]).sort_values(["ts_utc", "pin id"]).reset_index(drop=True)

    out = pd.merge_asof(
        q,
        obs_sorted.rename(columns={
            "ts_utc": "last_ts",
            "status_ord": "last_status_ord",
            "parking status": "last_status_txt",
        }),
        left_on=time_col,
        right_on="last_ts",
        by="pin id",
        direction="backward",
        allow_exact_matches=True,
    )

    out["time_since_last_obs_min"] = (out[time_col] - out["last_ts"]).dt.total_seconds() / 60

    stale_cutoff = pd.Timedelta("6h")
    too_stale = (out[time_col] - out["last_ts"]) > stale_cutoff
    out.loc[too_stale, ["last_status_ord", "last_status_txt", "last_ts"]] = pd.NA
    return out


def bearing_from_points(lat1, lon1, lat2, lon2):
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)

    return (math.degrees(math.atan2(x, y)) + 360) % 360


def bearing_to_compass(bearing):
    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return directions[round(bearing / 45) % 8]


def extract_route_direction_near_stop(path_output):
    path = path_output.get("coordinate_path")
    if path is None or len(path) < 2:
        return pd.Series([None, None])

    lat1, lon1 = path[-2]
    lat2, lon2 = path[-1]
    bearing = bearing_from_points(lat1, lon1, lat2, lon2)
    direction = bearing_to_compass(bearing)
    return pd.Series([bearing, direction])


def bearing_to_travel_dir(bearing):
    if pd.isna(bearing):
        return np.nan
    if bearing >= 315 or bearing < 45:
        return 1
    if bearing < 135:
        return 3
    if bearing < 225:
        return 5
    return 7


def minmax_score(series, reverse: bool = False):
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(0.5, index=series.index)

    s_min = s.min()
    s_max = s.max()
    if pd.isna(s_min) or pd.isna(s_max) or s_max == s_min:
        out = pd.Series(1.0, index=series.index)
    else:
        out = (s - s_min) / (s_max - s_min)

    if reverse:
        out = 1 - out
    return out.fillna(0.5)


def normalize_weights(weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    merged = DEFAULT_UTILITY_WEIGHTS.copy()
    if weights:
        merged.update(weights)

    total = sum(max(float(v), 0.0) for v in merged.values())
    if total <= 0:
        return DEFAULT_UTILITY_WEIGHTS.copy()

    return {k: max(float(v), 0.0) / total for k, v in merged.items()}


def model_engine_func(
        driver_lat=25.7752,
        driver_lon=-80.2086,
        dest_lat=33.76052,
        dest_lon=-78.91463,
        hos_left_hr=5.0,
        freeflow_mph=55.0,
        start_time="2023-12-02 12:20:00",
        amenity_weight: Optional[float] = None,
        utility_weights: Optional[Dict[str, float]] = None,
        scenario_id: Optional[str] = None,
):
    refs = load_reference_data()
    cong_4_df = refs["cong_4_df"].copy()
    cong_3_df = refs["cong_3_df"].copy()
    cong_2_df = refs["cong_2_df"].copy()
    truck_stop_df = refs["truck_stop_df"].copy()

    model_bundle = refs["model_bundle"]
    parking_model = model_bundle["model"]
    threshold_full = model_bundle["threshold_full"]
    obs_sorted = refs["obs_sorted"]

    weights = normalize_weights(utility_weights)
    if amenity_weight is not None:
        weights["amenities"] = max(float(amenity_weight), 0.0)
        weights = normalize_weights(weights)

    truck_stop_df["driver_lat"] = float(driver_lat)
    truck_stop_df["driver_lon"] = float(driver_lon)
    truck_stop_df["start_time"] = pd.Timestamp(start_time, tz="UTC")
    truck_stop_df["dest_lat"] = float(dest_lat)
    truck_stop_df["dest_lon"] = float(dest_lon)
    truck_stop_df["HOS_left_hr"] = float(hos_left_hr)
    truck_stop_df["freeflow_mph"] = float(freeflow_mph)

    truck_stop_df["driver_stop_path"] = truck_stop_df.apply(
        lambda row: cached_shortest_path(row["driver_lat"], row["driver_lon"], row["lat"], row["lng"]),
        axis=1,
    )
    truck_stop_df["truck_stop_mi"] = truck_stop_df["driver_stop_path"].apply(lambda x: x["length"])
    truck_stop_df[["bearing_driver_to_stop_route", "dir_driver_to_stop_route"]] = (
        truck_stop_df["driver_stop_path"].apply(extract_route_direction_near_stop)
    )

    truck_stop_df["stop_dest_path"] = truck_stop_df.apply(
        lambda row: cached_shortest_path(row["lat"], row["lng"], row["dest_lat"], row["dest_lon"]),
        axis=1,
    )
    truck_stop_df["stop_dest_mi"] = truck_stop_df["stop_dest_path"].apply(lambda x: x["length"])

    driver_dest_path = cached_shortest_path(float(driver_lat), float(driver_lon), float(dest_lat), float(dest_lon))
    truck_dest_mi = driver_dest_path["length"]
    truck_stop_df["truck_dest_mi"] = truck_dest_mi

    truck_stop_df["truck_stop_after_dest"] = truck_stop_df["truck_stop_mi"] > truck_stop_df["truck_dest_mi"]
    truck_stop_df["ETA_stop"] = truck_stop_df["start_time"] + pd.to_timedelta(
        truck_stop_df["truck_stop_mi"] / truck_stop_df["freeflow_mph"], unit="h"
    )
    truck_stop_df["day_of_week"] = truck_stop_df["ETA_stop"].dt.weekday + 1
    truck_stop_df["hour_24"] = "hour_" + truck_stop_df["ETA_stop"].dt.hour.astype(str).str.zfill(2)
    truck_stop_df = truck_stop_df.drop(columns=["driver_stop_path", "stop_dest_path"])

    truck_stop_df["travel_dir"] = truck_stop_df["bearing_driver_to_stop_route"].apply(bearing_to_travel_dir)
    truck_stop_df["pol_2"] = truck_stop_df.apply(lambda r: h3.latlng_to_cell(r["lat"], r["lng"], 2), axis=1)
    truck_stop_df["pol_3"] = truck_stop_df.apply(lambda r: h3.latlng_to_cell(r["lat"], r["lng"], 3), axis=1)
    truck_stop_df["pol_4"] = truck_stop_df.apply(lambda r: h3.latlng_to_cell(r["lat"], r["lng"], 4), axis=1)

    cong_4_df = cong_4_df.groupby(["polygon", "travel_dir", "day_of_week", "hours"]).agg(
        {"avg_traffic": "mean"}).reset_index()
    cong_4_df.rename(columns={"avg_traffic": "traffic_h4"}, inplace=True)
    truck_stop_df = pd.merge(
        truck_stop_df,
        cong_4_df,
        left_on=["pol_4", "travel_dir", "day_of_week", "hour_24"],
        right_on=["polygon", "travel_dir", "day_of_week", "hours"],
        how="left",
    )

    cong_3_df = cong_3_df.groupby(["polygon", "travel_dir", "day_of_week", "hours"]).agg(
        {"avg_traffic": "mean"}).reset_index()
    cong_3_df.rename(columns={"avg_traffic": "traffic_h3"}, inplace=True)
    truck_stop_df = pd.merge(
        truck_stop_df,
        cong_3_df,
        left_on=["pol_3", "travel_dir", "day_of_week", "hour_24"],
        right_on=["polygon", "travel_dir", "day_of_week", "hours"],
        how="left",
    )

    cong_2_df = cong_2_df.groupby(["polygon", "travel_dir", "day_of_week", "hours"]).agg(
        {"avg_traffic": "mean"}).reset_index()
    cong_2_df.rename(columns={"avg_traffic": "traffic_h2"}, inplace=True)
    truck_stop_df = pd.merge(
        truck_stop_df,
        cong_2_df,
        left_on=["pol_2", "travel_dir", "day_of_week", "hour_24"],
        right_on=["polygon", "travel_dir", "day_of_week", "hours"],
        how="left",
    )

    truck_stop_df["traffic"] = np.where(truck_stop_df["traffic_h4"].isnull(), truck_stop_df["traffic_h3"],
                                        truck_stop_df["traffic_h4"])
    truck_stop_df["traffic"] = np.where(truck_stop_df["traffic"].isnull(), truck_stop_df["traffic_h2"],
                                        truck_stop_df["traffic"])

    p90 = truck_stop_df["traffic"].quantile(0.90)
    if pd.isna(p90) or p90 == 0:
        p90 = 1.0

    truck_stop_df["traffic_factor"] = 1 - (truck_stop_df["traffic"] / p90) * 0.5
    truck_stop_df["traffic_factor"] = truck_stop_df["traffic_factor"].clip(lower=0.3, upper=1.0)

    truck_stop_df["adj_speed_mph"] = truck_stop_df["freeflow_mph"]
    truck_stop_df["ETA_stop_adj"] = truck_stop_df["start_time"] + pd.to_timedelta(
        truck_stop_df["truck_stop_mi"] / truck_stop_df["adj_speed_mph"], unit="h"
    )

    truck_stop_df["eta_hour"] = truck_stop_df["ETA_stop_adj"].dt.hour
    truck_stop_df["eta_day_of_week"] = truck_stop_df["ETA_stop_adj"].dt.dayofweek
    truck_stop_df["eta_month"] = truck_stop_df["ETA_stop_adj"].dt.month

    if "route_num" not in truck_stop_df.columns:
        truck_stop_df["route_num"] = "Unknown"

    parking_query = truck_stop_df[["pin id", "start_time"]].copy().rename(columns={"start_time": "query_ts"})
    parking_feat = attach_last_obs_before_inference(parking_query, obs_sorted=obs_sorted, time_col="query_ts")
    truck_stop_df["last_status_ord"] = parking_feat["last_status_ord"].values
    truck_stop_df["time_since_last_obs_min"] = parking_feat["time_since_last_obs_min"].values

    feature_cols = [
        "last_status_ord",
        "time_since_last_obs_min",
        "eta_hour",
        "eta_day_of_week",
        "eta_month",
        "route_num",
    ]

    X_pred = truck_stop_df[feature_cols].copy()
    truck_stop_df["p_full"] = parking_model.predict_proba(X_pred)[:, 1]
    truck_stop_df["parking_available_pred"] = (truck_stop_df["p_full"] < threshold_full).astype(int)
    truck_stop_df["p_available"] = 1 - truck_stop_df["p_full"]

    truck_stop_df["min_dist"] = truck_stop_df["truck_stop_mi"].min()
    truck_stop_df["extra_dist"] = truck_stop_df["truck_stop_mi"] - truck_stop_df["min_dist"]
    truck_stop_df["time_to_stop_hr_adj"] = truck_stop_df["truck_stop_mi"] / truck_stop_df["adj_speed_mph"]
    truck_stop_df["HOS_exceeded"] = (truck_stop_df["time_to_stop_hr_adj"] > truck_stop_df["HOS_left_hr"]).astype(int)
    truck_stop_df["detour_mi"] = truck_stop_df["truck_stop_mi"] + truck_stop_df["stop_dest_mi"] - truck_stop_df[
        "truck_dest_mi"]

    truck_stop_df["feasible_stop"] = (
            (truck_stop_df["HOS_exceeded"] == 0)
            & (truck_stop_df["truck_stop_after_dest"] == 0)
            & (truck_stop_df["p_available"] >= 0.40)
    ).astype(int)

    truck_stop_df["stop_dest_norm"] = minmax_score(truck_stop_df["stop_dest_mi"], reverse=True)
    truck_stop_df["score_parking"] = truck_stop_df["p_available"].fillna(0.5)
    truck_stop_df["score_amenities"] = minmax_score(truck_stop_df["amenities_score"])
    truck_stop_df["score_capacity"] = minmax_score(truck_stop_df["truckParkingSpotCount"])
    truck_stop_df["score_detour"] = minmax_score(truck_stop_df["detour_mi"], reverse=True)
    truck_stop_df["score_traffic"] = minmax_score(truck_stop_df["traffic"], reverse=True)

    truck_stop_df["utility_score"] = (
            weights["parking"] * truck_stop_df["score_parking"]
            + weights["amenities"] * truck_stop_df["score_amenities"]
            + weights["capacity"] * truck_stop_df["score_capacity"]
            + weights["detour"] * truck_stop_df["score_detour"]
            + weights["traffic"] * truck_stop_df["score_traffic"]
            + weights["remaining_route"] * truck_stop_df["stop_dest_norm"]
    )

    truck_stop_df["scenario_utility"] = np.where(truck_stop_df["feasible_stop"] == 1, truck_stop_df["utility_score"],
                                                 0.0)
    truck_stop_df["utility_score"] = np.where(truck_stop_df["feasible_stop"] == 1, truck_stop_df["utility_score"],
                                              -999.0)

    truck_stop_df = truck_stop_df.sort_values(["utility_score", "p_available", "truckParkingSpotCount"],
                                              ascending=False)
    truck_stop_df["utility_rank"] = range(1, len(truck_stop_df) + 1)

    truck_stop_df["utility_weight_parking"] = weights["parking"]
    truck_stop_df["utility_weight_amenities"] = weights["amenities"]
    truck_stop_df["utility_weight_capacity"] = weights["capacity"]
    truck_stop_df["utility_weight_detour"] = weights["detour"]
    truck_stop_df["utility_weight_traffic"] = weights["traffic"]
    truck_stop_df["utility_weight_remaining_route"] = weights["remaining_route"]
    if scenario_id is not None:
        truck_stop_df["scenario_id"] = str(scenario_id)

    return truck_stop_df
