import math
from pathlib import Path

import h3
import joblib
import numpy as np
import pandas as pd
from scgraph.geographs.us_freeway import us_freeway_geograph

from model_stops import model_stop_func

path = r"C:\Users\bhavy\Massachusetts Institute of Technology\Truck Parking Capstone - General\Truck Stop Finder 🚚⛽\\"
# path = r"C:\Users\samcl\Massachusetts Institute of Technology\Truck Parking Capstone - Truck Stop Finder 🚚⛽\\"

# Sourced directly from TruckerPath
cong_4_df = pd.read_csv(
    path + r"5. Source & Refrence Files\Congestion_speed_r_4.csv")
cong_3_df = pd.read_csv(
    path + r"5. Source & Refrence Files\Congestion_speed_r_3.csv")
cong_2_df = pd.read_csv(
    path + r"5. Source & Refrence Files\Congestion_speed_r_2.csv")


def attach_last_obs_before_inference(query_df, obs_sorted, time_col="query_ts"):
    q = query_df.copy()
    q[time_col] = pd.to_datetime(q[time_col], utc=True, errors="coerce").astype("datetime64[ns, UTC]")
    q = q.dropna(subset=[time_col]).sort_values([time_col, "pin id"]).reset_index(drop=True)

    obs_sorted = obs_sorted.copy()
    obs_sorted["ts_utc"] = pd.to_datetime(obs_sorted["ts_utc"], utc=True, errors="coerce").astype("datetime64[ns, UTC]")
    # obs_sorted["ts_utc"] = pd.to_datetime(obs_sorted["ts_utc"], utc=True, errors="coerce")
    obs_sorted = obs_sorted.dropna(subset=["ts_utc"]).sort_values(["ts_utc", "pin id"]).reset_index(drop=True)

    out = pd.merge_asof(
        q,
        obs_sorted.rename(
            columns={
                "ts_utc": "last_ts",
                "status_ord": "last_status_ord",
                "parking status": "last_status_txt"
            }
        ),
        left_on=time_col,
        right_on="last_ts",
        by="pin id",
        direction="backward",
        allow_exact_matches=True
    )

    out["time_since_last_obs_min"] = (
                                             out[time_col] - out["last_ts"]
                                     ).dt.total_seconds() / 60

    # same staleness rule as training
    stale_cutoff = pd.Timedelta("6h")
    too_stale = (out[time_col] - out["last_ts"]) > stale_cutoff
    out.loc[too_stale, ["last_status_ord", "last_status_txt", "last_ts"]] = pd.NA

    return out

def get_shortest_path_output(row, origin_lat, origin_lon, dest_lat, dest_lon):
    return us_freeway_geograph.get_shortest_path(
        origin_node={
            "latitude": float(row[origin_lat]),
            "longitude": float(row[origin_lon])
        },
        destination_node={
            "latitude": float(row[dest_lat]),
            "longitude": float(row[dest_lon])
        },
        output_units="mi",
        cache=False,
    )


def bearing_from_points(lat1, lon1, lat2, lon2):
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    dlon = lon2 - lon1

    x = math.sin(dlon) * math.cos(lat2)
    y = (
            math.cos(lat1) * math.sin(lat2)
            - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    )

    bearing = (math.degrees(math.atan2(x, y)) + 360) % 360
    return bearing


def bearing_to_compass(bearing):
    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return directions[round(bearing / 45) % 8]


def extract_route_direction_near_stop(path_output):
    path = path_output.get("coordinate_path")

    if path is None or len(path) < 2:
        return pd.Series([None, None])

    # use the last two points, so direction is near the stop / destination end
    lat1, lon1 = path[-2]
    lat2, lon2 = path[-1]

    bearing = bearing_from_points(lat1, lon1, lat2, lon2)
    direction = bearing_to_compass(bearing)

    return pd.Series([bearing, direction])


truck_stop_df, traffic_df = model_stop_func()

truck_stop_df = truck_stop_df[
    [
        "pin id", "pinname", "link_id", "lat", "lng",
        "truckParkingSpotCount", "amenities_score", "f_system", "route_num"
    ]
].copy()

freeflow_mph_dict = {
    1: 70,
    2: 60,
    3: 65
}

truck_stop_df["driver_lat"] = 28.509696
truck_stop_df["driver_lon"] = -81.737782
truck_stop_df["start_time"] = pd.Timestamp("2023-12-02 12:20:00", tz="UTC")
truck_stop_df["dest_lat"] = 27.95
truck_stop_df["dest_lon"] = -82.46
truck_stop_df["HOS_left_hr"] = 5

# Either use mapped speed:
# truck_stop_df["freeflow_mph"] = truck_stop_df["f_system"].map(freeflow_mph_dict)

# Or fixed speed:
truck_stop_df["freeflow_mph"] = 55

# ---------------------------------------------------
# 1. Driver -> Stop path output
# ---------------------------------------------------
truck_stop_df["driver_stop_path"] = truck_stop_df.apply(
    get_shortest_path_output,
    axis=1,
    args=("driver_lat", "driver_lon", "lat", "lng")
)

truck_stop_df["truck_stop_mi"] = truck_stop_df["driver_stop_path"].apply(lambda x: x["length"])

truck_stop_df[["bearing_driver_to_stop_route", "dir_driver_to_stop_route"]] = (
    truck_stop_df["driver_stop_path"].apply(extract_route_direction_near_stop)
)

# ---------------------------------------------------
# 2. Stop -> Destination path output
# ---------------------------------------------------
truck_stop_df["stop_dest_path"] = truck_stop_df.apply(
    get_shortest_path_output,
    axis=1,
    args=("lat", "lng", "dest_lat", "dest_lon")
)

truck_stop_df["stop_dest_mi"] = truck_stop_df["stop_dest_path"].apply(lambda x: x["length"])

# ---------------------------------------------------
# 3. Driver -> Destination path output
# ---------------------------------------------------
truck_stop_df["driver_dest_path"] = truck_stop_df.apply(
    get_shortest_path_output,
    axis=1,
    args=("driver_lat", "driver_lon", "dest_lat", "dest_lon")
)

truck_stop_df["truck_dest_mi"] = truck_stop_df["driver_dest_path"].apply(lambda x: x["length"])

# ---------------------------------------------------
# 4. Other derived columns
# ---------------------------------------------------
truck_stop_df["truck_stop_after_dest"] = truck_stop_df["truck_stop_mi"] > truck_stop_df["truck_dest_mi"]

truck_stop_df["ETA_stop"] = truck_stop_df["start_time"] + pd.to_timedelta(
    truck_stop_df["truck_stop_mi"] / truck_stop_df["freeflow_mph"],
    unit="h"
)

truck_stop_df["day_of_week"] = truck_stop_df["ETA_stop"].dt.weekday + 1
truck_stop_df["hour_24"] = "hour_" + truck_stop_df["ETA_stop"].dt.hour.astype(str).str.zfill(2)

# Optional: remove heavy path objects before saving
truck_stop_df = truck_stop_df.drop(columns=["driver_stop_path", "stop_dest_path", "driver_dest_path"])


def bearing_to_travel_dir(bearing):
    if bearing >= 315 or bearing < 45:
        return 1  # North
    elif bearing < 135:
        return 3  # East
    elif bearing < 225:
        return 5  # South
    else:
        return 7  # West


truck_stop_df["travel_dir"] = truck_stop_df["bearing_driver_to_stop_route"].apply(bearing_to_travel_dir)

truck_stop_df["pol_2"] = truck_stop_df.apply(
    lambda r: h3.latlng_to_cell(r["lat"], r["lng"], 2),
    axis=1
)

truck_stop_df["pol_3"] = truck_stop_df.apply(
    lambda r: h3.latlng_to_cell(r["lat"], r["lng"], 3),
    axis=1
)

truck_stop_df["pol_4"] = truck_stop_df.apply(
    lambda r: h3.latlng_to_cell(r["lat"], r["lng"], 4),
    axis=1
)

cong_4_df = cong_4_df.groupby(["polygon", "travel_dir", "day_of_week", "hours"]).agg(
    {"avg_traffic": "mean"}).reset_index()
cong_4_df.rename(columns={"avg_traffic": "traffic_h4"}, inplace=True)

a = truck_stop_df.shape[0]
truck_stop_df = pd.merge(truck_stop_df, cong_4_df, left_on=["pol_4", "travel_dir", "day_of_week", "hour_24"],
                         right_on=["polygon", "travel_dir", "day_of_week", "hours"], how="left")
b = truck_stop_df.shape[0]

assert a == b, f"a ({a}) is not equal to b ({b})"

cong_3_df = cong_3_df.groupby(["polygon", "travel_dir", "day_of_week", "hours"]).agg(
    {"avg_traffic": "mean"}).reset_index()
cong_3_df.rename(columns={"avg_traffic": "traffic_h3"}, inplace=True)

a = truck_stop_df.shape[0]
truck_stop_df = pd.merge(truck_stop_df, cong_3_df, left_on=["pol_3", "travel_dir", "day_of_week", "hour_24"],
                         right_on=["polygon", "travel_dir", "day_of_week", "hours"], how="left")
b = truck_stop_df.shape[0]

assert a == b, f"a ({a}) is not equal to b ({b})"

cong_2_df = cong_2_df.groupby(["polygon", "travel_dir", "day_of_week", "hours"]).agg(
    {"avg_traffic": "mean"}).reset_index()
cong_2_df.rename(columns={"avg_traffic": "traffic_h2"}, inplace=True)

a = truck_stop_df.shape[0]
truck_stop_df = pd.merge(truck_stop_df, cong_2_df, left_on=["pol_2", "travel_dir", "day_of_week", "hour_24"],
                         right_on=["polygon", "travel_dir", "day_of_week", "hours"], how="left")
b = truck_stop_df.shape[0]

assert a == b, f"a ({a}) is not equal to b ({b})"

truck_stop_df["traffic"] = np.where(truck_stop_df["traffic_h4"].isnull(), truck_stop_df["traffic_h3"],
                                    truck_stop_df["traffic_h4"])
truck_stop_df["traffic"] = np.where(truck_stop_df["traffic"].isnull(), truck_stop_df["traffic_h2"],
                                    truck_stop_df["traffic"])

# TODO: Change this later
p90 = truck_stop_df["traffic"].quantile(0.90)

truck_stop_df["traffic_factor"] = 1 - (truck_stop_df["traffic"] / p90) * 0.5
truck_stop_df["traffic_factor"] = truck_stop_df["traffic_factor"].clip(lower=0.3, upper=1.0)

truck_stop_df["adj_speed_mph"] = truck_stop_df["freeflow_mph"] * truck_stop_df["traffic_factor"]

truck_stop_df["ETA_stop_adj"] = truck_stop_df["start_time"] + pd.to_timedelta(
    truck_stop_df["truck_stop_mi"] / truck_stop_df["adj_speed_mph"],
    unit="h"
)

truck_stop_df["day_of_week_adj"] = truck_stop_df["ETA_stop_adj"].dt.weekday + 1
truck_stop_df["hour_24_adj"] = "hour_" + truck_stop_df["ETA_stop_adj"].dt.hour.astype(str).str.zfill(2)

# ----------------------------
# Build parking-model features
# ----------------------------

# training used Monday=0 for eta_day_of_week
truck_stop_df["eta_hour"] = truck_stop_df["ETA_stop_adj"].dt.hour
truck_stop_df["eta_day_of_week"] = truck_stop_df["ETA_stop_adj"].dt.dayofweek
truck_stop_df["eta_month"] = truck_stop_df["ETA_stop_adj"].dt.month

# if route_num is not present in model_engine, create a fallback
if "route_num" not in truck_stop_df.columns:
    truck_stop_df["route_num"] = "Unknown"

artifact_dir = Path("output_excel")

model_bundle = joblib.load(artifact_dir / "parking_availability_model.joblib")
parking_model = model_bundle["model"]
threshold_full = model_bundle["threshold_full"]

obs_sorted = pd.read_parquet(artifact_dir / "parking_obs_sorted.parquet")

parking_query = truck_stop_df[["pin id", "start_time"]].copy()
parking_query = parking_query.rename(columns={"start_time": "query_ts"})

parking_feat = attach_last_obs_before_inference(
    parking_query,
    obs_sorted=obs_sorted,
    time_col="query_ts"
)

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
truck_stop_df["parking_available_pred"] = (
        truck_stop_df["p_full"] < threshold_full
).astype(int)

truck_stop_df["p_available"] = 1 - truck_stop_df["p_full"]

truck_stop_df["min_dist"] = truck_stop_df["truck_stop_mi"].min()
truck_stop_df["extra_dist"] = truck_stop_df["truck_stop_mi"] - truck_stop_df["min_dist"]

truck_stop_df.to_csv("1_with_parking_predictions.csv", index=False)


print(truck_stop_df.columns)
