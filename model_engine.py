import math

import pandas as pd
from scgraph.geographs.us_freeway import us_freeway_geograph

from model_stops import model_stop_func


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


def extract_route_direction(path_output):
    path = path_output.get("coordinate_path")

    if path is None or len(path) < 2:
        return pd.Series([None, None])

    lat1, lon1 = path[0]
    lat2, lon2 = path[1]

    bearing = bearing_from_points(lat1, lon1, lat2, lon2)
    direction = bearing_to_compass(bearing)

    return pd.Series([bearing, direction])


truck_stop_df, traffic_df = model_stop_func()

truck_stop_df = truck_stop_df[
    [
        "pin id", "pinname", "link_id", "lat", "lng",
        "truckParkingSpotCount", "amenities_score", "f_system"
    ]
].copy()

freeflow_mph_dict = {
    1: 70,
    2: 60,
    3: 65
}


truck_stop_df["driver_lat"] = 32.08
truck_stop_df["driver_lon"] = -81.09
truck_stop_df["start_time"] = pd.Timestamp("2023-12-02 12:20:00")
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
    truck_stop_df["driver_stop_path"].apply(extract_route_direction)
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

truck_stop_df.to_csv("1.csv", index=False)

print(truck_stop_df.columns)
print(truck_stop_df.head())
