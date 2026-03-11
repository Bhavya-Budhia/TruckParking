import pandas as pd
from scgraph.geographs.us_freeway import us_freeway_geograph

from model_stops import model_stop_func


def scgraph_distance_mi(row, origin_lat, origin_lon, dest_lat, dest_lon) -> float:
    out = us_freeway_geograph.get_shortest_path(
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
    return out["length"]


truck_stop_df, traffic_df = model_stop_func()

truck_stop_df = truck_stop_df[["pin id", "pinname", "link_id", "lat", "lng", "truckParkingSpotCount", "amenities_score",
                               "f_system"]].copy()

freeflow_mph_dict = {1: 70,
                     2: 60,
                     3: 65}

truck_stop_df["driver_lat"] = 32.08
truck_stop_df["driver_lon"] = -81.09
truck_stop_df["start_time"] = pd.Timestamp("2023-12-02 12:20:00")
truck_stop_df["dest_lat"] = 27.95
truck_stop_df["dest_lon"] = -82.46
truck_stop_df["HOS_left_hr"] = 5
# truck_stop_df["freeflow_mph"] = truck_stop_df["f_system"].map(freeflow_mph_dict)
truck_stop_df["freeflow_mph"] = 55
truck_stop_df["truck_stop_mi"] = truck_stop_df.apply(
    scgraph_distance_mi,
    axis=1,
    args=("driver_lat", "driver_lon", "lat", "lng")
)
truck_stop_df["stop_dest_mi"] = truck_stop_df.apply(
    scgraph_distance_mi,
    axis=1,
    args=("lat", "lng", "dest_lat", "dest_lon")
)
truck_stop_df["truck_dest_mi"] = truck_stop_df.apply(
    scgraph_distance_mi,
    axis=1,
    args=("driver_lat", "driver_lon", "dest_lat", "dest_lon")
)

truck_stop_df["truck_stop_after_dest"] = truck_stop_df["truck_stop_mi"] > truck_stop_df["truck_dest_mi"]

truck_stop_df["ETA_stop"] = truck_stop_df["start_time"] + pd.to_timedelta(
    truck_stop_df["truck_stop_mi"] / truck_stop_df["freeflow_mph"], unit="h")

# Day of week (name)
truck_stop_df["day_of_week"] = truck_stop_df["ETA_stop"].dt.day_name()

# Hour in 24-hour format
truck_stop_df["hour_24"] = "hour_" + truck_stop_df["ETA_stop"].dt.hour.astype(str).str.zfill(2)

truck_stop_df.to_csv("1.csv", index=False)

print(truck_stop_df.columns)
