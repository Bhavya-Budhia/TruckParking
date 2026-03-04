import pandas as pd
from scgraph.geographs.us_freeway import us_freeway_geograph

from model_stops import model_stop_func


def scgraph_distance_mi(row) -> float:
    out = us_freeway_geograph.get_shortest_path(
        origin_node={"latitude": float(row["driver_lat"]), "longitude": float(row["driver_lon"])},
        destination_node={"latitude": float(row["lat"]), "longitude": float(row["lng"])},
        output_units="mi",
        # cache=True can speed things up IF you reuse the same origins a lot
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
truck_stop_df["freeflow_mph"] = truck_stop_df["f_system"].map(freeflow_mph_dict)
truck_stop_df["truck_stop_mi"] = truck_stop_df.apply(scgraph_distance_mi, axis=1)

print(truck_stop_df.columns)
