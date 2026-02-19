from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from haversine import haversine_vector, Unit


# -----------------------------
# Config / Paths
# -----------------------------
@dataclass(frozen=True)
class DataPaths:
    base: str

    @property
    def truck_path_poi(self) -> str:
        return self.base + r"5. Source & Refrence Files\0. TruckerPath Data\MIT_2025_High_Volume_Routes_POI_Data - Copy.csv"

    @property
    def merged_traffic(self) -> str:
        return self.base + r"4. Working Data Files\Traffic Files\Capstone_truck\merged_filtered_file_11_18.csv"

    @property
    def amenities_xlsx(self) -> str:
        return self.base + r"5. Source & Refrence Files\0. TruckerPath Data\MIT parking location amenities.xlsx"

    @property
    def parking_1(self) -> str:
        return self.base + r"5. Source & Refrence Files\0. TruckerPath Data\MIT_2025_High_Volume_Routes_Parking_Data_1 - Copy.csv"

    @property
    def parking_2(self) -> str:
        return self.base + r"5. Source & Refrence Files\0. TruckerPath Data\MIT_2025_High_Volume_Routes_Parking_Data_2 - Copy.csv"

    @property
    def parking_3(self) -> str:
        return self.base + r"5. Source & Refrence Files\0. TruckerPath Data\MIT_2025_High_Volume_Routes_Parking_Data_3 - Copy.csv"

    @property
    def ginny_christina_combined(self) -> str:
        return self.base + r"4. Working Data Files\TruckStopsCombined.csv"


# -----------------------------
# Helper functions
# -----------------------------
def haversine_miles(
        df: pd.DataFrame,
        latlon_a_cols: List[str],
        latlon_b_cols: List[str],
        inflate_factor: float = 1.16,
) -> np.ndarray:
    """Vectorized haversine miles, optionally inflated by a factor (your code uses 1.16)."""
    return haversine_vector(df[latlon_a_cols], df[latlon_b_cols], Unit.MILES) * inflate_factor


def normalize_within_group(df: pd.DataFrame, group_col: str, value_col: str) -> pd.Series:
    """Min-max normalize value_col within each group_col. Handles constant groups safely."""

    def _minmax(x: pd.Series) -> pd.Series:
        mn, mx = x.min(), x.max()
        if pd.isna(mn) or pd.isna(mx) or mx == mn:
            return pd.Series([0.0] * len(x), index=x.index)
        return (x - mn) / (mx - mn)

    return df.groupby(group_col)[value_col].transform(_minmax)


def coerce_state_ids(df: pd.DataFrame, col: str = "stateid") -> pd.DataFrame:
    """Maps a few numeric state codes to 2-letter abbreviations."""
    mapping = {
        "1": "AL", 1: "AL",
        "12": "FL",
        "13": "GA",
        "45": "SC",
    }
    df = df.copy()
    df[col] = df[col].replace(mapping)
    return df


def resolve_close_duplicates(
        df: pd.DataFrame,
        dist_threshold_miles: float = 0.1,
        inflate_factor: float = 1.16,
) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Your old nested `test()` function:
    - Cross join stops against stops
    - Find pairs within threshold
    - If counts equal, drop one ID (set to 0 later)
    - If counts differ, keep the bigger count and zero out the smaller
    Returns (unique_id_table, updated_df, check)
    """
    df = df.copy()
    if "ucount2" in df.columns:
        df.drop(columns=["ucount2"], inplace=True)

    a = df[["pin id", "pinname", "lat", "lng", "truckParkingSpotCount"]].drop_duplicates().copy()
    a = pd.merge(a, a, how="cross")

    a["distance_miles"] = haversine_vector(
        a[["lat_x", "lng_x"]],
        a[["lat_y", "lng_y"]],
        Unit.MILES
    ) * inflate_factor

    # Make unordered pair key
    a["pair"] = a.apply(lambda x: tuple(sorted([x["pin id_x"], x["pin id_y"]])), axis=1)
    a = a.drop_duplicates(subset="pair", keep="first").drop(columns=["pair"])

    a = a[(a["distance_miles"] < dist_threshold_miles) & (a["pin id_x"] != a["pin id_y"])].copy()
    a.dropna(inplace=True)

    a = a[(a["truckParkingSpotCount_x"] != 0) & (a["truckParkingSpotCount_y"] != 0)].copy()

    # If counts are equal, mark one side for removal
    list_to_rm = a[a["truckParkingSpotCount_y"] == a["truckParkingSpotCount_x"]]["pin id_y"].unique()

    # Only keep unequal pairs for "winner takes all"
    a = a[a["truckParkingSpotCount_y"] != a["truckParkingSpotCount_x"]].copy()
    check = a.shape[0]

    if check == 0:
        # Still need to apply list_to_rm if any
        df["truckParkingSpotCount"] = np.where(df["pin id"].isin(list_to_rm), 0, df["truckParkingSpotCount"])
        return pd.DataFrame(columns=["pin id", "ucount2"]), df, 0

    a["bigger"] = np.where(a["truckParkingSpotCount_y"] > a["truckParkingSpotCount_x"], "y", "x")
    a["truckParkingSpotCount_y"] = np.where(a["bigger"] == "x", 0, a["truckParkingSpotCount_y"])
    a["truckParkingSpotCount_x"] = np.where(a["bigger"] == "y", 0, a["truckParkingSpotCount_x"])

    id_x = a[["pin id_x", "truckParkingSpotCount_x"]].drop_duplicates().copy()
    id_y = a[["pin id_y", "truckParkingSpotCount_y"]].drop_duplicates().copy()

    id_x.rename(columns={"pin id_x": "pin id", "truckParkingSpotCount_x": "truckParkingSpotCount"}, inplace=True)
    id_y.rename(columns={"pin id_y": "pin id", "truckParkingSpotCount_y": "truckParkingSpotCount"}, inplace=True)

    unique_id = pd.concat([id_x, id_y], ignore_index=True).drop_duplicates()
    unique_id.rename(columns={"truckParkingSpotCount": "ucount2"}, inplace=True)

    df = pd.merge(df, unique_id, on="pin id", how="left")
    df["truckParkingSpotCount"] = np.where(df["ucount2"].isnull(), df["truckParkingSpotCount"], df["ucount2"])
    df["truckParkingSpotCount"] = np.where(df["pin id"].isin(list_to_rm), 0, df["truckParkingSpotCount"])

    return unique_id, df, check


# -----------------------------
# Main class
# -----------------------------
class TruckStopModel:
    def __init__(self, base_path: str):
        self.paths = DataPaths(base_path)
        self.df_truck_path: Optional[pd.DataFrame] = None
        self.df_merged_file: Optional[pd.DataFrame] = None
        self.df_amenities: Optional[pd.DataFrame] = None
        self.park_data: Optional[pd.DataFrame] = None
        self.df_comb: Optional[pd.DataFrame] = None

    def load(self) -> "TruckStopModel":
        self.df_truck_path = pd.read_csv(self.paths.truck_path_poi)
        self.df_merged_file = pd.read_csv(self.paths.merged_traffic)
        self.df_amenities = pd.read_excel(self.paths.amenities_xlsx)

        p1 = pd.read_csv(self.paths.parking_1)
        p2 = pd.read_csv(self.paths.parking_2)
        p3 = pd.read_csv(self.paths.parking_3)
        self.park_data = pd.concat([p1, p2, p3], ignore_index=True)

        self.df_comb = pd.read_csv(self.paths.ginny_christina_combined)
        return self

    def _prep_inputs(self) -> None:
        assert self.df_truck_path is not None
        assert self.df_merged_file is not None

        self.df_truck_path = (
            self.df_truck_path[["Pin ID", "lat", "lng", "truckParkingSpotCount", "state", "overnightParking"]]
            .drop_duplicates()
        )

        self.df_merged_file = (
            self.df_merged_file[["routeid", "beginpoint", "endpoint", "f_system", "MID_LAT", "MID_LONG", "stateid"]]
            .drop_duplicates()
        )

        # Remove invalid geography
        self.df_merged_file = self.df_merged_file[~self.df_merged_file["MID_LAT"].isnull()].copy()

        # Fix state IDs
        self.df_merged_file = coerce_state_ids(self.df_merged_file, col="stateid")

    def _nearest_stop_per_pin_within_state(self) -> pd.DataFrame:
        """Your state loop: for each state, cross join + pick nearest road segment per Pin ID."""
        assert self.df_truck_path is not None
        assert self.df_merged_file is not None

        s_list = self.df_merged_file["stateid"].unique()

        df_stop = []
        for s in s_list:
            print(s)
            truck_df = self.df_truck_path[self.df_truck_path["state"] == s].copy()
            traffic_df = self.df_merged_file[self.df_merged_file["stateid"] == s].copy()

            if truck_df.empty or traffic_df.empty:
                continue

            temp = pd.merge(truck_df, traffic_df, how="cross")
            temp["distance_miles"] = haversine_miles(temp, ["lat", "lng"], ["MID_LAT", "MID_LONG"])

            idx = temp.groupby("Pin ID")["distance_miles"].idxmin()
            nearest = temp.loc[idx].reset_index(drop=True)
            df_stop.append(nearest)

        if not df_stop:
            return pd.DataFrame()

        df_stop = pd.concat(df_stop, ignore_index=True)

        nearest = df_stop[
            ["Pin ID", "lat", "lng", "truckParkingSpotCount", "f_system", "routeid", "beginpoint", "endpoint",
             "overnightParking"]
        ].copy()
        return nearest

    def _attach_stop_names_and_link_id(self, nearest: pd.DataFrame) -> pd.DataFrame:
        assert self.park_data is not None

        stop_name_df = self.park_data[["pinname", "pin id"]].drop_duplicates()

        print(nearest.shape)
        stop_tab_df = pd.merge(nearest, stop_name_df, left_on="Pin ID", right_on="pin id", how="left")
        print(stop_tab_df.shape)

        stop_tab_df["link_id"] = (
                stop_tab_df["routeid"].astype(str)
                + "_"
                + stop_tab_df["beginpoint"].astype(str)
                + "_"
                + stop_tab_df["endpoint"].astype(str)
        )

        stop_tab_df = stop_tab_df[
            ["pin id", "pinname", "lat", "lng", "truckParkingSpotCount", "f_system", "link_id", "overnightParking"]
        ].copy()
        stop_tab_df["review_score"] = ""
        return stop_tab_df

    def _override_parking_counts_with_gc(self, stop_tab_df: pd.DataFrame) -> pd.DataFrame:
        """Your Ginny & Christina cross-check logic."""
        assert self.df_comb is not None

        df_comb = self.df_comb[["StoreNumber", "Latitude", "Longitude", "ParkingSpaces"]].drop_duplicates()
        df_cross = pd.merge(stop_tab_df, df_comb, how="cross")

        df_cross["distance_miles"] = haversine_miles(df_cross, ["lat", "lng"], ["Latitude", "Longitude"])

        idx = df_cross.groupby("pin id")["distance_miles"].idxmin()
        df_comb_nearest = df_cross.loc[idx].reset_index(drop=True)

        gc_tp_df = df_comb_nearest[df_comb_nearest["distance_miles"] < 0.1].copy()

        # Count store occurrences
        gc_tp_df_gp = gc_tp_df.groupby(["StoreNumber"]).agg({"Latitude": "count"}).reset_index()
        gc_tp_df_gp.rename(columns={"Latitude": "count_store"}, inplace=True)

        # Keep only mismatches
        gc_tp_df = gc_tp_df[gc_tp_df["truckParkingSpotCount"] != gc_tp_df["ParkingSpaces"]].copy()
        gc_tp_df = pd.merge(gc_tp_df, gc_tp_df_gp, on="StoreNumber", how="left")

        # If store appears once, assume G&C is better
        gc_tp_df = gc_tp_df[gc_tp_df["count_store"] == 1].copy()
        gc_tp_df["truckParkingSpotCount"] = gc_tp_df["ParkingSpaces"]

        # Remove null & 0
        print(gc_tp_df.shape)
        gc_tp_df = gc_tp_df[(gc_tp_df["truckParkingSpotCount"] != 0)].copy()
        gc_tp_df = gc_tp_df[(~gc_tp_df["truckParkingSpotCount"].isnull())].copy()
        print(gc_tp_df.shape)

        gc_tp_df = gc_tp_df[["pin id", "truckParkingSpotCount"]].copy()
        gc_tp_df.rename(columns={"truckParkingSpotCount": "ucount"}, inplace=True)

        out = pd.merge(stop_tab_df, gc_tp_df, on="pin id", how="left")
        out["truckParkingSpotCount"] = np.where(out["ucount"].isnull(), out["truckParkingSpotCount"], out["ucount"])
        return out

    def _dedupe_close_stops_iteratively(self, stop_tab_df: pd.DataFrame) -> pd.DataFrame:
        """Your while-loop that repeatedly resolves close-by duplicates."""
        # One-off manual fix
        stop_tab_df = stop_tab_df.copy()
        stop_tab_df["truckParkingSpotCount"] = np.where(
            stop_tab_df["pinname"] == "ONE9 Dealer (One9 Fuel Network) #1365",
            60,
            stop_tab_df["truckParkingSpotCount"],
        )

        check = 1
        i = 0
        while check != 0:
            i += 1
            stop_tab_df = stop_tab_df[stop_tab_df["truckParkingSpotCount"] != 0].copy()
            stop_tab_df = stop_tab_df[~stop_tab_df["truckParkingSpotCount"].isnull()].copy()

            unique_id, stop_tab_df, check = resolve_close_duplicates(stop_tab_df)
            print(f"For {i} run, the check is {check}")

            if not unique_id.empty:
                a = unique_id.groupby(["pin id"]).agg({"ucount2": "count"}).reset_index()
                dupes = unique_id[unique_id["pin id"].isin(a[a["ucount2"] > 1]["pin id"].unique())]
                print(dupes.shape)

        return stop_tab_df

    def _compute_amenities_score(self, stop_tab_df: pd.DataFrame) -> pd.DataFrame:
        assert self.df_amenities is not None

        df_amenities = self.df_amenities[
            [
                "state", "petFriendly", "atmCount", "transfloExpress", "showerCount", "rvDumpStations", "wifi",
                "tireCare", "overnightParking", "faxScanService", "Pin ID", "pool", "laundry", "gym", "work24h7d",
                "lightedParking", "lightedBathroomAccess", "reserved_parking", "freePark"
            ]
        ].copy()

        df_amenities = df_amenities[df_amenities["Pin ID"].isin(stop_tab_df["pin id"].unique())].copy()
        df_amenities.fillna(0, inplace=True)

        for c in ["showerCount", "atmCount"]:
            df_amenities[f"{c}_norm"] = normalize_within_group(df_amenities, "state", c)

        cols = [
            "petFriendly", "transfloExpress", "rvDumpStations", "wifi", "tireCare", "overnightParking",
            "faxScanService", "pool", "laundry", "gym", "work24h7d", "lightedParking",
            "lightedBathroomAccess", "reserved_parking", "freePark", "showerCount_norm", "atmCount_norm",
        ]
        df_amenities["amenities_score"] = df_amenities[cols].sum(axis=1) / len(cols)
        df_amenities = df_amenities[["Pin ID", "amenities_score"]].copy()

        out = pd.merge(stop_tab_df, df_amenities, left_on="pin id", right_on="Pin ID", how="left")
        out["overnightParking"] = out["overnightParking"].fillna(False)
        return out

    def run(self) -> pd.DataFrame:
        """
        Main pipeline.
        Returns the same output columns as your original function.
        """
        if any(x is None for x in
               [self.df_truck_path, self.df_merged_file, self.df_amenities, self.park_data, self.df_comb]):
            self.load()

        self._prep_inputs()

        nearest = self._nearest_stop_per_pin_within_state()
        if nearest.empty:
            return pd.DataFrame(columns=[
                "pin id", "pinname", "lat", "lng", "truckParkingSpotCount", "f_system",
                "link_id", "review_score", "amenities_score", "overnightParking"
            ])

        stop_tab_df = self._attach_stop_names_and_link_id(nearest)
        stop_tab_df = self._override_parking_counts_with_gc(stop_tab_df)
        stop_tab_df = self._dedupe_close_stops_iteratively(stop_tab_df)
        stop_tab_df = self._compute_amenities_score(stop_tab_df)

        columns = [
            "pin id", "pinname", "lat", "lng", "truckParkingSpotCount", "f_system",
            "link_id", "review_score", "amenities_score", "overnightParking"
        ]
        return stop_tab_df[columns]


# -----------------------------
# Backwards-compatible wrapper
# -----------------------------
def model_stop(
        base_path: str = r"C:\Users\bhavy\Massachusetts Institute of Technology\Truck Parking Capstone - General\Truck Stop Finder 🚚⛽\\"
) -> pd.DataFrame:
    return TruckStopModel(base_path).run()
