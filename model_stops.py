import numpy as np
import pandas as pd
from haversine import haversine_vector, Unit


def model_stop():
    am_weight_dict = {}
    path = r"C:\Users\bhavy\Massachusetts Institute of Technology\Truck Parking Capstone - General\Truck Stop Finder 🚚⛽\\"
    # path = r"C:\Users\samcl\Massachusetts Institute of Technology\Truck Parking Capstone - Truck Stop Finder 🚚⛽\\"

    # Sourced directly from TruckerPath
    df_truck_path = pd.read_csv(
        path + r"5. Source & Refrence Files\0. TruckerPath Data\MIT_2025_High_Volume_Routes_POI_Data - Copy.csv")

    # Coming from ArcGIS
    df_merged_file = pd.read_csv(
        path + r"4. Working Data Files\Traffic Files\Capstone_truck\merged_filtered_file_11_18.csv")

    df_amenities = pd.read_excel(
        path + r"5. Source & Refrence Files\0. TruckerPath Data\MIT parking location amenities.xlsx")

    df_am_weight = pd.read_csv(path + r"5. Source & Refrence Files\am_weight.csv")

    # Sourced directly from TruckerPath
    park_data_1 = pd.read_csv(
        path + r"5. Source & Refrence Files\0. TruckerPath Data\MIT_2025_High_Volume_Routes_Parking_Data_1 - Copy.csv")
    park_data_2 = pd.read_csv(
        path + r"5. Source & Refrence Files\0. TruckerPath Data\MIT_2025_High_Volume_Routes_Parking_Data_2 - Copy.csv")
    park_data_3 = pd.read_csv(
        path + r"5. Source & Refrence Files\0. TruckerPath Data\MIT_2025_High_Volume_Routes_Parking_Data_3 - Copy.csv")
    park_data = pd.concat([park_data_1, park_data_2, park_data_3], ignore_index=True)

    df_truck_path = df_truck_path[
        ["Pin ID", "lat", "lng", "truckParkingSpotCount", "state", "overnightParking"]].drop_duplicates()
    df_merged_file = df_merged_file[["routeid", "beginpoint", "endpoint", "f_system", "MID_LAT",
                                     "MID_LONG", "stateid"]].drop_duplicates()

    # Removing road segments without valid geography
    df_merged_file = df_merged_file[~df_merged_file["MID_LAT"].isnull()].copy()

    df_merged_file['stateid'] = np.where((df_merged_file['stateid'] == '1') |
                                         (df_merged_file['stateid'] == 1), 'AL', df_merged_file['stateid'])
    df_merged_file['stateid'] = np.where(df_merged_file['stateid'] == '12', 'FL', df_merged_file['stateid'])
    df_merged_file['stateid'] = np.where(df_merged_file['stateid'] == '13', 'GA', df_merged_file['stateid'])
    df_merged_file['stateid'] = np.where(df_merged_file['stateid'] == '45', 'SC', df_merged_file['stateid'])

    df_ex = df_merged_file.copy()

    # df_merged_file.to_csv(
    #     path + r"4. Working Data Files\Traffic Files\Capstone_truck\merged_filtered_modified.csv", index=False)

    s_list = df_merged_file['stateid'].unique()

    df_stop = pd.DataFrame()
    for s in s_list:
        print(s)
        truck_df = df_truck_path[df_truck_path['state'] == s].copy()
        traffic_df = df_merged_file[df_merged_file['stateid'] == s].copy()
        temp = pd.merge(truck_df, traffic_df, how="cross")
        temp['distance_miles'] = haversine_vector(temp[['lat', 'lng']],
                                                  temp[['MID_LAT', 'MID_LONG']],
                                                  Unit.MILES
                                                  ) * 1.16
        idx = temp.groupby('Pin ID')['distance_miles'].idxmin()
        nearest = temp.loc[idx].reset_index()
        df_stop = pd.concat([df_stop, nearest], ignore_index=True)

    nearest = df_stop[
        ["Pin ID", "lat", "lng", "truckParkingSpotCount", "f_system", "routeid", "beginpoint", "endpoint",
         "overnightParking"]].copy()

    # Getting the truck ids name
    stop_name_df = park_data[["pinname", "pin id"]].drop_duplicates()
    # stop_name_df

    print(nearest.shape)
    stop_tab_df = pd.merge(nearest, stop_name_df, left_on="Pin ID", right_on="pin id", how="left")
    print(stop_tab_df.shape)

    stop_tab_df["link_id"] = stop_tab_df["routeid"].astype("str") + "_" + stop_tab_df["beginpoint"].astype(
        "str") + "_" + \
                             stop_tab_df["endpoint"].astype("str")
    stop_tab_df = stop_tab_df[
        ["pin id", "pinname", "lat", "lng", "truckParkingSpotCount", "f_system", "link_id", "overnightParking"]].copy()
    stop_tab_df["review_score"] = ""
    # stop_tab_df

    # Ginny & Christina data check

    df_comb = pd.read_csv(path + r"4. Working Data Files\TruckStopsCombined.csv")
    df_comb = df_comb[["StoreNumber", "Latitude", "Longitude", "ParkingSpaces"]].drop_duplicates()

    df_comb = pd.merge(stop_tab_df, df_comb, how="cross")

    df_comb['distance_miles'] = haversine_vector(df_comb[['lat', 'lng']],
                                                 df_comb[['Latitude', 'Longitude']],
                                                 Unit.MILES
                                                 ) * 1.16

    idx = df_comb.groupby('pin id')['distance_miles'].idxmin()
    df_comb_nearest = df_comb.loc[idx].reset_index()

    gc_tp_df = df_comb_nearest.copy()

    # Identified same stop as <.1 mile dist
    gc_tp_df = gc_tp_df[gc_tp_df["distance_miles"] < .1].copy()
    # Got the stopnumber count in the data
    gc_tp_df_gp = gc_tp_df.groupby(["StoreNumber"]).agg({"Latitude": "count"}).reset_index()
    gc_tp_df_gp.rename(columns={"Latitude": "count_store"}, inplace=True)

    # Removed where G&C & TruckerPath has same data
    gc_tp_df = gc_tp_df[gc_tp_df["truckParkingSpotCount"] != gc_tp_df["ParkingSpaces"]].copy()
    gc_tp_df = pd.merge(gc_tp_df, gc_tp_df_gp, on="StoreNumber", how="left")

    # For count of 1, we assume G&C has better data
    gc_tp_df = gc_tp_df[gc_tp_df["count_store"] == 1].copy()
    gc_tp_df["truckParkingSpotCount"] = gc_tp_df["ParkingSpaces"]

    # Removed null & 0 truck count rows
    print(gc_tp_df.shape)
    gc_tp_df = gc_tp_df[(gc_tp_df["truckParkingSpotCount"] != 0)].copy()
    gc_tp_df = gc_tp_df[(~gc_tp_df["truckParkingSpotCount"].isnull())].copy()
    print(gc_tp_df.shape)

    gc_tp_df = gc_tp_df[["pin id", 'truckParkingSpotCount']].copy()
    gc_tp_df.rename(columns={"truckParkingSpotCount": "ucount"}, inplace=True)

    stop_tab_df = pd.merge(stop_tab_df, gc_tp_df, on="pin id", how="left")

    stop_tab_df["truckParkingSpotCount"] = np.where(stop_tab_df["ucount"].isnull(),
                                                    stop_tab_df["truckParkingSpotCount"],
                                                    stop_tab_df["ucount"])

    def test(df):
        if 'ucount2' in df.columns:
            df.drop(columns=["ucount2"], inplace=True)
        a = df[['pin id', 'pinname', 'lat', 'lng', 'truckParkingSpotCount']].drop_duplicates().copy()
        a = pd.merge(a, a, how="cross")
        a['distance_miles'] = haversine_vector(a[['lat_x', 'lng_x']],
                                               a[['lat_y', 'lng_y']],
                                               Unit.MILES
                                               ) * 1.16
        a["pair"] = a.apply(lambda x: tuple(sorted([x["pin id_x"], x["pin id_y"]])), axis=1)
        a = a.drop_duplicates(subset="pair", keep="first").drop(columns=["pair"])

        a = a[a["distance_miles"] < .1].copy()
        a = a[a["pin id_x"] != a["pin id_y"]].copy()
        a.dropna(inplace=True)
        a = a[a["truckParkingSpotCount_x"] != 0].copy()
        a = a[a["truckParkingSpotCount_y"] != 0].copy()
        print(a.shape)
        list_to_rm = a[a["truckParkingSpotCount_y"] == a["truckParkingSpotCount_x"]]["pin id_y"].unique()
        a = a[a["truckParkingSpotCount_y"] != a["truckParkingSpotCount_x"]].copy()
        check = a.shape[0]
        print(a.shape)
        a['bigger'] = np.where(a["truckParkingSpotCount_y"] > a["truckParkingSpotCount_x"], "y", "x")
        a["truckParkingSpotCount_y"] = np.where(a['bigger'] == "x", 0, a["truckParkingSpotCount_y"])
        a["truckParkingSpotCount_x"] = np.where(a['bigger'] == "y", 0, a["truckParkingSpotCount_x"])

        id_x = a[["pin id_x", "truckParkingSpotCount_x"]].drop_duplicates().copy()
        id_y = a[["pin id_y", "truckParkingSpotCount_y"]].drop_duplicates().copy()
        id_x.rename(columns={"pin id_x": "pin id", "truckParkingSpotCount_x": "truckParkingSpotCount"}, inplace=True)
        id_y.rename(columns={"pin id_y": "pin id", "truckParkingSpotCount_y": "truckParkingSpotCount"}, inplace=True)
        unique_id = pd.concat([id_x, id_y], ignore_index=True)

        unique_id.drop_duplicates(inplace=True)
        unique_id.rename(columns={"truckParkingSpotCount": "ucount2"}, inplace=True)

        print(df.shape)
        df = pd.merge(df, unique_id, on="pin id", how="left")
        print(df.shape)
        df["truckParkingSpotCount"] = np.where(df["ucount2"].isnull(), df["truckParkingSpotCount"],
                                               df["ucount2"])

        df["truckParkingSpotCount"] = np.where(df["pin id"].isin(list_to_rm), 0,
                                               df["truckParkingSpotCount"])

        return unique_id, df, check

    stop_tab_df["truckParkingSpotCount"] = np.where(stop_tab_df["pinname"] == "ONE9 Dealer (One9 Fuel Network) #1365",
                                                    60,
                                                    stop_tab_df["truckParkingSpotCount"])

    check = 1
    i = 0
    while check != 0:
        i += 1
        stop_tab_df = stop_tab_df[stop_tab_df["truckParkingSpotCount"] != 0].copy()
        stop_tab_df = stop_tab_df[~stop_tab_df["truckParkingSpotCount"].isnull()]
        unique_id, stop_tab_df, check = test(stop_tab_df)
        print(f"For {i} run, the check is {check}")
        a = unique_id.groupby(["pin id"]).agg({"ucount2": "count"}).reset_index()
        print(unique_id[unique_id["pin id"].isin(a[a["ucount2"] > 1]["pin id"].unique())].shape)

    columns = ['pin id', 'pinname', 'lat', 'lng', 'truckParkingSpotCount', 'f_system',
               'link_id', 'review_score', 'amenities_score', "overnightParking"]

    df_amenities = df_amenities[
        ['state', 'petFriendly', 'atmCount', 'transfloExpress', 'showerCount', 'rvDumpStations', 'wifi', 'tireCare',
         'overnightParking', 'faxScanService', 'Pin ID', 'pool', 'laundry', 'gym', 'work24h7d', 'lightedParking',
         'lightedBathroomAccess', 'reserved_parking', 'freePark']].copy()

    df_amenities = df_amenities[df_amenities['Pin ID'].isin(stop_tab_df["pin id"].unique())].copy()

    df_amenities.fillna(0, inplace=True)

    for c in ['showerCount', 'atmCount']:
        df_amenities[f'{c}_norm'] = (
            df_amenities.groupby('state')[c]
            .transform(lambda x: (x - x.min()) / (x.max() - x.min()))
        )

    cols = ['petFriendly', 'transfloExpress',
            'rvDumpStations', 'wifi', 'tireCare', 'overnightParking',
            'faxScanService', 'pool', 'laundry', 'gym', 'work24h7d',
            'lightedParking', 'lightedBathroomAccess', 'reserved_parking',
            'freePark', 'showerCount_norm', 'atmCount_norm']

    df_am_weight["weight"] = df_am_weight["weight"] / df_am_weight["weight"].sum()

    for i, row in df_am_weight.iterrows():
        c = row["column"]
        w = row["weight"]
        am_weight_dict[c] = w

    df_amenities["amenities_score"] = (df_amenities[cols] * pd.Series(am_weight_dict)).sum(axis=1)
    # df_amenities["amenities_score"] = df_amenities[cols].sum(axis=1) / len(cols)

    print(df_amenities["amenities_score"].describe())
    # sys.exit()

    df_amenities = df_amenities[['Pin ID', 'amenities_score']].copy()

    stop_tab_df = pd.merge(stop_tab_df, df_amenities, left_on="pin id", right_on="Pin ID", how="left")

    stop_tab_df['overnightParking'] = stop_tab_df['overnightParking'].fillna(False)

    return stop_tab_df[columns], df_ex


model_stop()
