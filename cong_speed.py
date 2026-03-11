import os
import warnings
import zipfile

import duckdb
import h3
import pandas as pd

from model_stops import model_stop_func

warnings.filterwarnings("ignore")

path = r"C:\Users\bhavy\Massachusetts Institute of Technology\Truck Parking Capstone - General\Truck Stop Finder 🚚⛽\\"
# path = r"C:\Users\samcl\Massachusetts Institute of Technology\Truck Parking Capstone - Truck Stop Finder 🚚⛽\\"

sensor_loc = pd.read_csv(path + r"5. Source & Refrence Files\sensor_loc_w_ZIP5.csv",
                         dtype={"station_id": 'str', 'ZIP5': 'str'})
state_map = pd.read_csv(path + r"5. Source & Refrence Files\State_mapping.csv")
model_stop, traffic_df = model_stop_func()
# traffic_df = pd.read_csv(path + r"4. Working Data Files\Traffic Files\Capstone_truck\merged_filtered_modified.csv")


sensor_loc = sensor_loc[~sensor_loc["State"].isin(["AK", "HI"])].copy()

# For hexagons: https://h3geo.org/docs/core-library/restable/

for r in [2]:
    print(f"Starting r {r}")
    resolution = r


    def hr_p(row):
        return h3.latlng_to_cell(row["Latitude"], row["Longitude"], resolution)


    sensor_loc["polygon"] = sensor_loc.apply(hr_p, axis=1)

    sensor_loc.drop(columns="ZIP5", inplace=True)

    run = False

    if run:
        folder = path + r"5. Source & Refrence Files\2024_traffic_data"
        out_dir = os.path.join(path, r"5. Source & Refrence Files\2024_traffic_parquet")
        os.makedirs(out_dir, exist_ok=True)

        # build mapping dict ONCE from your state_map df
        state_dict = state_map.set_index("state_code")["State"].to_dict()

        id_var_col = [
            'record_type', 'state_code', 'f_system', 'station_id', 'travel_dir',
            'travel_lane', 'year_record', 'month_record', 'day_record',
            'day_of_week', 'restrictions'
        ]

        part = 0

        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)

            if not filename.lower().endswith(".zip"):
                continue

            print(f"Opening ZIP: {filename}")

            with zipfile.ZipFile(file_path, 'r') as z:
                for inner_name in z.namelist():
                    if inner_name.endswith("/"):
                        continue

                    print(f"  Processing inside ZIP: {inner_name}")

                    with z.open(inner_name) as f:
                        # Read CSV from inside the ZIP
                        df = pd.read_csv(
                            f,
                            delimiter="|",
                            low_memory=False  # avoids dtype warning at cost of some RAM, OK for chunk
                            # You can also pass dtype={} here if you know them
                        )

                        # Melt wide hours columns into long format
                        df = pd.melt(
                            df,
                            id_vars=id_var_col,
                            var_name="hours",
                            value_name="traffic"
                        )

                        # Add state_name via map instead of big merge later
                        df["State"] = df["state_code"].map(state_dict)
                        df["station_id"] = df["station_id"].astype(str)

                        # Save this chunk to Parquet and drop from memory
                        out_path = os.path.join(out_dir, f"traffic_part_{part}.parquet")
                        df.to_parquet(out_path, index=False)
                        # print(f"    → wrote {out_path}")

                        del df
                        part += 1

        print(f"Done. Wrote {part} parquet files to {out_dir}")

    out_dir = os.path.join(path, r"5. Source & Refrence Files\2024_traffic_parquet")
    traffic_parquet_glob = f"{out_dir}/traffic_part_*.parquet"

    con = duckdb.connect()
    con.register("sensor_loc", sensor_loc)

    out_dir = os.path.join(path, r"5. Source & Refrence Files\2024_traffic_parquet")
    traffic_parquet_glob = f"{out_dir}/traffic_part_*.parquet"

    con.execute(f"""
        CREATE OR REPLACE TABLE traffic_matched AS
        SELECT
            t.*,                           -- all columns from traffic
            s.*
        FROM read_parquet('{traffic_parquet_glob}') AS t
        LEFT JOIN sensor_loc AS s
          ON t.station_id = s."Station Id"
         AND t.State      = s.State
        WHERE s."Latitude" IS NOT NULL
    """)

    con.execute(f"""
        CREATE OR REPLACE TABLE traffic_unmatched  AS
        SELECT
            t.*,                           -- all columns from traffic
            s.*
        FROM read_parquet('{traffic_parquet_glob}') AS t
        LEFT JOIN sensor_loc AS s
          ON t.station_id = s."Station Id"
         AND t.State      = s.State
        WHERE s."Latitude" IS NULL
    """)

    con.execute("""
                UPDATE traffic_unmatched
                SET station_id = ltrim(station_id, '0')
                """)

    con.execute('ALTER TABLE traffic_unmatched DROP COLUMN Latitude')
    con.execute('ALTER TABLE traffic_unmatched DROP COLUMN Longitude')
    con.execute('ALTER TABLE traffic_unmatched DROP COLUMN "Functional Class"')
    con.execute('ALTER TABLE traffic_unmatched DROP COLUMN State_1')
    con.execute('ALTER TABLE traffic_unmatched DROP COLUMN "Station Id"')

    con.execute('ALTER TABLE traffic_unmatched DROP COLUMN polygon')

    con.execute("""
                INSERT INTO traffic_matched
                SELECT t.*,
                       s.*
                FROM traffic_unmatched t
                         LEFT JOIN sensor_loc s
                                   ON t.station_id = s."Station Id"
                                       AND t.State = s.State
                WHERE s."Latitude" IS NOT NULL
                """)

    con.execute("""CREATE OR REPLACE TABLE traffic_gp_matched  AS
                select record_type,state_code,f_system,station_id,travel_dir,year_record,month_record,day_record,day_of_week,restrictions,hours, sum(traffic) as "traffic_volume",State,Latitude,Longitude,"Functional Class",State_1,"Station Id",polygon, count(distinct travel_lane) as "lane_count"
                   from traffic_matched
                   group by record_type,state_code,f_system,station_id,travel_dir,year_record,month_record,day_record,day_of_week,restrictions,hours,State,Latitude,Longitude,"Functional Class",State_1,"Station Id",polygon
                """)

    df = con.execute("""select State_1,
                               polygon,
                               travel_dir,
                               day_of_week,
                               hours,
                               lane_count,
                               avg(traffic_volume)                          as avg_traffic,
                               median(traffic_volume),
                               stddev(traffic_volume),
                               stddev(traffic_volume) / avg(traffic_volume) as CV
                        from traffic_gp_matched
                        group by State_1, polygon, travel_dir, day_of_week, hours, lane_count""").df()

    df.to_csv(path + rf"5. Source & Refrence Files\Congestion_speed_r_{r}.csv", index=False)

    # df['delta'] = abs((df["avg_traffic"] - df["median(traffic_volume)"]) / df["avg_traffic"])

    # def hr_p_model(row):
    #     return h3.latlng_to_cell(row["lat"], row["lng"], resolution)
    #
    #
    # model_stop["polygon"] = model_stop.apply(hr_p_model, axis=1)
    #
    # graph = df.groupby(["polygon"]).agg({"avg_traffic": "sum"}).reset_index()

    # def h3_to_polygon(h):
    #     # H3 returns (lat, lng); shapely wants (lng, lat)
    #     boundary = h3.cell_to_boundary(h)
    #     return Polygon([(lng, lat) for lat, lng in boundary])
    #
    #
    # gdf = gpd.GeoDataFrame(
    #     graph,
    #     geometry=df["polygon"].apply(h3_to_polygon),
    #     crs="EPSG:4326"  # lat/lon
    # )

    print(f"Done for {r}")
