import os
import warnings

import duckdb
import pandas as pd

# from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

path = r"C:\Users\bhavy\Massachusetts Institute of Technology\Truck Parking Capstone - General\Truck Stop Finder 🚚⛽\\"
# path = r"C:\Users\samcl\Massachusetts Institute of Technology\Truck Parking Capstone - Truck Stop Finder 🚚⛽\\"

sensor_loc = pd.read_csv(path + r"5. Source & Refrence Files\sensor_loc_w_ZIP5.csv",
                         dtype={"station_id": 'str', 'ZIP5': 'str'})

sensor_loc = sensor_loc[sensor_loc["State"] != "AK"].copy()

print(1)

# search = SearchEngine()
#
# def get_zip(row):
#     r = search.by_coordinates(row['Latitude'], row['Longitude'], radius=60, returns=1)
#     return r[0].zipcode if r else None
#
# sensor_loc['ZIP5'] = sensor_loc.apply(get_zip, axis=1)

sensor_loc["ZIP3"] = sensor_loc["ZIP5"].str[:3]

sensor_loc.drop(columns="ZIP5", inplace=True)

print(2)

con = duckdb.connect()
con.register("sensor_loc", sensor_loc)

out_dir = os.path.join(path, r"5. Source & Refrence Files\2024_traffic_parquet")
traffic_parquet_glob = f"{out_dir}/traffic_part_*.parquet"

print(3)

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

print(4)

con.execute("""
            UPDATE traffic_unmatched
            SET station_id = ltrim(station_id, '0')
            """)

con.execute('ALTER TABLE traffic_unmatched DROP COLUMN Latitude')
con.execute('ALTER TABLE traffic_unmatched DROP COLUMN Longitude')
con.execute('ALTER TABLE traffic_unmatched DROP COLUMN "Functional Class"')
con.execute('ALTER TABLE traffic_unmatched DROP COLUMN State_1')
con.execute('ALTER TABLE traffic_unmatched DROP COLUMN "Station Id"')

con.execute('ALTER TABLE traffic_unmatched DROP COLUMN ZIP3')

print(5)

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
            select record_type,state_code,f_system,station_id,travel_dir,year_record,month_record,day_record,day_of_week,restrictions,hours, sum(traffic) as "traffic_volume",State,Latitude,Longitude,"Functional Class",State_1,"Station Id",ZIP3, count(distinct travel_lane) as "lane_count"
               from traffic_matched
               group by record_type,state_code,f_system,station_id,travel_dir,year_record,month_record,day_record,day_of_week,restrictions,hours,State,Latitude,Longitude,"Functional Class",State_1,"Station Id",ZIP3
            """)

print(6)

df = con.execute("""select State_1,
                           ZIP3,
                           travel_dir,
                           day_of_week,
                           hours,
                           avg(traffic_volume)                          as avg_traffic,
                           median(traffic_volume),
                           stddev(traffic_volume),
                           stddev(traffic_volume) / avg(traffic_volume) as CV,
                           lane_count
                    from traffic_gp_matched
                    group by State_1, ZIP3, travel_dir, day_of_week, hours, lane_count""").df()

print(7)

df.to_csv("ZIP3_agg_data.csv", index=False)
