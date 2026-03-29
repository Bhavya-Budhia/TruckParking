import folium
import pandas as pd
from folium import Element

from model_engine import model_engine_func

# --------------------------------------------------
# Config
# --------------------------------------------------
output_map = "truck_stops_map.html"

# --------------------------------------------------
# Load data
# --------------------------------------------------
df = model_engine_func()

# Basic checks
required_cols = [
    "driver_lat", "driver_lon",
    "dest_lat", "dest_lon",
    "lat", "lng",
    "pinname",
    "utility_score"
]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns in CSV: {missing}")

# If feasible_stop is not present, create a fallback
if "feasible_stop" not in df.columns:
    if "HOS_exceeded" in df.columns and "truck_stop_after_dest" in df.columns:
        df["feasible_stop"] = (
                (df["HOS_exceeded"] == 0) &
                (df["truck_stop_after_dest"] == 0)
        ).astype(int)
    else:
        df["feasible_stop"] = 1

# --------------------------------------------------
# Center map
# --------------------------------------------------
source_lat = df["driver_lat"].iloc[0]
source_lon = df["driver_lon"].iloc[0]
dest_lat = df["dest_lat"].iloc[0]
dest_lon = df["dest_lon"].iloc[0]

center_lat = (source_lat + dest_lat) / 2
center_lon = (source_lon + dest_lon) / 2

m = folium.Map(location=[center_lat, center_lon], zoom_start=7, tiles="CartoDB Positron")

# --------------------------------------------------
# Add source and destination
# --------------------------------------------------
folium.CircleMarker(
    location=[source_lat, source_lon],
    radius=8,
    color="blue",
    fill=True,
    fill_color="blue",
    fill_opacity=1,
    popup="Source",
    tooltip="Source"
).add_to(m)

folium.CircleMarker(
    location=[dest_lat, dest_lon],
    radius=8,
    color="blue",
    fill=True,
    fill_color="blue",
    fill_opacity=1,
    popup="Destination",
    tooltip="Destination"
).add_to(m)

# --------------------------------------------------
# Define feasible vs infeasible
# --------------------------------------------------
feasible_df = df[df["feasible_stop"] == 1].copy()
infeasible_df = df[df["feasible_stop"] == 0].copy()

# --------------------------------------------------
# Quantile buckets for feasible stops
# 3 quantile groups:
#   low    -> red/orange
#   mid    -> yellow
#   high   -> green
# infeasible -> black
# --------------------------------------------------
if len(feasible_df) > 0:
    # rank-based qcut is more stable if many repeated values
    feasible_df["utility_bucket"] = pd.qcut(
        feasible_df["utility_score"].rank(method="first"),
        q=3,
        labels=["low", "mid", "high"]
    )
else:
    feasible_df["utility_bucket"] = pd.Series(dtype="object")


def get_stop_color(row):
    if row["feasible_stop"] == 0:
        return "black"

    bucket = row.get("utility_bucket", None)
    if bucket == "high":
        return "green"
    elif bucket == "mid":
        return "yellow"
    else:
        return "red"


# --------------------------------------------------
# Add stop markers
# --------------------------------------------------
for _, row in feasible_df.iterrows():
    color = get_stop_color(row)

    popup_text = f"""
    <b>{row.get('pinname', 'Unknown Stop')}</b><br>
    Utility Score: {row.get('utility_score', 'NA'):.4f}<br>
    p_available: {row.get('p_available', float('nan')):.4f}<br>
    p_full: {row.get('p_full', float('nan')):.4f}<br>
    Amenities: {row.get('amenities_score', 'NA')}<br>
    Spots: {row.get('truckParkingSpotCount', 'NA')}<br>
    Truck Stop Miles: {row.get('truck_stop_mi', 'NA')}<br>
    Stop to Dest Miles: {row.get('stop_dest_mi', 'NA')}<br>
    Extra Dist: {row.get('extra_dist', 'NA')}<br>
    """

    folium.CircleMarker(
        location=[row["lat"], row["lng"]],
        radius=6,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.9,
        popup=folium.Popup(popup_text, max_width=300),
        tooltip=row.get("pinname", "Stop")
    ).add_to(m)

for _, row in infeasible_df.iterrows():
    popup_text = f"""
    <b>{row.get('pinname', 'Unknown Stop')}</b><br>
    Utility Score: {row.get('utility_score', 'NA'):.4f}<br>
    Feasible: No<br>
    p_available: {row.get('p_available', float('nan')):.4f}<br>
    p_full: {row.get('p_full', float('nan')):.4f}<br>
    Amenities: {row.get('amenities_score', 'NA')}<br>
    Spots: {row.get('truckParkingSpotCount', 'NA')}<br>
    Truck Stop Miles: {row.get('truck_stop_mi', 'NA')}<br>
    Stop to Dest Miles: {row.get('stop_dest_mi', 'NA')}<br>
    Extra Dist: {row.get('extra_dist', 'NA')}<br>
    """

    folium.CircleMarker(
        location=[row["lat"], row["lng"]],
        radius=6,
        color="black",
        fill=True,
        fill_color="black",
        fill_opacity=0.9,
        popup=folium.Popup(popup_text, max_width=300),
        tooltip=row.get("pinname", "Infeasible Stop")
    ).add_to(m)

# --------------------------------------------------
# Fit map bounds
# --------------------------------------------------
bounds_points = [[source_lat, source_lon], [dest_lat, dest_lon]] + df[["lat", "lng"]].values.tolist()
m.fit_bounds(bounds_points)

# --------------------------------------------------
# Add legend
# --------------------------------------------------
legend_html = """
<div style="
    position: fixed;
    bottom: 40px;
    left: 40px;
    width: 230px;
    z-index: 9999;
    background-color: white;
    border: 2px solid grey;
    border-radius: 8px;
    padding: 12px;
    font-size: 14px;
    ">
    <b>Legend</b><br><br>
    <span style="color: blue;">●</span> Source / Destination<br>
    <span style="color: green;">●</span> High utility feasible stop<br>
    <span style="color: yellow;">●</span> Mid utility feasible stop<br>
    <span style="color: red;">●</span> Low utility feasible stop<br>
    <span style="color: black;">●</span> Infeasible stop
</div>
"""
m.get_root().html.add_child(Element(legend_html))

# --------------------------------------------------
# Save
# --------------------------------------------------
m.save(output_map)
print(f"Map saved to: {output_map}")
