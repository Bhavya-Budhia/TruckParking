import folium
import pandas as pd
import streamlit as st
from folium import Element
from streamlit.components.v1 import html

from model_engine import model_engine_func

st.set_page_config(page_title="Truck Stop Finder", layout="wide")

st.title("Truck Stop Finder")
st.write("Enter trip details, click the button, and view ranked truck stops on the map.")

# -----------------------------
# Sidebar inputs
# -----------------------------
st.sidebar.header("Trip Inputs")

driver_lat = st.sidebar.number_input("Driver Latitude", value=25.7752, format="%.6f")
driver_lon = st.sidebar.number_input("Driver Longitude", value=-80.2086, format="%.6f")

dest_lat = st.sidebar.number_input("Destination Latitude", value=33.76052, format="%.6f")
dest_lon = st.sidebar.number_input("Destination Longitude", value=-78.91463, format="%.6f")

hos_left_hr = st.sidebar.number_input("HOS Left (hours)", value=5.0, min_value=0.0, step=0.5)
freeflow_mph = st.sidebar.number_input("Freeflow Speed (mph)", value=55.0, min_value=1.0, step=1.0)

start_time = st.sidebar.text_input("Start Time (UTC)", value="2023-12-02 12:20:00")

run_button = st.sidebar.button("Run Model")


# -----------------------------
# Map builder
# -----------------------------
def build_map(df):
    required_cols = [
        "driver_lat", "driver_lon",
        "dest_lat", "dest_lon",
        "lat", "lng",
        "pinname",
        "utility_score"
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataframe: {missing}")

    if "feasible_stop" not in df.columns:
        if "HOS_exceeded" in df.columns and "truck_stop_after_dest" in df.columns:
            df["feasible_stop"] = (
                    (df["HOS_exceeded"] == 0) &
                    (df["truck_stop_after_dest"] == 0)
            ).astype(int)
        else:
            df["feasible_stop"] = 1

    source_lat = df["driver_lat"].iloc[0]
    source_lon = df["driver_lon"].iloc[0]
    dest_lat = df["dest_lat"].iloc[0]
    dest_lon = df["dest_lon"].iloc[0]

    center_lat = (source_lat + dest_lat) / 2
    center_lon = (source_lon + dest_lon) / 2

    m = folium.Map(location=[center_lat, center_lon], zoom_start=7, tiles="CartoDB Positron")

    # source + destination
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

    feasible_df = df[df["feasible_stop"] == 1].copy()
    infeasible_df = df[df["feasible_stop"] == 0].copy()

    if len(feasible_df) > 0:
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

    for _, row in feasible_df.iterrows():
        color = get_stop_color(row)

        popup_text = f"""
        <b>{row.get('pinname', 'Unknown Stop')}</b><br>
        Utility Score: {row.get('utility_score', float('nan')):.4f}<br>
        Utility Rank: {row.get('utility_rank', 'NA')}<br>
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
        Utility Score: {row.get('utility_score', float('nan')):.4f}<br>
        Feasible: No<br>
        Utility Rank: {row.get('utility_rank', 'NA')}<br>
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
            radius=3,
            color="black",
            fill=True,
            fill_color="black",
            fill_opacity=0.9,
            popup=folium.Popup(popup_text, max_width=300),
            tooltip=row.get("pinname", "Infeasible Stop")
        ).add_to(m)

    bounds_points = [[source_lat, source_lon], [dest_lat, dest_lon]] + df[["lat", "lng"]].values.tolist()
    m.fit_bounds(bounds_points)

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
    return m


# -----------------------------
# Run model
# -----------------------------
if run_button:
    with st.spinner("Running model and building map..."):
        df = model_engine_func(
            driver_lat=driver_lat,
            driver_lon=driver_lon,
            dest_lat=dest_lat,
            dest_lon=dest_lon,
            hos_left_hr=hos_left_hr,
            freeflow_mph=freeflow_mph,
            start_time=start_time
        )

        st.subheader("Top Ranked Stops")
        show_cols = [
            "utility_rank", "pinname", "utility_score", "feasible_stop",
            "p_available", "p_full", "truck_stop_mi", "stop_dest_mi",
            "extra_dist", "detour_mi", "amenities_score", "truckParkingSpotCount"
        ]
        show_cols = [c for c in show_cols if c in df.columns]
        st.dataframe(df[show_cols].head(20), use_container_width=True)

        fmap = build_map(df)
        html(fmap._repr_html_(), height=700, scrolling=True)
