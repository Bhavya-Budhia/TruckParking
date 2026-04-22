import folium
import pandas as pd
import streamlit as st
from folium import Element
from streamlit.components.v1 import html

from model_engine_v2 import model_engine_func
from simulation_engine import HOS_HOURS, aggregate_simulation_results, run_simulation

st.set_page_config(page_title="Truck Stop Finder", layout="wide")


@st.cache_data(show_spinner=False)
def run_single_cached(driver_lat, driver_lon, dest_lat, dest_lon, hos_left_hr, freeflow_mph, start_time):
    return model_engine_func(
        driver_lat=driver_lat,
        driver_lon=driver_lon,
        dest_lat=dest_lat,
        dest_lon=dest_lon,
        hos_left_hr=hos_left_hr,
        freeflow_mph=freeflow_mph,
        start_time=start_time,
    )


@st.cache_data(show_spinner=False)
def run_sim_cached(
        driver_lat,
        driver_lon,
        dest_lat,
        dest_lon,
        num_runs,
        freeflow_mph,
        base_date,
        base_amenity_weight,
        location_jitter_deg,
        amenity_jitter,
        seed,
):
    return run_simulation(
        driver_lat=driver_lat,
        driver_lon=driver_lon,
        dest_lat=dest_lat,
        dest_lon=dest_lon,
        num_runs=num_runs,
        freeflow_mph=freeflow_mph,
        base_date=base_date,
        base_amenity_weight=base_amenity_weight,
        location_jitter_deg=location_jitter_deg,
        amenity_jitter=amenity_jitter,
        seed=seed,
    )


def build_single_run_map(df: pd.DataFrame):
    source_lat = df["driver_lat"].iloc[0]
    source_lon = df["driver_lon"].iloc[0]
    dest_lat = df["dest_lat"].iloc[0]
    dest_lon = df["dest_lon"].iloc[0]

    center_lat = (source_lat + dest_lat) / 2
    center_lon = (source_lon + dest_lon) / 2
    m = folium.Map(location=[center_lat, center_lon], zoom_start=7, tiles="CartoDB Positron")

    folium.CircleMarker([source_lat, source_lon], radius=8, color="blue", fill=True, fill_color="blue",
                        popup="Source").add_to(m)
    folium.CircleMarker([dest_lat, dest_lon], radius=8, color="blue", fill=True, fill_color="blue",
                        popup="Destination").add_to(m)

    feasible_df = df[df["feasible_stop"] == 1].copy()
    infeasible_df = df[df["feasible_stop"] == 0].copy()

    if len(feasible_df) > 0:
        feasible_df["utility_bucket"] = pd.qcut(
            feasible_df["utility_score"].rank(method="first"), q=3, labels=["low", "mid", "high"]
        )
    else:
        feasible_df["utility_bucket"] = pd.Series(dtype="object")

    def get_color(row):
        if row.get("utility_bucket") == "high":
            return "green"
        if row.get("utility_bucket") == "mid":
            return "yellow"
        return "red"

    for _, row in feasible_df.iterrows():
        color = get_color(row)
        popup = f"""
        <b>{row['pinname']}</b><br>
        Utility Score: {row['utility_score']:.4f}<br>
        Rank: {row['utility_rank']}<br>
        p_available: {row['p_available']:.4f}<br>
        Detour (mi): {row['detour_mi']:.2f}<br>
        Amenities: {row['amenities_score']:.4f}<br>
        Spots: {row['truckParkingSpotCount']}
        """
        folium.CircleMarker(
            [row["lat"], row["lng"]], radius=6, color=color, fill=True,
            fill_color=color, fill_opacity=0.9,
            popup=folium.Popup(popup, max_width=300), tooltip=row["pinname"],
        ).add_to(m)

    for _, row in infeasible_df.iterrows():
        popup = f"""
        <b>{row['pinname']}</b><br>
        Feasible: No<br>
        p_available: {row['p_available']:.4f}<br>
        HOS Exceeded: {row['HOS_exceeded']}<br>
        After Destination: {row['truck_stop_after_dest']}
        """
        folium.CircleMarker(
            [row["lat"], row["lng"]], radius=3, color="black", fill=True,
            fill_color="black", fill_opacity=0.9,
            popup=folium.Popup(popup, max_width=300), tooltip=row["pinname"],
        ).add_to(m)

    bounds_points = [[source_lat, source_lon], [dest_lat, dest_lon]] + df[["lat", "lng"]].values.tolist()
    m.fit_bounds(bounds_points)

    legend_html = """
    <div style="position: fixed; bottom: 40px; left: 40px; width: 240px; z-index: 9999;
        background-color: white; border: 2px solid grey; border-radius: 8px; padding: 12px; font-size: 14px;">
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


def build_simulation_map(summary_df: pd.DataFrame, scenario_df: pd.DataFrame, title_prefix: str = ""):
    if summary_df.empty or scenario_df.empty:
        return folium.Map(location=[39.5, -98.35], zoom_start=4, tiles="CartoDB Positron")

    source_lat = scenario_df["sim_driver_lat"].mean()
    source_lon = scenario_df["sim_driver_lon"].mean()
    dest_lat = scenario_df["sim_dest_lat"].mean()
    dest_lon = scenario_df["sim_dest_lon"].mean()

    center_lat = (source_lat + dest_lat) / 2
    center_lon = (source_lon + dest_lon) / 2
    m = folium.Map(location=[center_lat, center_lon], zoom_start=7, tiles="CartoDB Positron")

    folium.CircleMarker([source_lat, source_lon], radius=8, color="blue", fill=True, fill_color="blue",
                        popup=f"{title_prefix}Source Avg").add_to(m)
    folium.CircleMarker([dest_lat, dest_lon], radius=8, color="blue", fill=True, fill_color="blue",
                        popup=f"{title_prefix}Destination Avg").add_to(m)

    top_df = summary_df.head(50).copy()
    if len(top_df) > 0:
        top_df["combined_bucket"] = pd.qcut(
            top_df["combined_utility"].rank(method="first"), q=3, labels=["low", "mid", "high"]
        )
    else:
        top_df["combined_bucket"] = pd.Series(dtype="object")

    def get_color(bucket):
        if bucket == "high":
            return "green"
        if bucket == "mid":
            return "yellow"
        return "red"

    for _, row in top_df.iterrows():
        popup = f"""
        <b>{row['pinname']}</b><br>
        Simulation Rank: {row['simulation_rank']}<br>
        Combined Utility: {row['combined_utility']:.4f}<br>
        Feasible Rate: {row['feasible_rate']:.2%}<br>
        Top-10 Rate: {row['top_10_rate']:.2%}<br>
        Avg p_available: {row['avg_p_available']:.4f}<br>
        Avg Detour (mi): {row['avg_detour_mi']:.2f}
        """
        color = get_color(row["combined_bucket"])
        folium.CircleMarker(
            [row["lat"], row["lng"]], radius=6, color=color, fill=True,
            fill_color=color, fill_opacity=0.9,
            popup=folium.Popup(popup, max_width=320), tooltip=row["pinname"],
        ).add_to(m)

    bounds_points = [[source_lat, source_lon], [dest_lat, dest_lon]] + top_df[["lat", "lng"]].values.tolist()
    if len(bounds_points) >= 2:
        m.fit_bounds(bounds_points)

    legend_html = """
    <div style="position: fixed; bottom: 40px; left: 40px; width: 260px; z-index: 9999;
        background-color: white; border: 2px solid grey; border-radius: 8px; padding: 12px; font-size: 14px;">
        <b>Legend</b><br><br>
        <span style="color: blue;">●</span> Average source / destination<br>
        <span style="color: green;">●</span> High combined utility<br>
        <span style="color: yellow;">●</span> Mid combined utility<br>
        <span style="color: red;">●</span> Low combined utility
    </div>
    """
    m.get_root().html.add_child(Element(legend_html))
    return m


def show_simulation_results_page():
    st.subheader("Simulation Results")

    scenario_df = st.session_state.get("simulation_scenario_df")
    full_summary_df = st.session_state.get("simulation_summary_df")

    if scenario_df is None or full_summary_df is None or scenario_df.empty:
        st.info("Run the simulation first from the Simulation Setup page. Then the filtered results will appear here.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        hos_filter = st.selectbox("Filter by HOS hour", options=["All"] + HOS_HOURS, index=0)
    with col2:
        time_filter = st.selectbox("Filter by time window", options=["All", "morning", "afternoon", "evening"], index=0)
    with col3:
        top_n = st.slider("Rows to show", min_value=10, max_value=100, value=25, step=5)

    filtered_df = scenario_df.copy()
    if hos_filter != "All":
        filtered_df = filtered_df[filtered_df["hos_hour"] == int(hos_filter)].copy()
    if time_filter != "All":
        filtered_df = filtered_df[filtered_df["time_window"] == time_filter].copy()

    filtered_summary = aggregate_simulation_results(filtered_df)

    metric_cols = st.columns(4)
    metric_cols[0].metric("Scenario rows", f"{len(filtered_df):,}")
    metric_cols[1].metric("Unique stops", f"{filtered_df['pin id'].nunique():,}" if not filtered_df.empty else "0")
    metric_cols[2].metric("Runs included", f"{filtered_df['run_id'].nunique():,}" if not filtered_df.empty else "0")
    metric_cols[3].metric("Top stop utility",
                          f"{filtered_summary['combined_utility'].iloc[0]:.3f}" if not filtered_summary.empty else "NA")

    st.markdown("### Filtered stop summary")
    summary_cols = [
        "simulation_rank", "pinname", "combined_utility", "feasible_rate", "top_10_rate",
        "avg_p_available", "avg_detour_mi", "avg_truck_stop_mi", "avg_stop_dest_mi", "scenario_count"
    ]
    if not filtered_summary.empty:
        st.dataframe(filtered_summary[summary_cols].head(top_n), use_container_width=True)
    else:
        st.warning("No rows for the current filter.")

    st.markdown("### Filtered map")
    html(build_simulation_map(filtered_summary, filtered_df, title_prefix="Filtered ")._repr_html_(), height=700,
         scrolling=True)

    st.markdown("### Scenario-level results")
    scenario_cols = [
        "scenario_id", "run_id", "hos_hour", "time_window", "utility_rank", "pinname",
        "feasible_stop", "utility_score", "p_available", "detour_mi",
        "sim_hos_left_hr", "sim_amenity_weight"
    ]
    scenario_cols = [c for c in scenario_cols if c in filtered_df.columns]
    if not filtered_df.empty:
        st.dataframe(filtered_df[scenario_cols].head(top_n * 10), use_container_width=True)


st.title("Truck Stop Finder")
st.write("Single-run ranking, simulation setup, and filtered simulation results in one app.")

page = st.sidebar.radio("Page", ["Single Run", "Simulation Setup", "Simulation Results"], index=0)

st.sidebar.markdown("---")
driver_lat = st.sidebar.number_input("Driver Latitude", value=25.7752, format="%.6f")
driver_lon = st.sidebar.number_input("Driver Longitude", value=-80.2086, format="%.6f")
dest_lat = st.sidebar.number_input("Destination Latitude", value=33.76052, format="%.6f")
dest_lon = st.sidebar.number_input("Destination Longitude", value=-78.91463, format="%.6f")
freeflow_mph = st.sidebar.number_input("Freeflow Speed (mph)", value=55.0, min_value=1.0, step=1.0)

if page == "Single Run":
    hos_left_hr = st.sidebar.number_input("HOS Left (hours)", value=5.0, min_value=0.0, step=0.5)
    start_time = st.sidebar.text_input("Start Time (UTC)", value="2023-12-02 12:20:00")
    run_button = st.sidebar.button("Run Single Scenario")

    st.subheader("Single Run")
    st.caption("This is your original workflow, using normalized utility weights under the hood.")

    if run_button:
        with st.spinner("Running single scenario..."):
            df = run_single_cached(driver_lat, driver_lon, dest_lat, dest_lon, hos_left_hr, freeflow_mph, start_time)
            show_cols = [
                "utility_rank", "pinname", "utility_score", "feasible_stop", "p_available", "p_full",
                "truck_stop_mi", "stop_dest_mi", "detour_mi", "amenities_score", "truckParkingSpotCount"
            ]
            st.dataframe(df[show_cols].head(20), use_container_width=True)
            html(build_single_run_map(df)._repr_html_(), height=700, scrolling=True)

elif page == "Simulation Setup":
    num_runs = st.sidebar.number_input("Number of route variations", value=20, min_value=1, max_value=250, step=1)
    base_date = st.sidebar.text_input("Simulation Date", value="2023-12-02")
    base_amenity_weight = st.sidebar.slider("Base Amenity Weight", min_value=0.05, max_value=0.60, value=0.20,
                                            step=0.01)
    location_jitter_deg = st.sidebar.slider("Location Shift (+/- degrees)", min_value=0.01, max_value=1.00, value=0.20,
                                            step=0.01)
    amenity_jitter = st.sidebar.slider("Amenity Weight Shift (+/-)", min_value=0.0, max_value=0.30, value=0.10,
                                       step=0.01)
    seed = st.sidebar.number_input("Random Seed", value=42, min_value=0, step=1)
    sim_button = st.sidebar.button("Run Simulation")

    st.subheader("Simulation Setup")
    st.caption(
        "For each route variation, the app now runs all HOS scenarios from 1 to 6 hours, and for each HOS it evaluates morning, afternoon, and evening."
    )
    st.write(
        f"Each run creates {len(HOS_HOURS) * 3} scenarios: 6 HOS values × 3 time windows. "
        f"So with {int(num_runs)} route variations, you will get {int(num_runs) * len(HOS_HOURS) * 3} total scenarios."
    )

    if sim_button:
        with st.spinner("Running simulation across route, HOS, and time scenarios..."):
            summary_df, scenario_df = run_sim_cached(
                driver_lat,
                driver_lon,
                dest_lat,
                dest_lon,
                num_runs,
                freeflow_mph,
                base_date,
                base_amenity_weight,
                location_jitter_deg,
                amenity_jitter,
                seed,
            )
            st.session_state["simulation_summary_df"] = summary_df
            st.session_state["simulation_scenario_df"] = scenario_df

            st.success("Simulation finished. Open the Simulation Results page to explore HOS-specific outputs.")

            preview_cols = [
                "simulation_rank", "pinname", "combined_utility", "feasible_rate", "top_10_rate",
                "avg_p_available", "avg_detour_mi", "scenario_count"
            ]
            st.markdown("### Overall simulation preview")
            st.dataframe(summary_df[preview_cols].head(20), use_container_width=True)
            html(build_simulation_map(summary_df, scenario_df)._repr_html_(), height=700, scrolling=True)

else:
    show_simulation_results_page()
