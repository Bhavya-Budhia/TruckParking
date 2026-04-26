import altair as alt
import folium
import numpy as np
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


def get_relevant_simulation_stops(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Keep only stops that are feasible in at least one included scenario."""
    if summary_df.empty:
        return summary_df.copy()

    relevant_df = summary_df[
        (summary_df["feasible_rate"] > 0)
        & (summary_df["combined_utility"] > 0)
    ].copy()

    relevant_df = relevant_df.sort_values(
        ["combined_utility", "feasible_rate", "avg_p_available"],
        ascending=False,
    ).reset_index(drop=True)
    relevant_df["simulation_rank"] = range(1, len(relevant_df) + 1)
    return relevant_df


def add_utility_buckets(df: pd.DataFrame, score_col: str = "combined_utility") -> pd.DataFrame:
    """Create red/yellow/green buckets only for the relevant stops shown."""
    out = df.copy()
    n = len(out)

    if n == 0:
        out["combined_bucket"] = pd.Series(dtype="object")
    elif n == 1:
        out["combined_bucket"] = "high"
    elif n == 2:
        out["combined_bucket"] = ["high", "low"]
    else:
        out["combined_bucket"] = pd.qcut(
            out[score_col].rank(method="first"),
            q=3,
            labels=["low", "mid", "high"],
        )
    return out


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

    map_df = add_utility_buckets(get_relevant_simulation_stops(summary_df))

    def get_color(bucket):
        if bucket == "high":
            return "green"
        if bucket == "mid":
            return "yellow"
        return "red"

    for _, row in map_df.iterrows():
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

    bounds_points = [[source_lat, source_lon], [dest_lat, dest_lon]] + map_df[["lat", "lng"]].values.tolist()
    if len(bounds_points) >= 2:
        m.fit_bounds(bounds_points)

    legend_html = """
    <div style="position: fixed; bottom: 40px; left: 40px; width: 260px; z-index: 9999;
        background-color: white; border: 2px solid grey; border-radius: 8px; padding: 12px; font-size: 14px;">
        <b>Legend</b><br><br>
        <span style="color: blue;">●</span> Average source / destination<br>
        <span style="color: green;">●</span> High relevant feasible stop<br>
        <span style="color: yellow;">●</span> Mid relevant feasible stop<br>
        <span style="color: red;">●</span> Low relevant feasible stop
    </div>
    """
    m.get_root().html.add_child(Element(legend_html))
    return m


FRONTIER_COLORS = {
    1: "#8e44ad",
    2: "#2980b9",
    3: "#16a085",
    4: "#f39c12",
    5: "#d35400",
    6: "#c0392b",
}


def weighted_average_distance(distance_series: pd.Series, weight_series: pd.Series) -> float:
    """Return a utility-weighted average distance, with safe fallbacks."""
    distances = pd.to_numeric(distance_series, errors="coerce")
    weights = pd.to_numeric(weight_series, errors="coerce").fillna(0)

    valid = distances.notna()
    distances = distances[valid]
    weights = weights[valid]

    if len(distances) == 0:
        return 0.0

    if weights.sum() <= 0:
        return float(distances.mean())

    return float(np.average(distances, weights=weights))


def build_hos_frontier_summary(scenario_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize stop color counts and a data-driven frontier distance for each HOS hour.

    The frontier distance is based on the average distance of relevant stops available to the
    driver under each HOS scenario, weighted by combined utility and feasible rate. This makes
    the circle describe the reachable stop frontier rather than only speed × HOS.
    """
    rows = []

    for hos_hour in HOS_HOURS:
        hos_df = scenario_df[scenario_df["hos_hour"] == int(hos_hour)].copy()
        if hos_df.empty:
            rows.append({
                "hos_hour": int(hos_hour),
                "raw_frontier_distance_mi": 0.0,
                "frontier_distance_mi": 0.0,
                "green_stops": 0,
                "yellow_stops": 0,
                "red_stops": 0,
                "total_relevant_stops": 0,
            })
            continue

        hos_summary = aggregate_simulation_results(hos_df)
        relevant_hos_summary = add_utility_buckets(get_relevant_simulation_stops(hos_summary))

        bucket_counts = relevant_hos_summary[
            "combined_bucket"].value_counts() if not relevant_hos_summary.empty else pd.Series(dtype=int)

        if not relevant_hos_summary.empty:
            distance_weights = (
                    relevant_hos_summary["combined_utility"].fillna(0)
                    * relevant_hos_summary["feasible_rate"].fillna(0)
            )
            raw_frontier_distance_mi = weighted_average_distance(
                relevant_hos_summary["avg_truck_stop_mi"],
                distance_weights,
            )
        elif "adj_speed_mph" in hos_df.columns:
            raw_frontier_distance_mi = float((hos_df["sim_hos_left_hr"] * hos_df["adj_speed_mph"]).mean())
        else:
            raw_frontier_distance_mi = float(hos_hour) * float(freeflow_mph)

        rows.append({
            "hos_hour": int(hos_hour),
            "raw_frontier_distance_mi": raw_frontier_distance_mi,
            "frontier_distance_mi": raw_frontier_distance_mi,
            "green_stops": int(bucket_counts.get("high", 0)),
            "yellow_stops": int(bucket_counts.get("mid", 0)),
            "red_stops": int(bucket_counts.get("low", 0)),
            "total_relevant_stops": int(len(relevant_hos_summary)),
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        # The visible frontier should move outward or stay flat as HOS increases.
        out["frontier_distance_mi"] = out["raw_frontier_distance_mi"].cummax()

    return out


def build_hos_frontier_map(scenario_df: pd.DataFrame, frontier_summary: pd.DataFrame):
    if scenario_df.empty:
        return folium.Map(location=[39.5, -98.35], zoom_start=4, tiles="CartoDB Positron")

    source_lat = scenario_df["sim_driver_lat"].mean()
    source_lon = scenario_df["sim_driver_lon"].mean()
    dest_lat = scenario_df["sim_dest_lat"].mean()
    dest_lon = scenario_df["sim_dest_lon"].mean()

    center_lat = (source_lat + dest_lat) / 2
    center_lon = (source_lon + dest_lon) / 2
    m = folium.Map(location=[center_lat, center_lon], zoom_start=7, tiles="CartoDB Positron")

    folium.CircleMarker(
        [source_lat, source_lon], radius=8, color="blue", fill=True, fill_color="blue",
        popup="Average Source"
    ).add_to(m)
    folium.CircleMarker(
        [dest_lat, dest_lon], radius=8, color="blue", fill=True, fill_color="blue",
        popup="Average Destination"
    ).add_to(m)

    # Draw frontier rings first so the stop markers sit on top.
    for _, row in frontier_summary.iterrows():
        hos_hour = int(row["hos_hour"])
        color = FRONTIER_COLORS.get(hos_hour, "gray")
        radius_m = float(row["frontier_distance_mi"]) * 1609.344
        folium.Circle(
            location=[source_lat, source_lon],
            radius=radius_m,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.10,
            weight=3,
            opacity=0.75,
            popup=(
                f"HOS {hos_hour}: displayed frontier {row['frontier_distance_mi']:.1f} mi "
                f"(raw weighted stop distance {row['raw_frontier_distance_mi']:.1f} mi)"
            ),
        ).add_to(m)

    # Show each relevant stop once, colored using its overall simulation bucket.
    overall_summary = add_utility_buckets(get_relevant_simulation_stops(aggregate_simulation_results(scenario_df)))

    def get_color(bucket):
        if bucket == "high":
            return "green"
        if bucket == "mid":
            return "yellow"
        return "red"

    for _, row in overall_summary.iterrows():
        color = get_color(row["combined_bucket"])
        popup = f"""
        <b>{row['pinname']}</b><br>
        Simulation Rank: {row['simulation_rank']}<br>
        Combined Utility: {row['combined_utility']:.4f}<br>
        Feasible Rate: {row['feasible_rate']:.2%}<br>
        Avg p_available: {row['avg_p_available']:.4f}<br>
        Avg Truck Stop Miles: {row['avg_truck_stop_mi']:.2f}
        """
        folium.CircleMarker(
            [row["lat"], row["lng"]], radius=6, color=color, fill=True,
            fill_color=color, fill_opacity=0.9,
            popup=folium.Popup(popup, max_width=320), tooltip=row["pinname"],
        ).add_to(m)

    bounds_points = [[source_lat, source_lon], [dest_lat, dest_lon]]
    if not overall_summary.empty:
        bounds_points += overall_summary[["lat", "lng"]].values.tolist()
    m.fit_bounds(bounds_points)

    return m


def render_hos_frontier_side_legend(frontier_summary: pd.DataFrame):
    """Render the HOS frontier legend outside the Folium map.

    This uses Streamlit's HTML component instead of st.markdown, so the HTML
    cannot accidentally be interpreted as a Markdown code block.
    """
    frontier_rows = []
    if frontier_summary is not None and not frontier_summary.empty:
        for _, row in frontier_summary.iterrows():
            hos_hour = int(row["hos_hour"])
            color = FRONTIER_COLORS.get(hos_hour, "gray")
            frontier_rows.append(
                f'<div class="legend-row">'
                f'<span class="legend-dot" style="background:{color};"></span>'
                f'<span>HOS {hos_hour}</span>'
                f'<span class="legend-value">{row["frontier_distance_mi"]:.1f} mi</span>'
                f'</div>'
            )

    frontier_html = "".join(frontier_rows) if frontier_rows else "<p>No frontier data available.</p>"

    legend_html = f"""<!DOCTYPE html>
<html>
<head>
<style>
body {{
    margin: 0;
    padding: 0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    color: #31333f;
}}
.legend-card {{
    background: #ffffff;
    border: 1px solid rgba(49, 51, 63, 0.16);
    border-radius: 18px;
    padding: 18px 16px;
    box-shadow: 0 8px 22px rgba(0, 0, 0, 0.08);
}}
.legend-title {{
    font-size: 20px;
    font-weight: 800;
    margin-bottom: 14px;
}}
.legend-section-title {{
    font-size: 13px;
    font-weight: 800;
    color: #596579;
    margin-top: 14px;
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}}
.legend-row {{
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 8px 0;
    font-size: 14px;
}}
.legend-dot {{
    width: 12px;
    height: 12px;
    border-radius: 999px;
    display: inline-block;
    border: 1px solid rgba(0, 0, 0, 0.2);
    flex: 0 0 12px;
}}
.legend-value {{
    margin-left: auto;
    font-weight: 700;
    color: #2f3b52;
}}
.legend-note {{
    margin-top: 16px;
    padding: 12px;
    border-radius: 12px;
    background: #f6f8fb;
    color: #465268;
    font-size: 13px;
    line-height: 1.35;
}}
</style>
</head>
<body>
<div class="legend-card">
    <div class="legend-title">Map Legend</div>
    <div class="legend-section-title">Stops</div>
    <div class="legend-row"><span class="legend-dot" style="background:green;"></span><span>High utility</span></div>
    <div class="legend-row"><span class="legend-dot" style="background:gold;"></span><span>Medium utility</span></div>
    <div class="legend-row"><span class="legend-dot" style="background:red;"></span><span>Low utility</span></div>
    <div class="legend-row"><span class="legend-dot" style="background:blue;"></span><span>Avg. source / destination</span></div>
    <div class="legend-section-title">HOS Frontier</div>
    {frontier_html}
    <div class="legend-note">Filled zones show the data-driven reachable frontier. Radius is based on relevant feasible stops, then made cumulative so the frontier does not shrink as HOS increases.</div>
</div>
</body>
</html>"""

    html(legend_html, height=520, scrolling=False)

def show_hos_frontier_page():
    st.subheader("HOS Frontier Movement")

    scenario_df = st.session_state.get("simulation_scenario_df")
    if scenario_df is None or scenario_df.empty:
        st.info("Run the simulation first from the Simulation Setup page. Then the HOS frontier will appear here.")
        return

    st.caption(
        "Each filled zone shows the data-driven frontier for that HOS hour, based on the weighted average distance "
        "of relevant feasible stops available to the driver. The visible frontier is cumulative, so it moves outward as HOS increases. "
        "Stop colors use the same relevant-stop red/yellow/green buckets used in Simulation Results."
    )

    frontier_summary = build_hos_frontier_summary(scenario_df)

    st.markdown("### Frontier map")
    map_col, legend_col = st.columns([4, 1.15])
    with map_col:
        html(build_hos_frontier_map(scenario_df, frontier_summary)._repr_html_(), height=750, scrolling=True)
    with legend_col:
        render_hos_frontier_side_legend(frontier_summary)

    st.markdown("### Relevant stop counts by HOS")
    display_df = frontier_summary[[
        "hos_hour", "raw_frontier_distance_mi", "frontier_distance_mi",
        "green_stops", "yellow_stops", "red_stops", "total_relevant_stops"
    ]].copy()
    display_df.rename(columns={
        "hos_hour": "HOS Hour",
        "raw_frontier_distance_mi": "Weighted Stop Distance (mi)",
        "frontier_distance_mi": "Displayed Frontier Distance (mi)",
        "green_stops": "Green Stops",
        "yellow_stops": "Yellow Stops",
        "red_stops": "Red Stops",
        "total_relevant_stops": "Total Relevant Stops",
    }, inplace=True)
    st.dataframe(display_df, use_container_width=True)

    st.markdown("### Stop count trend by HOS")
    chart_df = frontier_summary[["hos_hour", "green_stops", "yellow_stops", "red_stops"]].copy()
    chart_df.rename(columns={
        "hos_hour": "HOS Hour",
        "green_stops": "Green",
        "yellow_stops": "Yellow",
        "red_stops": "Red",
    }, inplace=True)

    chart_long = chart_df.melt(
        id_vars="HOS Hour",
        value_vars=["Green", "Yellow", "Red"],
        var_name="Stop Category",
        value_name="Stop Count",
    )

    trend_chart = (
        alt.Chart(chart_long)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X("HOS Hour:O", title="HOS Hour"),
            y=alt.Y("Stop Count:Q", title="Number of Relevant Stops"),
            color=alt.Color(
                "Stop Category:N",
                scale=alt.Scale(
                    domain=["Green", "Yellow", "Red"],
                    range=["green", "gold", "red"],
                ),
                title="Stop Category",
            ),
            tooltip=["HOS Hour:O", "Stop Category:N", "Stop Count:Q"],
        )
        .properties(height=360)
    )
    st.altair_chart(trend_chart, use_container_width=True)

def show_simulation_results_page():
    st.subheader("Simulation Results")

    scenario_df = st.session_state.get("simulation_scenario_df")
    full_summary_df = st.session_state.get("simulation_summary_df")

    if scenario_df is None or full_summary_df is None or scenario_df.empty:
        st.info("Run the simulation first from the Simulation Setup page. Then the filtered results will appear here.")
        return

    col1, col2 = st.columns(2)
    with col1:
        hos_filter = st.selectbox("Filter by HOS hour", options=["All"] + HOS_HOURS, index=0)
    with col2:
        time_filter = st.selectbox("Filter by time window", options=["All", "morning", "afternoon", "evening"], index=0)

    filtered_df = scenario_df.copy()
    if hos_filter != "All":
        filtered_df = filtered_df[filtered_df["hos_hour"] == int(hos_filter)].copy()
    if time_filter != "All":
        filtered_df = filtered_df[filtered_df["time_window"] == time_filter].copy()

    filtered_summary = aggregate_simulation_results(filtered_df)
    relevant_summary = get_relevant_simulation_stops(filtered_summary)
    feasible_scenario_df = filtered_df[
        filtered_df["feasible_stop"] == 1].copy() if not filtered_df.empty else filtered_df

    metric_cols = st.columns(4)
    metric_cols[0].metric("Scenario rows", f"{len(filtered_df):,}")
    metric_cols[1].metric("Relevant feasible stops", f"{len(relevant_summary):,}")
    metric_cols[2].metric("Runs included", f"{filtered_df['run_id'].nunique():,}" if not filtered_df.empty else "0")
    metric_cols[3].metric("Top stop utility",
                          f"{relevant_summary['combined_utility'].iloc[0]:.3f}" if not relevant_summary.empty else "NA")

    st.markdown("### Filtered stop summary")
    summary_cols = [
        "simulation_rank", "pinname", "combined_utility", "feasible_rate", "top_10_rate",
        "avg_p_available", "avg_detour_mi", "avg_truck_stop_mi", "avg_stop_dest_mi", "scenario_count"
    ]
    if not relevant_summary.empty:
        st.dataframe(relevant_summary[summary_cols], use_container_width=True)
    else:
        st.warning("No feasible/relevant stops for the current filter.")

    st.markdown("### Filtered map")
    html(build_simulation_map(relevant_summary, filtered_df, title_prefix="Filtered ")._repr_html_(), height=700,
         scrolling=True)

    st.markdown("### Scenario-level feasible results")
    scenario_cols = [
        "scenario_id", "run_id", "hos_hour", "time_window", "utility_rank", "pinname",
        "feasible_stop", "utility_score", "p_available", "detour_mi",
        "sim_hos_left_hr", "sim_amenity_weight"
    ]
    scenario_cols = [c for c in scenario_cols if c in feasible_scenario_df.columns]
    if not feasible_scenario_df.empty:
        st.dataframe(feasible_scenario_df[scenario_cols], use_container_width=True)
    else:
        st.warning("No feasible scenario-level rows for the current filter.")


st.title("Truck Stop Finder")
st.write("Single-run ranking, simulation setup, filtered simulation results, and HOS frontier analysis in one app.")

st.markdown("""
<style>
button[data-baseweb="tab"] {
    padding: 12px 22px;
    font-size: 16px;
    font-weight: 600;
    border-radius: 999px 999px 0 0;
}
button[data-baseweb="tab"][aria-selected="true"] {
    border-bottom: 4px solid #1f77b4;
    color: #1f77b4;
    font-weight: 800;
}
div[data-baseweb="tab-list"] {
    gap: 8px;
}
.legend-card {
    background: #ffffff;
    border: 1px solid rgba(49, 51, 63, 0.16);
    border-radius: 18px;
    padding: 18px 16px;
    box-shadow: 0 8px 22px rgba(0, 0, 0, 0.08);
    margin-top: 2px;
}
.legend-title {
    font-size: 20px;
    font-weight: 800;
    margin-bottom: 14px;
}
.legend-section-title {
    font-size: 13px;
    font-weight: 800;
    color: #596579;
    margin-top: 14px;
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.legend-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 8px 0;
    font-size: 14px;
}
.legend-dot {
    width: 12px;
    height: 12px;
    border-radius: 999px;
    display: inline-block;
    border: 1px solid rgba(0, 0, 0, 0.2);
    flex: 0 0 12px;
}
.legend-value {
    margin-left: auto;
    font-weight: 700;
    color: #2f3b52;
}
.legend-note {
    margin-top: 16px;
    padding: 12px;
    border-radius: 12px;
    background: #f6f8fb;
    color: #465268;
    font-size: 13px;
    line-height: 1.35;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Global sidebar inputs shared by all tabs
# --------------------------------------------------
st.sidebar.markdown("### Route Inputs")
driver_lat = st.sidebar.number_input("Driver Latitude", value=25.7752, format="%.6f")
driver_lon = st.sidebar.number_input("Driver Longitude", value=-80.2086, format="%.6f")
dest_lat = st.sidebar.number_input("Destination Latitude", value=33.76052, format="%.6f")
dest_lon = st.sidebar.number_input("Destination Longitude", value=-78.91463, format="%.6f")
freeflow_mph = st.sidebar.number_input("Freeflow Speed (mph)", value=55.0, min_value=1.0, step=1.0)

tabs = st.tabs([
    "🚚 Single Run",
    "⚙️ Simulation Setup",
    "📊 Simulation Results",
    "🧭 HOS Frontier",
])

# --------------------------------------------------
# Tab 1: Single Run
# --------------------------------------------------
with tabs[0]:
    st.subheader("Single Run")
    st.caption("This is your original workflow, using normalized utility weights under the hood.")

    c1, c2, c3 = st.columns([1, 2, 1])
    with c1:
        hos_left_hr = st.number_input("HOS Left (hours)", value=5.0, min_value=0.0, step=0.5, key="single_hos")
    with c2:
        start_time = st.text_input("Start Time (UTC)", value="2023-12-02 12:20:00", key="single_start_time")
    with c3:
        st.markdown("<br>", unsafe_allow_html=True)
        run_button = st.button("Run Single Scenario", key="run_single_button", use_container_width=True)

    if run_button:
        with st.spinner("Running single scenario..."):
            df = run_single_cached(driver_lat, driver_lon, dest_lat, dest_lon, hos_left_hr, freeflow_mph, start_time)
            show_cols = [
                "utility_rank", "pinname", "utility_score", "feasible_stop", "p_available", "p_full",
                "truck_stop_mi", "stop_dest_mi", "detour_mi", "amenities_score", "truckParkingSpotCount"
            ]
            st.dataframe(df[show_cols].head(20), use_container_width=True)
            html(build_single_run_map(df)._repr_html_(), height=700, scrolling=True)

# --------------------------------------------------
# Tab 2: Simulation Setup
# --------------------------------------------------
with tabs[1]:
    st.subheader("Simulation Setup")
    st.caption(
        "For each route variation, the app runs all HOS scenarios from 1 to 6 hours, "
        "and for each HOS it evaluates morning, afternoon, and evening."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        num_runs = st.number_input("Number of route variations", value=20, min_value=1, max_value=250, step=1,
                                   key="sim_num_runs")
        base_date = st.text_input("Simulation Date", value="2023-12-02", key="sim_base_date")
    with c2:
        base_amenity_weight = st.slider("Base Amenity Weight", min_value=0.05, max_value=0.60, value=0.20,
                                        step=0.01, key="sim_base_amenity_weight")
        amenity_jitter = st.slider("Amenity Weight Shift (+/-)", min_value=0.0, max_value=0.30, value=0.10,
                                   step=0.01, key="sim_amenity_jitter")
    with c3:
        location_jitter_deg = st.slider("Location Shift (+/- degrees)", min_value=0.01, max_value=1.00, value=0.20,
                                        step=0.01, key="sim_location_jitter")
        seed = st.number_input("Random Seed", value=42, min_value=0, step=1, key="sim_seed")

    st.write(
        f"Each run creates {len(HOS_HOURS) * 3} scenarios: 6 HOS values × 3 time windows. "
        f"So with {int(num_runs)} route variations, you will get {int(num_runs) * len(HOS_HOURS) * 3} total scenarios."
    )

    sim_button = st.button("Run Simulation", key="run_sim_button", use_container_width=True)

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

            st.success("Simulation finished. Open the Simulation Results or HOS Frontier tab to explore outputs.")

            preview_cols = [
                "simulation_rank", "pinname", "combined_utility", "feasible_rate", "top_10_rate",
                "avg_p_available", "avg_detour_mi", "scenario_count"
            ]
            st.markdown("### Overall simulation preview")
            st.dataframe(summary_df[preview_cols].head(20), use_container_width=True)
            html(build_simulation_map(summary_df, scenario_df)._repr_html_(), height=700, scrolling=True)

# --------------------------------------------------
# Tab 3: Simulation Results
# --------------------------------------------------
with tabs[2]:
    show_simulation_results_page()

# --------------------------------------------------
# Tab 4: HOS Frontier
# --------------------------------------------------
with tabs[3]:
    show_hos_frontier_page()
