# Truck Parking Finder 🚚⛽

A machine learning and simulation-based truck parking recommendation system developed as part of the MIT Supply Chain
Management Capstone Project.

---

# Project Overview

This system combines:

- Historical truck parking availability
- Traffic congestion forecasting
- Hours-of-Service (HOS) constraints
- Route feasibility analysis
- Truck stop amenities
- Parking capacity
- Monte Carlo simulation

to recommend the most suitable truck parking locations for a driver traveling between a source and destination.

---

# Features

## 🚚 Single Route Analysis

Evaluate a specific route using:

- Driver location
- Destination location
- Remaining HOS
- Start time
- Average speed

Outputs:

- Ranked truck stops
- Parking availability probability
- Utility score
- Feasibility status
- Interactive map

---

## ⚙️ Simulation Engine

Run hundreds of route variations by changing:

- Driver starting location
- Destination location
- HOS values (1–6 hours)
- Time of day
- Amenity preferences

Outputs:

- Robust truck stop rankings
- Feasibility rates
- Top-10 appearance rates
- Combined utility scores

---

## 🧭 HOS Frontier Analysis

Visualize how reachable truck parking territory expands as HOS increases.

Outputs:

- H3 frontier maps
- Reachability zones
- Utility-weighted distance trends
- Frontier growth charts

---

# Repository Structure

```text
TruckParking/
│
├── app_v2.py
├── model_engine_v2.py
├── simulation_engine.py
├── parking_avail.py
├── model_stops.py
├── cong_speed.py
│
├── stop_tab.csv
├── Congestion_speed_r_2.csv
├── Congestion_speed_r_3.csv
├── Congestion_speed_r_4.csv
│
└── output_excel/
    ├── parking_availability_model.joblib
    └── parking_obs_sorted.parquet
```

---

# File Descriptions

## app_v2.py

Main Streamlit application.

Contains four tabs:

### 🚚 Single Run

Evaluate one route scenario.

### ⚙️ Simulation Setup

Run route variations across multiple HOS values and times of day.

### 📊 Simulation Results

Analyze simulation outputs using HOS and time-of-day filters.

### 🧭 HOS Frontier

Visualize reachability expansion as HOS increases.

---

## model_engine_v2.py

Core recommendation engine.

Responsibilities:

- Route distance calculations
- Congestion adjustment using BPR functions
- Parking availability prediction
- Feasibility screening
- Utility score calculation
- Final truck stop ranking

---

## simulation_engine.py

Simulation framework.

For every route variation:

- Randomly shifts source location
- Randomly shifts destination location
- Varies HOS from 1–6 hours
- Evaluates morning, afternoon, and evening departures

Produces:

- Scenario-level outputs
- Aggregated rankings
- Robustness metrics

---

## parking_avail.py

Parking availability model training pipeline.

### Creates Training Dataset

Builds synthetic decision scenarios from historical observations.

### Features

- Last observed parking status
- Observation staleness
- ETA hour
- ETA day of week
- ETA month
- Route identifier

### Model

XGBoost Classifier

### Outputs

```text
output_excel/
├── parking_availability_model.joblib
└── parking_obs_sorted.parquet
```

---

## model_stops.py

Truck stop preprocessing pipeline.

Responsibilities:

- Truck stop consolidation
- Capacity reconciliation
- Duplicate removal
- Amenity scoring

Output:

```text
stop_tab.csv
```

---

## cong_speed.py

Traffic preprocessing pipeline.

Responsibilities:

- ATR traffic processing
- Sensor-to-H3 mapping
- Traffic aggregation
- Congestion statistics

Outputs:

```text
Congestion_speed_r_2.csv
Congestion_speed_r_3.csv
Congestion_speed_r_4.csv
```

---

# Methodology

## Parking Availability Prediction

The parking prediction model classifies:

- 1 = Parking Full
- 0 = Parking Available

Features include:

- Last observed parking status
- Time since observation
- ETA hour
- ETA day of week
- ETA month
- Route number

Model:

```text
XGBoost Classifier
```

---

## Feasibility Logic

A stop is considered feasible only if:

```text
HOS not exceeded
AND
Stop is before destination
AND
Stop is not behind origin
AND
Parking availability ≥ 40%
```

---

## Utility Function

The final truck stop utility score combines:

| Component             | Description                      |
|-----------------------|----------------------------------|
| Parking Score         | Parking availability probability |
| Amenity Score         | Quality of truck stop amenities  |
| Capacity Score        | Number of parking spaces         |
| Detour Score          | Route efficiency                 |
| Traffic Score         | Congestion conditions            |
| Remaining Route Score | Progress toward destination      |
| HOS Utilization Score | Efficient use of available HOS   |

Default weights:

| Component       | Weight |
|-----------------|--------|
| Parking         | 0.20   |
| Amenities       | 0.15   |
| Capacity        | 0.15   |
| Detour          | 0.20   |
| Traffic         | 0.10   |
| Remaining Route | 0.09   |
| HOS Utilization | 0.10   |

---

# Simulation Ranking Logic

For each truck stop:

- Average Utility
- Worst-Case Utility (10th percentile)
- Feasible Rate
- Top-10 Appearance Rate

Combined Utility:

```text
0.50 × Average Utility
+ 0.20 × Feasible Rate
+ 0.15 × Top-10 Rate
+ 0.15 × Worst-Case Utility
```

---

# Understanding Results

## Single Run Map

### Green

High utility feasible stop

### Yellow

Medium utility feasible stop

### Red

Low utility feasible stop

### Black

Infeasible stop

Possible reasons:

- HOS exceeded
- Beyond destination
- Behind source
- Low parking availability

---

## Simulation Results

### Combined Utility

Overall robustness score.

### Feasible Rate

Percentage of scenarios where a stop remains feasible.

### Top-10 Rate

Percentage of scenarios where a stop appears in the Top 10.

---

## HOS Frontier

Shows how reachable truck parking territory expands as HOS increases.

Each H3 cell is colored according to the first HOS level at which it becomes reachable.

---

# Installation

## Clone Repository

```bash
git clone https://github.com/Bhavya-Budhia/TruckParking.git
cd TruckParking
```

## Create Environment

```bash
conda create -n truck_parking python=3.11
conda activate truck_parking
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

Suggested packages:

```text
streamlit
pandas
numpy
scikit-learn
xgboost
joblib
duckdb
folium
h3
altair
matplotlib
seaborn
pyarrow
haversine
scgraph
```

---

# Data Preparation Workflow

## Step 1

```bash
python model_stops.py
```

Creates:

```text
stop_tab.csv
```

## Step 2

```bash
python cong_speed.py
```

Creates:

```text
Congestion_speed_r_2.csv
Congestion_speed_r_3.csv
Congestion_speed_r_4.csv
```

## Step 3

```bash
python parking_avail.py
```

Creates:

```text
output_excel/
├── parking_availability_model.joblib
└── parking_obs_sorted.parquet
```

## Step 4

```bash
streamlit run app_v2.py
```

---

# Application Workflow

## Single Run

1. Enter source and destination coordinates
2. Enter HOS remaining
3. Enter start time
4. Click **Run Single Scenario**

## Simulation

1. Enter route coordinates
2. Select number of route variations
3. Configure simulation settings
4. Click **Run Simulation**

## Explore Results

### Simulation Results Tab

Filter by:

- HOS
- Time window

### HOS Frontier Tab

Analyze:

- Reachability expansion
- Frontier movement
- Utility-weighted distance growth

---

# Future Enhancements

- Real-time parking feeds
- Weather integration
- Dynamic HOS regulations
- Driver preference learning
- Autonomous vehicle routing support
- Parking demand forecasting

---

# Author

**Bhavya Budhia**
**Sam Clarke**

MIT Supply Chain Management Capstone Project
