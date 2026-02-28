import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scgraph.geographs.us_freeway import us_freeway_geograph
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

path = r"C:\Users\bhavy\Massachusetts Institute of Technology\Truck Parking Capstone - General\Truck Stop Finder 🚚⛽\\"
# path = r"C:\Users\samcl\Massachusetts Institute of Technology\Truck Parking Capstone - Truck Stop Finder 🚚⛽\\"

# Sourced directly from TruckerPath
park_data_1 = pd.read_csv(
    path + r"5. Source & Refrence Files\0. TruckerPath Data\MIT_2025_High_Volume_Routes_Parking_Data_1 - Copy.csv")
park_data_2 = pd.read_csv(
    path + r"5. Source & Refrence Files\0. TruckerPath Data\MIT_2025_High_Volume_Routes_Parking_Data_2 - Copy.csv")
park_data_3 = pd.read_csv(
    path + r"5. Source & Refrence Files\0. TruckerPath Data\MIT_2025_High_Volume_Routes_Parking_Data_3 - Copy.csv")
park_data = pd.concat([park_data_1, park_data_2, park_data_3], ignore_index=True)

truck_stop = pd.read_excel("output_excel\Model_Stops_V3.xlsx")

avail_park = park_data[park_data["pin id"].isin(truck_stop["pin id"].unique())].copy()

avail_park["ts_utc"] = pd.to_datetime(avail_park["time(utc)"], utc=True)

print(avail_park["ts_utc"].isna().mean(), "fraction of timestamps failed to parse")
print(avail_park["ts_utc"].min(), "to", avail_park["ts_utc"].max())

avail_park = avail_park[avail_park["parking status"] != 'Paid'].copy()

avail_park = avail_park.sort_values(["pin id", "ts_utc"])

status_map = {
    "Full": 0,
    "Some": 1,
    "Lots": 2
}

avail_park["status_ord"] = avail_park["parking status"].map(status_map)

LABEL_TOL = pd.Timedelta("60min")  # ±60 min for label lookup
STALE_CUTOFF = pd.Timedelta("6h")  # 6h staleness rule for last_known_status

obs = avail_park[["pin id", "ts_utc", "status_ord", "parking status"]].dropna(subset=["ts_utc", "status_ord"])
# obs = obs.sort_values(["pin id", "ts_utc"])
obs_sorted = obs.dropna(subset=["ts_utc"]).sort_values(["ts_utc", "pin id"]).reset_index(drop=True)


# ---- 2) Helper A: last observation at or before query time (decision-time features)
def attach_last_obs_before(query_df, time_col="query_ts"):
    """
    query_df must have columns: ['pin id', time_col]
    Returns query_df + last observed status at/before query_ts and staleness.
    """
    q = query_df.copy()
    q[time_col] = pd.to_datetime(q[time_col], utc=True, errors="coerce")
    # q = q.sort_values(["pin id", time_col], ignore_index=True)
    q = q.dropna(subset=[time_col]).sort_values([time_col, "pin id"]).reset_index(drop=True)

    out = pd.merge_asof(
        q,
        obs_sorted.rename(
            columns={"ts_utc": "last_ts", "status_ord": "last_status_ord", "parking status": "last_status_txt"}),
        left_on=time_col,
        right_on="last_ts",
        by="pin id",
        direction="backward",  # <= query time
        allow_exact_matches=True
    )

    out["time_since_last_obs_min"] = (out[time_col] - out["last_ts"]).dt.total_seconds() / 60

    # Apply 6h staleness rule: if too old, treat last_known_status as unknown (but keep staleness numeric)
    too_stale = (out[time_col] - out["last_ts"]) > STALE_CUTOFF
    out.loc[too_stale, ["last_status_ord", "last_status_txt", "last_ts"]] = pd.NA

    return out


# ---- 3) Helper B: nearest observation within ±60 minutes (labels around ETA)
def attach_label_nearest(query_df, time_col="eta_ts"):
    """
    query_df must have columns: ['pin id', time_col]
    Returns query_df + label status closest to eta_ts within ±60 minutes.
    If no observation within tolerance, label fields stay NA.
    """
    q = query_df.copy()
    q[time_col] = pd.to_datetime(q[time_col], utc=True, errors="coerce")
    # q = q.sort_values(["pin id", time_col])
    q = q.dropna(subset=[time_col]).sort_values([time_col, "pin id"]).reset_index(drop=True)

    out = pd.merge_asof(
        q,
        obs_sorted.rename(
            columns={"ts_utc": "label_ts", "status_ord": "label_status_ord", "parking status": "label_status_txt"}),
        left_on=time_col,
        right_on="label_ts",
        by="pin id",
        direction="nearest",
        tolerance=LABEL_TOL,
        allow_exact_matches=True
    )

    out["label_time_error_min"] = (out["label_ts"] - out[time_col]).abs().dt.total_seconds() / 60
    return out


# ---- Settings for the first batch
N_DECISIONS = 500_000  # start here; later you can go 500k / 1M
MIN_TAU = 15  # min travel time in minutes
MAX_TAU = 180  # max travel time in minutes

# 1) Build a "decision pool" from real observed timestamps
#    We sample from actual rows to get realistic t0 distribution.
decision_pool = avail_park[[
    "pin id", "ts_utc", "pinlat", "pinlon", "city", "route_num", "object"
]].dropna(subset=["pin id", "ts_utc"]).copy()

decisions = decision_pool.sample(N_DECISIONS, random_state=0).rename(columns={"ts_utc": "t_obs"}).copy()

rng = np.random.default_rng(0)

# unique stops (destination set)
stops = (
    avail_park[["pin id", "city", "route_num", "object", "pinlat", "pinlon"]]
    .dropna(subset=["pin id", "pinlat", "pinlon"])
    .drop_duplicates(subset=["pin id"])
    .reset_index(drop=True)
)
#
# # timestamp pool (realistic time-of-day / day-of-week distribution)
# t_pool = (
#     avail_park[["ts_utc"]]
#     .dropna()
#     .assign(ts_utc=lambda d: pd.to_datetime(d["ts_utc"], utc=True, errors="coerce"))
#     .dropna()
#     .reset_index(drop=True)
# )


N_DRIVERS = 10

drivers = pd.DataFrame({
    "driver_id": np.arange(N_DRIVERS)
})

# pick a random anchor stop per driver
anchors = stops.sample(N_DRIVERS, replace=True, random_state=2)[["pinlat", "pinlon"]].reset_index(drop=True)

# jitter ~ up to ~0.2 degrees (tune this!). 0.1 deg lat ~ 11km.
jitter_lat = rng.normal(loc=0.0, scale=0.15, size=N_DRIVERS)
jitter_lon = rng.normal(loc=0.0, scale=0.20, size=N_DRIVERS)

drivers["orig_lat"] = anchors["pinlat"] + jitter_lat
drivers["orig_lon"] = anchors["pinlon"] + jitter_lon

# Cross join: each driver paired with every stop
decisions = drivers.merge(decisions, how="cross")

# columns you now have:
# driver_id, t0, orig_lat, orig_lon, pin id, pinlat, pinlon, city, route_num, object
print(decisions.shape)

# Unique drivers
drivers_unique = (
    decisions[["driver_id", "orig_lat", "orig_lon"]]
    .drop_duplicates("driver_id")
    .reset_index(drop=True)
)

# Unique stops
stops_unique = (
    decisions[["pin id", "pinlat", "pinlon"]]
    .drop_duplicates("pin id")
    .reset_index(drop=True)
)

# Build coordinate list: drivers first, then stops
coords = []

# Drivers
for _, row in drivers_unique.iterrows():
    coords.append({
        "latitude": row["orig_lat"],
        "longitude": row["orig_lon"]
    })

# Stops
for _, row in stops_unique.iterrows():
    coords.append({
        "latitude": row["pinlat"],
        "longitude": row["pinlon"]
    })

distance_matrix = us_freeway_geograph.distance_matrix(
    coords,
    output_units="mi"
)

D = len(drivers_unique)
S = len(stops_unique)

records = []

for i in range(D):
    for j in range(S):
        records.append({
            "driver_id": drivers_unique.loc[i, "driver_id"],
            "pin id": stops_unique.loc[j, "pin id"],
            "distance_mi": distance_matrix[i][D + j]
        })

distance_df = pd.DataFrame(records)

print(decisions.shape)
decisions = decisions.merge(
    distance_df,
    on=["driver_id", "pin id"],
    how="left"
)
print(decisions.shape)

# 2) Sample travel time tau (minutes)

decisions["tau_min"] = decisions["distance_mi"] / 55
decisions.rename(columns={"t_obs": "t0"}, inplace=True)

# 3) Compute ETA
decisions["eta_ts"] = decisions["t0"] + pd.to_timedelta(decisions["tau_min"], unit="m")

# 4) Add ETA calendar features (these are allowed; you know ETA at decision time)
decisions["eta_hour"] = decisions["eta_ts"].dt.hour
decisions["eta_day_of_week"] = decisions["eta_ts"].dt.dayofweek  # Monday=0
decisions["eta_month"] = decisions["eta_ts"].dt.month

# 5) Attach last observation at/before t0 (features)
feat = attach_last_obs_before(
    decisions.rename(columns={"t0": "query_ts"}),
    time_col="query_ts"
).rename(columns={"query_ts": "t0"})

# 6) Attach label near ETA within ±60 minutes
lab = attach_label_nearest(
    feat[["pin id", "eta_ts"]].copy(),
    time_col="eta_ts"
)

# Merge labels back (by row order, since we kept same ordering)
feat["label_ts"] = lab["label_ts"].values
feat["label_status_ord"] = lab["label_status_ord"].values
feat["label_status_txt"] = lab["label_status_txt"].values
feat["label_time_error_min"] = lab["label_time_error_min"].values

# 7) Keep only rows with valid labels
train_batch = feat.dropna(subset=["label_status_ord"]).copy()

print("Decision rows:", len(decisions))
print("Training rows with labels:", len(train_batch))
print("Label coverage:", round(len(train_batch) / len(decisions), 3))

# 8) Keep just the columns we care about for modeling (first version)
train_batch = train_batch[[
    "pin id", "pinlat", "pinlon", "city", "route_num", "object",
    "t0", "tau_min", "eta_ts", "eta_hour", "eta_day_of_week", "eta_month",
    "last_status_ord", "time_since_last_obs_min",
    "label_status_ord", "label_time_error_min"
]].reset_index(drop=True)

train_batch.head()

n_all = avail_park["pin id"].nunique()
n_train = train_batch["pin id"].nunique()
print("Stops in raw data:", n_all)
print("Stops in training batch:", n_train)
print("Coverage:", n_train / n_all)

df = train_batch.copy()

df["y_full"] = (df["label_status_ord"] == 0).astype(int)
df = df.sort_values("t0").reset_index(drop=True)
cut = df["t0"].quantile(0.80)  # last 20% time as test (you can change later)
train_df = df[df["t0"] <= cut].copy()
test_df = df[df["t0"] > cut].copy()

print("Train rows:", len(train_df), "Test rows:", len(test_df))
print("Train Full rate:", train_df["y_full"].mean().round(3), "Test Full rate:", test_df["y_full"].mean().round(3))
print("Time split cutoff:", cut)

num_features = ["last_status_ord", "time_since_last_obs_min", "eta_hour", "eta_day_of_week", "eta_month"]
cat_features = ["route_num"]  # add "city" later if you want

X_train = train_df[num_features + cat_features]
y_train = train_df["y_full"]

X_test = test_df[num_features + cat_features]
y_test = test_df["y_full"]

numeric_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe, num_features),
        ("cat", categorical_pipe, cat_features)
    ],
    remainder="drop"
)

xgb = XGBClassifier(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    min_child_weight=1.0,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

base_model = Pipeline(steps=[("prep", preprocess), ("clf", xgb)])

# p_cal = cal_model.predict_proba(X_test)[:, 1]

# ---- Create a calibration split inside train (time-based)
train_df = train_df.sort_values("t0").reset_index(drop=True)
cal_cut = train_df["t0"].quantile(0.80)  # last 20% of TRAIN becomes calibration set

fit_df = train_df[train_df["t0"] <= cal_cut].copy()
cal_df = train_df[train_df["t0"] > cal_cut].copy()

X_fit, y_fit = fit_df[num_features + cat_features], fit_df["y_full"]
X_cal, y_cal = cal_df[num_features + cat_features], cal_df["y_full"]

print("\nFit rows:", len(fit_df), "Cal rows:", len(cal_df))
print("Fit Full rate:", round(y_fit.mean(), 3), "Cal Full rate:", round(y_cal.mean(), 3))
print("Calibration cutoff:", cal_cut)

base_model.fit(X_fit, y_fit)

p_uncal = base_model.predict_proba(X_test)[:, 1]

cal_model = CalibratedClassifierCV(base_model, method="sigmoid", cv="prefit")
cal_model.fit(X_cal, y_cal)

# ---------- 4) Calibration curve plot (reliability diagram) ----------
# Using sklearn.calibration.calibration_curve (like the link you shared)
n_bins = 10
frac_pos_uncal, mean_pred_uncal = calibration_curve(y_test, p_uncal, n_bins=n_bins, strategy="uniform")
# frac_pos_cal, mean_pred_cal = calibration_curve(y_test, p_cal, n_bins=n_bins, strategy="uniform")

plt.figure(figsize=(7, 6))
# Perfect calibration line
plt.plot([0, 1], [0, 1], linestyle="--")

# Model curves
plt.plot(mean_pred_uncal, frac_pos_uncal, marker="o", label="XGBoost (uncalibrated)")
# plt.plot(mean_pred_cal, frac_pos_cal, marker="o", label="XGBoost (calibrated)")

plt.title("Calibration Curve (Reliability Diagram)")
plt.xlabel("Mean Predicted Probability (bin)")
plt.ylabel("Fraction of Positives (bin)")
plt.legend()
plt.tight_layout()
plt.show()

threshold = 0.33
y_pred = (p_uncal >= threshold).astype(int)

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix (rows=true, cols=pred):")
print(cm)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

print("Confusion Matrix (rows=true, cols=pred):")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=3))
