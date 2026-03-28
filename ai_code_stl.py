import os
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# ===============================
# CONFIG
# ===============================
stl_json_folder = "output"
output_file = "stl_ai_insight.json"
Z_THRESHOLD = 2.5

# ===============================
# LOAD FILES
# ===============================
files = [
    f for f in os.listdir(stl_json_folder)
    if f.endswith(".json") and "mesh_validation" not in f
]

if len(files) < 2:
    raise ValueError("Add at least 2 STL validation JSON files.")

data = []
raw_data = []

for file in files:
    path = os.path.join(stl_json_folder, file)

    with open(path) as f:
        j = json.load(f)

    triangles = float(j.get("triangle_count") or 0)
    degenerate = float(j.get("degenerate_count") or 0)
    size = float(j.get("file_size_bytes") or 0)

    # bounding box size (volume proxy)
    bbox = j.get("bounding_box", {})
    min_vals = bbox.get("min", [0,0,0])
    max_vals = bbox.get("max", [0,0,0])

    if len(min_vals) >= 3 and len(max_vals) >= 3:
        volume_proxy = abs((max_vals[0]-min_vals[0]) *
                           (max_vals[1]-min_vals[1]) *
                           (max_vals[2]-min_vals[2]))
    else:
        volume_proxy = 0

    data.append([triangles, degenerate, size, volume_proxy])
    raw_data.append(j)

X = np.array(data)

# ===============================
# Z-SCORE
# ===============================
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
z_scores = np.abs((X - mean) / (std + 1e-9))
z_outlier = (z_scores > Z_THRESHOLD).any(axis=1)

# ===============================
# CLUSTERING
# ===============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = DBSCAN(eps=0.8, min_samples=2)
clusters = model.fit_predict(X_scaled)

# ===============================
# RULE-BASED TAGGING
# ===============================
results = {}

for i, file in enumerate(files):

    anomalies = []

    if z_outlier[i]:
        anomalies.append("STATISTICAL_OUTLIER")

    if raw_data[i].get("degenerate_count", 0) > 0:
        anomalies.append("DEGENERATE_GEOMETRY_PRESENT")

    if raw_data[i].get("confidence") == "LOW":
        anomalies.append("LOW_CONFIDENCE_INPUT")

    if clusters[i] == -1:
        anomalies.append("CLUSTER_NOISE_POINT")

    results[file] = {
        "cluster_id": int(clusters[i]),
        "z_score_flag": bool(z_outlier[i]),
        "anomaly_tags": anomalies,
        "risk_level": (
            "HIGH" if "LOW_CONFIDENCE_INPUT" in anomalies
            else "MEDIUM" if anomalies
            else "LOW"
        )
    }

# ===============================
# SAVE
# ===============================
with open(os.path.join(stl_json_folder, output_file), "w") as f:
    json.dump(results, f, indent=2)

print("STL AI analysis complete.")