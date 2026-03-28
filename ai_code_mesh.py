import os
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# ===============================
# CONFIG
# ===============================
mesh_json_folder = "output"   # put your mesh validation json files here
output_file = "mesh_ai_insight.json"
Z_THRESHOLD = 2.5

# ===============================
# LOAD FILES
# ===============================
files = [
    f for f in os.listdir(mesh_json_folder)
    if f.endswith(".json") and "mesh_validation" in f
]

if len(files) < 2:
    raise ValueError("Add at least 2 mesh validation JSON files.")

data = []
raw_data = []

for file in files:
    path = os.path.join(mesh_json_folder, file)

    with open(path) as f:
        j = json.load(f)

    metrics = j.get("mesh_metrics", {})

    node = float(metrics.get("node_count") or 0)
    elem = float(metrics.get("element_count") or 0)
    density = float(metrics.get("elements_per_node_ratio") or 0)

    data.append([node, elem, density])
    raw_data.append(j)

X = np.array(data)

# ===============================
# Z-SCORE OUTLIER
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

    if raw_data[i].get("confidence") == "LOW":
        anomalies.append("LOW_CONFIDENCE_INPUT")

    if raw_data[i].get("flags"):
        anomalies.append("PREVIOUS_VALIDATION_FLAGS")

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
with open(os.path.join(mesh_json_folder, output_file), "w") as f:
    json.dump(results, f, indent=2)

print("Mesh AI analysis complete.")