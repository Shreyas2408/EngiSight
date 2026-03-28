import os
import json
import numpy as np
from stl import mesh

# ---------------------------
# Configuration
# ---------------------------
input_file = "mobile_controller_stlp/Y-PAD_lower iP15pro.stl"
output_dir = "output"
output_filename = "mobile_controller_stlp.json"

# Complexity threshold (adjust as needed)
COMPLEXITY_TRIANGLE_THRESHOLD = 100000

os.makedirs(output_dir, exist_ok=True)

# ---------------------------
# Initialize Output Structure
# ---------------------------
cad_output = {
    "file": "mobile_controller_stlp",
    "read_success": False,
    "triangle_count": None,
    "degenerate_count": None,
    "bounding_box": None,
    "file_size_bytes": None,
    "flags": [],
    "confidence": "LOW"
}

# ---------------------------
# File existence check
# ---------------------------
if not os.path.exists(input_file):
    cad_output["flags"].append("CAD_READ_ERROR")

    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, "w") as f:
        json.dump(cad_output, f, indent=2)

    raise FileNotFoundError(f"File not found: {input_file}")

# File size
cad_output["file_size_bytes"] = os.path.getsize(input_file)

# ---------------------------
# Try reading STL
# ---------------------------
try:
    your_mesh = mesh.Mesh.from_file(input_file)
    cad_output["read_success"] = True

    # Triangle count
    triangle_count = int(len(your_mesh.vectors))
    cad_output["triangle_count"] = triangle_count

    # ---------------------------
    # Bounding Box
    # ---------------------------
    min_bounds = your_mesh.points.min(axis=0)
    max_bounds = your_mesh.points.max(axis=0)

    cad_output["bounding_box"] = {
        "min": min_bounds.tolist(),
        "max": max_bounds.tolist()
    }

    # ---------------------------
    # Degenerate Triangle Check
    # ---------------------------
    def triangle_area(v0, v1, v2):
        return np.linalg.norm(np.cross(v1 - v0, v2 - v0)) * 0.5

    areas = [triangle_area(*tri) for tri in your_mesh.vectors]
    degenerate_triangles = int(sum(a < 1e-6 for a in areas))
    cad_output["degenerate_count"] = degenerate_triangles

    # ---------------------------
    # Flags Logic
    # ---------------------------
    if degenerate_triangles > 0:
        cad_output["flags"].append("DEGENERATE_GEOMETRY")

    if triangle_count > COMPLEXITY_TRIANGLE_THRESHOLD:
        cad_output["flags"].append("GEOMETRY_TOO_COMPLEX")

    # ---------------------------
    # Confidence Logic
    # ---------------------------
    if cad_output["read_success"]:
        if len(cad_output["flags"]) == 0:
            cad_output["confidence"] = "HIGH"
        else:
            cad_output["confidence"] = "MEDIUM"

except Exception as e:
    cad_output["flags"].append("CAD_READ_ERROR")
    cad_output["error"] = str(e)
    cad_output["confidence"] = "LOW"

# ---------------------------
# Save Output JSON
# ---------------------------
output_path = os.path.join(output_dir, output_filename)

with open(output_path, "w") as f:
    json.dump(cad_output, f, indent=2)

print(f"Validation JSON generated at: {output_path}")