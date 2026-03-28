import os
import json
import meshio
import numpy as np

# =====================================================
# Configuration
# =====================================================
input_file = "mesh_files/circle_2d.msh"   # change as needed
output_dir = "output"
output_filename = "circle_2d.json"

COMPLEXITY_THRESHOLD = 100000
LOW_DENSITY_THRESHOLD = 0.5   # elements per node ratio
HIGH_DENSITY_THRESHOLD = 10

os.makedirs(output_dir, exist_ok=True)

# =====================================================
# Initialize Output Structure
# =====================================================
output = {
    "file": input_file,
    "read_success": False,

    "mesh_metrics": {
        "node_count": None,
        "element_count": None,
        "element_types": [],
        "elements_per_node_ratio": None,
        "file_size_bytes": None
    },

    # Solver-style metrics (Not available in .msh)
    "solver_metrics": {
        "warning_count": 0,
        "error_count": 0,
        "iteration_lines": 0,
        "runtime_detected": False,
        "residual_trend": "NOT_AVAILABLE"
    },

    "flags": [],
    "confidence": "LOW"
}

# =====================================================
# File Existence Check
# =====================================================
if not os.path.exists(input_file):
    output["flags"].append("MESH_READ_ERROR")
else:
    output["mesh_metrics"]["file_size_bytes"] = os.path.getsize(input_file)

    try:
        mesh_data = meshio.read(input_file)

        nodes = int(mesh_data.points.shape[0])
        elements = int(sum(len(c.data) for c in mesh_data.cells))
        element_types = list(set(c.type for c in mesh_data.cells))

        output["read_success"] = True
        output["mesh_metrics"]["node_count"] = nodes
        output["mesh_metrics"]["element_count"] = elements
        output["mesh_metrics"]["element_types"] = element_types

        # =================================================
        # Density Calculation
        # =================================================
        if nodes > 0:
            ratio = elements / nodes
            output["mesh_metrics"]["elements_per_node_ratio"] = round(ratio, 4)

            if ratio < LOW_DENSITY_THRESHOLD:
                output["flags"].append("POOR_MESH_DENSITY")

            if ratio > HIGH_DENSITY_THRESHOLD:
                output["flags"].append("GEOMETRY_TOO_COMPLEX")

        # =================================================
        # Invalid Element Check
        # =================================================
        valid_element_types = {
            "line", "triangle", "quad",
            "tetra", "hexahedron", "wedge", "pyramid"
        }

        for etype in element_types:
            if etype not in valid_element_types:
                output["flags"].append("INVALID_ELEMENTS")
                break

        # =================================================
        # Complexity Check
        # =================================================
        if elements > COMPLEXITY_THRESHOLD:
            output["flags"].append("GEOMETRY_TOO_COMPLEX")

        # =================================================
        # Confidence Logic
        # =================================================
        if not output["flags"]:
            output["confidence"] = "HIGH"
        else:
            # severe flags
            if "MESH_READ_ERROR" in output["flags"]:
                output["confidence"] = "LOW"
            else:
                output["confidence"] = "MEDIUM"

    except Exception as e:
        output["flags"].append("MESH_READ_ERROR")
        output["error"] = str(e)
        output["confidence"] = "LOW"

# =====================================================
# Save JSON
# =====================================================
output_path = os.path.join(output_dir, output_filename)

with open(output_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"Mesh validation completed: {output_path}")