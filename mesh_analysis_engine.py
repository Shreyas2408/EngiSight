import os
import sys
import json
import warnings
import numpy as np
import meshio
import pyvista as pv

warnings.filterwarnings("ignore")


def load_mesh(filepath):
    try:
        mesh = meshio.read(filepath)
        return mesh, None
    except Exception as e:
        return None, str(e)


def convert_to_pyvista(meshio_mesh):
    try:
        cells_combined = []
        for cell_block in meshio_mesh.cells:
            ctype = cell_block.type
            data = cell_block.data
            vtk_type_map = {
                "vertex": 1, "line": 3, "triangle": 5, "quad": 9,
                "tetra": 10, "hexahedron": 12, "wedge": 13, "pyramid": 14,
                "tetra10": 24, "hexahedron20": 25, "hexahedron27": 29,
                "wedge15": 26, "pyramid13": 27, "triangle6": 22, "quad8": 23,
                "quad9": 28, "line3": 21,
            }
            vtk_id = vtk_type_map.get(ctype)
            if vtk_id is None:
                continue
            n_pts = data.shape[1]
            for cell in data:
                cells_combined.append(n_pts)
                cells_combined.extend(cell.tolist())

        if not cells_combined:
            return None

        cell_types_list = []
        for cell_block in meshio_mesh.cells:
            ctype = cell_block.type
            vtk_type_map = {
                "vertex": 1, "line": 3, "triangle": 5, "quad": 9,
                "tetra": 10, "hexahedron": 12, "wedge": 13, "pyramid": 14,
                "tetra10": 24, "hexahedron20": 25, "hexahedron27": 29,
                "wedge15": 26, "pyramid13": 27, "triangle6": 22, "quad8": 23,
                "quad9": 28, "line3": 21,
            }
            vtk_id = vtk_type_map.get(ctype)
            if vtk_id is None:
                continue
            cell_types_list.extend([vtk_id] * cell_block.data.shape[0])

        points = meshio_mesh.points
        if points.shape[1] == 2:
            points = np.hstack([points, np.zeros((points.shape[0], 1))])

        cells_array = np.array(cells_combined)
        cell_types_array = np.array(cell_types_list, dtype=np.uint8)

        grid = pv.UnstructuredGrid(cells_array, cell_types_array, points.astype(np.float64))
        return grid
    except Exception:
        return None


def extract_basic_metrics(pv_mesh, meshio_mesh):
    data = {}
    data["node_count"] = pv_mesh.n_points
    data["element_count"] = pv_mesh.n_cells

    etypes = set()
    for cb in meshio_mesh.cells:
        etypes.add(cb.type)
    data["element_types"] = sorted(list(etypes))

    bounds = pv_mesh.bounds
    data["bounding_box"] = {
        "x_min": bounds[0], "x_max": bounds[1],
        "y_min": bounds[2], "y_max": bounds[3],
        "z_min": bounds[4], "z_max": bounds[5],
    }

    try:
        vol = pv_mesh.volume
        data["mesh_volume"] = float(vol) if vol and vol > 0 else None
    except Exception:
        data["mesh_volume"] = None

    try:
        surf = pv_mesh.extract_surface()
        data["surface_area"] = float(surf.area) if surf.area > 0 else None
    except Exception:
        data["surface_area"] = None

    return data


def _compute_quality(pv_mesh, metric_name):
    try:
        qual = pv_mesh.compute_cell_quality(quality_measure=metric_name)
        arr = qual.cell_data["CellQuality"]
        return arr
    except Exception:
        return None


def extract_quality_metrics(pv_mesh):
    data = {}
    quality_measures = ["skewness", "aspect_ratio", "jacobian", "scaled_jacobian"]

    all_quality_values = []

    for qm in quality_measures:
        arr = _compute_quality(pv_mesh, qm)
        if arr is not None and len(arr) > 0:
            finite = arr[np.isfinite(arr)]
            if len(finite) > 0:
                data[qm] = {
                    "min": float(np.min(finite)),
                    "max": float(np.max(finite)),
                    "mean": float(np.mean(finite)),
                    "std": float(np.std(finite)),
                }
                all_quality_values.extend(finite.tolist())
            else:
                data[qm] = None
        else:
            data[qm] = None

    if all_quality_values:
        q = np.array(all_quality_values)
        data["min_quality"] = float(np.min(q))
        data["max_quality"] = float(np.max(q))
        data["mean_quality"] = float(np.mean(q))
        data["std_quality"] = float(np.std(q))
    else:
        data["min_quality"] = None
        data["max_quality"] = None
        data["mean_quality"] = None
        data["std_quality"] = None

    return data


def extract_size_metrics(pv_mesh):
    data = {}
    try:
        sizes = pv_mesh.compute_cell_sizes(length=False, area=True, volume=True)
        vol_arr = sizes.cell_data.get("Volume")
        area_arr = sizes.cell_data.get("Area")

        if vol_arr is not None and np.any(vol_arr > 0):
            s = vol_arr[vol_arr > 0]
        elif area_arr is not None and np.any(area_arr > 0):
            s = area_arr[area_arr > 0]
        else:
            s = None

        if s is not None and len(s) > 0:
            data["min_element_size"] = float(np.min(s))
            data["max_element_size"] = float(np.max(s))
            data["average_element_size"] = float(np.mean(s))
        else:
            data["min_element_size"] = None
            data["max_element_size"] = None
            data["average_element_size"] = None
    except Exception:
        data["min_element_size"] = None
        data["max_element_size"] = None
        data["average_element_size"] = None

    return data


def extract_edge_metrics(pv_mesh):
    data = {}
    try:
        edges = pv_mesh.extract_all_edges()
        if edges.n_cells == 0:
            data["average_edge_length"] = None
            data["edge_length_std"] = None
            return data

        points = edges.points
        lines = edges.lines
        lengths = []
        i = 0
        while i < len(lines):
            n = lines[i]
            if n == 2:
                p0 = points[lines[i + 1]]
                p1 = points[lines[i + 2]]
                lengths.append(np.linalg.norm(p1 - p0))
            i += n + 1

        if lengths:
            l_arr = np.array(lengths)
            data["average_edge_length"] = float(np.mean(l_arr))
            data["edge_length_std"] = float(np.std(l_arr))
        else:
            data["average_edge_length"] = None
            data["edge_length_std"] = None
    except Exception:
        data["average_edge_length"] = None
        data["edge_length_std"] = None

    return data


def extract_connectivity_metrics(pv_mesh):
    data = {}
    try:
        conn = pv_mesh.connectivity(extraction_mode="all")
        region_ids = conn.cell_data.get("RegionId")
        if region_ids is not None and len(region_ids) > 0:
            n_regions = int(np.max(region_ids)) + 1
        else:
            n_regions = 1
        data["connected_components"] = n_regions
        data["region_count"] = n_regions
    except Exception:
        data["connected_components"] = None
        data["region_count"] = None
    return data


def extract_integrity_metrics(pv_mesh, meshio_mesh):
    data = {}

    # Duplicate nodes
    try:
        pts = pv_mesh.points
        rounded = np.round(pts, decimals=8)
        unique, counts = np.unique(rounded, axis=0, return_counts=True)
        dup_count = int(np.sum(counts[counts > 1]) - np.sum(counts > 1))
        data["duplicate_nodes"] = dup_count
    except Exception:
        data["duplicate_nodes"] = None

    # Duplicate elements
    try:
        all_cells = []
        for cb in meshio_mesh.cells:
            for c in cb.data:
                all_cells.append(tuple(sorted(c.tolist())))
        unique_cells = set()
        dup_elem = 0
        for c in all_cells:
            if c in unique_cells:
                dup_elem += 1
            else:
                unique_cells.add(c)
        data["duplicate_elements"] = dup_elem
    except Exception:
        data["duplicate_elements"] = None

    # Isolated nodes
    try:
        used_nodes = set()
        for cb in meshio_mesh.cells:
            used_nodes.update(cb.data.flatten().tolist())
        total_nodes = meshio_mesh.points.shape[0]
        data["isolated_nodes"] = total_nodes - len(used_nodes)
    except Exception:
        data["isolated_nodes"] = None

    # Non-manifold edges and boundary edges
    try:
        surf = pv_mesh.extract_surface()
        edges = surf.extract_feature_edges(
            boundary_edges=True,
            non_manifold_edges=True,
            feature_edges=False,
            manifold_edges=False,
        )

        non_manifold = surf.extract_feature_edges(
            boundary_edges=False,
            non_manifold_edges=True,
            feature_edges=False,
            manifold_edges=False,
        )
        data["non_manifold_edges"] = non_manifold.n_cells

        boundary = surf.extract_feature_edges(
            boundary_edges=True,
            non_manifold_edges=False,
            feature_edges=False,
            manifold_edges=False,
        )
        data["boundary_edges"] = boundary.n_cells
    except Exception:
        data["non_manifold_edges"] = None
        data["boundary_edges"] = None

    return data


def validate_mesh(data):
    flags = []

    if data.get("_read_failure"):
        flags.append("read_failure")
        return flags

    # Degenerate elements
    min_size = data.get("min_element_size")
    if min_size is not None and min_size < 1e-12:
        flags.append("degenerate_elements")

    sj = data.get("scaled_jacobian")
    if sj is not None and isinstance(sj, dict):
        if sj.get("min") is not None and sj["min"] < 1e-6:
            flags.append("degenerate_elements") if "degenerate_elements" not in flags else None

    # Duplicates
    if data.get("duplicate_nodes") and data["duplicate_nodes"] > 0:
        flags.append("duplicate_nodes")
    if data.get("duplicate_elements") and data["duplicate_elements"] > 0:
        flags.append("duplicate_elements")

    # Isolated nodes
    if data.get("isolated_nodes") and data["isolated_nodes"] > 0:
        flags.append("isolated_nodes")

    # Non-manifold
    if data.get("non_manifold_edges") and data["non_manifold_edges"] > 0:
        flags.append("non_manifold_edges")

    # Low quality
    sj = data.get("scaled_jacobian")
    if sj is not None and isinstance(sj, dict) and sj.get("mean") is not None:
        if sj["mean"] < 0.3:
            flags.append("low_quality_mesh")

    # Missing metrics
    critical_keys = [
        "node_count", "element_count", "mesh_volume", "surface_area",
        "min_element_size", "average_edge_length", "connected_components",
    ]
    missing = sum(1 for k in critical_keys if data.get(k) is None)
    if missing > 2:
        flags.append("missing_metrics")

    return flags


def calculate_confidence_score(flags):
    score = 1.0
    penalties = {
        "degenerate_elements": 0.2,
        "duplicate_nodes": 0.1,
        "duplicate_elements": 0.1,
        "non_manifold_edges": 0.2,
        "low_quality_mesh": 0.2,
        "missing_metrics": 0.1,
        "read_failure": 1.0,
    }
    for flag in flags:
        score -= penalties.get(flag, 0.0)
    return max(0.0, min(1.0, score))


def _safe_serialize(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(i) for i in obj]
    return obj


def process_mesh_file(filepath):
    result = {}

    # File metadata
    result["file_name"] = os.path.basename(filepath)
    result["file_type"] = os.path.splitext(filepath)[1].lower()
    try:
        result["file_size"] = os.path.getsize(filepath)
    except OSError:
        result["file_size"] = None

    # Load
    meshio_mesh, error = load_mesh(filepath)
    if meshio_mesh is None:
        result["_read_failure"] = True
        result["error"] = error
        flags = validate_mesh(result)
        result["flags"] = flags
        result["confidence_score"] = calculate_confidence_score(flags)
        result.pop("_read_failure", None)
        return _safe_serialize(result)

    # Convert to PyVista
    pv_mesh = convert_to_pyvista(meshio_mesh)
    if pv_mesh is None or pv_mesh.n_cells == 0:
        result["_read_failure"] = True
        result["error"] = "Failed to convert mesh to PyVista UnstructuredGrid or mesh has no cells."
        flags = validate_mesh(result)
        result["flags"] = flags
        result["confidence_score"] = calculate_confidence_score(flags)
        result.pop("_read_failure", None)
        return _safe_serialize(result)

    # Extract metrics
    basic = extract_basic_metrics(pv_mesh, meshio_mesh)
    result.update(basic)

    quality = extract_quality_metrics(pv_mesh)
    result.update(quality)

    size = extract_size_metrics(pv_mesh)
    result.update(size)

    edge = extract_edge_metrics(pv_mesh)
    result.update(edge)

    connectivity = extract_connectivity_metrics(pv_mesh)
    result.update(connectivity)

    integrity = extract_integrity_metrics(pv_mesh, meshio_mesh)
    result.update(integrity)

    # Mesh density
    vol = result.get("mesh_volume")
    elem_count = result.get("element_count", 0)
    if vol and vol > 0 and elem_count > 0:
        result["mesh_density"] = float(elem_count) / float(vol)
    else:
        result["mesh_density"] = None

    # Validate
    flags = validate_mesh(result)
    result["flags"] = flags
    result["confidence_score"] = calculate_confidence_score(flags)

    result.pop("_read_failure", None)
    return _safe_serialize(result)


def main():
    if len(sys.argv) < 2:
        print("Usage: python mesh_analysis_engine.py <mesh_file> [mesh_file2 ...]")
        sys.exit(1)

    filepaths = sys.argv[1:]
    results = []

    for fp in filepaths:
        fp = os.path.abspath(fp)
        if not os.path.isfile(fp):
            results.append({
                "file_name": os.path.basename(fp),
                "error": f"File not found: {fp}",
                "flags": ["read_failure"],
                "confidence_score": 0.0,
            })
            continue
        result = process_mesh_file(fp)
        results.append(result)

    if len(results) == 1:
        print(json.dumps(results[0], indent=2))
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
