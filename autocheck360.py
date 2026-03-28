import os
import sys
import json
import re
import math
import warnings
from pathlib import Path

import numpy as np
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

warnings.filterwarnings("ignore")

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

try:
    from OCP.STEPControl import STEPControl_Reader
    from OCP.IGESControl import IGESControl_Reader
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX, TopAbs_SOLID, TopAbs_SHELL
    from OCP.BRep import BRep_Tool
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp_GProps import GProp_GProps
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopExp import TopExp
    from OCP.TopoDS import topods
    HAS_OCC = True
    OCC_BACKEND = "OCP"
except ImportError:
    try:
        from OCC.Core.STEPControl import STEPControl_Reader
        from OCC.Core.IGESControl import IGESControl_Reader
        from OCC.Core.IFSelect import IFSelect_RetDone
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX, TopAbs_SOLID, TopAbs_SHELL
        from OCC.Core.BRep import BRep_Tool
        from OCC.Core.BRepGProp import brepgprop_VolumeProperties, brepgprop_SurfaceProperties
        from OCC.Core.GProp import GProp_GProps
        from OCC.Core.Bnd import Bnd_Box
        from OCC.Core.BRepBndLib import brepbndlib_Add
        from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
        from OCC.Core.TopoDS import topods
        HAS_OCC = True
        OCC_BACKEND = "OCC.Core"
    except ImportError:
        HAS_OCC = False
        OCC_BACKEND = None

try:
    import meshio
    HAS_MESHIO = True
except ImportError:
    HAS_MESHIO = False

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False

try:
    import vtk
    HAS_VTK = True
except ImportError:
    HAS_VTK = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CAD_EXTENSIONS = {".step", ".stp", ".iges", ".igs", ".stl"}
MESH_EXTENSIONS = {".msh", ".vtk", ".inp", ".cdb", ".fem", ".ccm"}
SOLVER_EXTENSIONS = {".log", ".out", ".dat", ".txt", ".csv"}

CAD_METRICS = [
    "file_name", "file_type", "file_size",
    "part_or_assembly", "multi_body", "solid_count",
    "bounding_box", "dimensions", "volume", "surface_area", "centroid",
    "face_count", "edge_count", "vertex_count",
    "degenerate_faces", "non_manifold_edges",
]

MESH_METRICS = [
    "file_name", "file_type", "file_size",
    "node_count", "element_count", "element_types", "bounding_box", "mesh_volume",
    "surface_area", "skewness", "aspect_ratio", "jacobian", "scaled_jacobian",
    "min_quality", "max_quality", "mean_quality", "std_quality", "min_element_size",
    "max_element_size", "average_element_size",
    "duplicate_nodes", "duplicate_elements", "isolated_nodes",
    "non_manifold_edges", "boundary_edges",
    "connected_components", "region_count", "mesh_quality_grade",
]

SOLVER_METRICS = [
    "warning_count", "error_count", "warning_types", "error_types", "residual_values",
    "initial_residual", "final_residual", "residual_decay_rate", "iteration_count",
    "convergence_status", "divergence_detection", "convergence_rate",
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _safe_float(v):
    """Convert to float safely, returning None for NaN/Inf."""
    if v is None:
        return None
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def _safe_serialize(obj):
    """Recursively convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(i) for i in obj]
    return obj


# ---------------------------------------------------------------------------
# Directory Scan & File Detection
# ---------------------------------------------------------------------------

def scan_directory(directory=None):
    """Scan the working directory and return list of supported file paths."""
    if directory is None:
        directory = Path(__file__).resolve().parent
    else:
        directory = Path(directory).resolve()

    supported = CAD_EXTENSIONS | MESH_EXTENSIONS | SOLVER_EXTENSIONS
    files = []
    for item in directory.rglob("*"):
        if item.is_file() and item.suffix.lower() in supported:
            rel = item.relative_to(directory)
            parts = rel.parts
            if any(p.startswith(".") or p == "__pycache__" or p == ".venv" for p in parts):
                continue
            files.append(str(item))
    return sorted(files)


def detect_file_type(filepath):
    """Classify a file as 'cad', 'mesh', 'solver', or None."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext in (".stp",):
        ext = ".step"
    if ext in (".igs",):
        ext = ".iges"

    if ext in {".step", ".iges", ".stl"}:
        return "cad"
    if ext in {".msh", ".vtk", ".fem", ".ccm"}:
        return "mesh"
    if ext in {".log", ".dat", ".csv"}:
        return "solver"
    if ext == ".inp":
        return "mesh"
    if ext == ".cdb":
        return "mesh"
    if ext in {".out", ".txt"}:
        return "solver"
    return None


# ===========================================================================
# CAD ENGINE
# ===========================================================================

def _load_step(filepath):
    """Load a STEP file via OpenCascade, return shape."""
    reader = STEPControl_Reader()
    status = reader.ReadFile(filepath)
    if status != IFSelect_RetDone:
        raise RuntimeError(f"STEP read failed: {filepath}")
    reader.TransferRoots()
    return reader.OneShape()


def _load_iges(filepath):
    """Load an IGES file via OpenCascade, return shape."""
    reader = IGESControl_Reader()
    status = reader.ReadFile(filepath)
    if status != IFSelect_RetDone:
        raise RuntimeError(f"IGES read failed: {filepath}")
    reader.TransferRoots()
    return reader.OneShape()


def _count_topology(shape, topo_type):
    """Count topological entities of a given type in a shape."""
    count = 0
    explorer = TopExp_Explorer(shape, topo_type)
    while explorer.More():
        count += 1
        explorer.Next()
    return count


def _collect_topology(shape, topo_type):
    """Collect all topological entities of a given type."""
    items = []
    explorer = TopExp_Explorer(shape, topo_type)
    while explorer.More():
        items.append(explorer.Current())
        explorer.Next()
    return items


def _compute_face_area(face):
    """Compute the surface area of a single face."""
    props = GProp_GProps()
    if OCC_BACKEND == "OCC.Core":
        brepgprop_SurfaceProperties(face, props)
    else:
        BRepGProp.SurfaceProperties_s(face, props)
    return props.Mass()


def _count_degenerate_faces(shape):
    """Count faces where computed area < 1e-6."""
    count = 0
    faces = _collect_topology(shape, TopAbs_FACE)
    for face in faces:
        try:
            area = _compute_face_area(topods.Face(face) if OCC_BACKEND == "OCC.Core" else topods.Face_s(face))
            if area < 1e-6:
                count += 1
        except Exception:
            count += 1
    return count


def _count_non_manifold_edges(shape):
    """Count edges shared by more than 2 faces."""
    from collections import defaultdict
    edge_face_count = defaultdict(int)
    faces = _collect_topology(shape, TopAbs_FACE)
    for face in faces:
        face_edges = _collect_topology(face, TopAbs_EDGE)
        for edge in face_edges:
            try:
                if OCC_BACKEND == "OCC.Core":
                    edge_hash = edge.__hash__()
                else:
                    edge_hash = edge.HashCode(2147483647)
            except Exception:
                edge_hash = id(edge)
            edge_face_count[edge_hash] += 1
    return sum(1 for c in edge_face_count.values() if c > 2)


def _extract_brep_features(shape, filepath):
    """Extract CAD metrics from a BRep shape (STEP/IGES)."""
    data = {}
    data["file_name"] = Path(filepath).name
    data["file_type"] = Path(filepath).suffix
    data["file_size"] = os.path.getsize(filepath)

    try:
        solid_count = _count_topology(shape, TopAbs_SOLID)
        data["solid_count"] = solid_count
        data["part_or_assembly"] = "part" if solid_count == 1 else "assembly"
        data["multi_body"] = solid_count > 1
    except Exception:
        data["solid_count"] = None
        data["part_or_assembly"] = None
        data["multi_body"] = None

    try:
        bbox = Bnd_Box()
        if OCC_BACKEND == "OCC.Core":
            brepbndlib_Add(shape, bbox)
        else:
            BRepBndLib.Add_s(shape, bbox)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
        data["bounding_box"] = {
            "xmin": _safe_float(xmin), "ymin": _safe_float(ymin), "zmin": _safe_float(zmin),
            "xmax": _safe_float(xmax), "ymax": _safe_float(ymax), "zmax": _safe_float(zmax),
        }
        data["dimensions"] = {
            "L": _safe_float(xmax - xmin),
            "W": _safe_float(ymax - ymin),
            "H": _safe_float(zmax - zmin),
        }
    except Exception:
        data["bounding_box"] = None
        data["dimensions"] = None

    try:
        props = GProp_GProps()
        if OCC_BACKEND == "OCC.Core":
            brepgprop_VolumeProperties(shape, props)
        else:
            BRepGProp.VolumeProperties_s(shape, props)
        volume = props.Mass()
        data["volume"] = _safe_float(abs(volume))
        c = props.CentreOfMass()
        data["centroid"] = {"x": _safe_float(c.X()), "y": _safe_float(c.Y()), "z": _safe_float(c.Z())}
    except Exception:
        data["volume"] = None
        data["centroid"] = None

    try:
        props = GProp_GProps()
        if OCC_BACKEND == "OCC.Core":
            brepgprop_SurfaceProperties(shape, props)
        else:
            BRepGProp.SurfaceProperties_s(shape, props)
        data["surface_area"] = _safe_float(props.Mass())
    except Exception:
        data["surface_area"] = None

    try:
        data["face_count"] = _count_topology(shape, TopAbs_FACE)
    except Exception:
        data["face_count"] = None

    try:
        data["edge_count"] = _count_topology(shape, TopAbs_EDGE)
    except Exception:
        data["edge_count"] = None

    try:
        data["vertex_count"] = _count_topology(shape, TopAbs_VERTEX)
    except Exception:
        data["vertex_count"] = None

    try:
        data["degenerate_faces"] = _count_degenerate_faces(shape)
    except Exception:
        data["degenerate_faces"] = None

    try:
        data["non_manifold_edges"] = _count_non_manifold_edges(shape)
    except Exception:
        data["non_manifold_edges"] = None

    return data


def _extract_stl_features(filepath):
    """Extract CAD metrics from an STL file using trimesh."""
    mesh = trimesh.load(filepath, force="mesh")
    data = {}
    data["file_name"] = Path(filepath).name
    data["file_type"] = Path(filepath).suffix
    data["file_size"] = os.path.getsize(filepath)

    data["solid_count"] = 1
    data["part_or_assembly"] = "part"
    data["multi_body"] = False

    try:
        bb = mesh.bounds
        xmin, ymin, zmin = float(bb[0][0]), float(bb[0][1]), float(bb[0][2])
        xmax, ymax, zmax = float(bb[1][0]), float(bb[1][1]), float(bb[1][2])
        data["bounding_box"] = {
            "xmin": xmin, "ymin": ymin, "zmin": zmin,
            "xmax": xmax, "ymax": ymax, "zmax": zmax,
        }
        data["dimensions"] = {
            "L": _safe_float(xmax - xmin),
            "W": _safe_float(ymax - ymin),
            "H": _safe_float(zmax - zmin),
        }
    except Exception:
        data["bounding_box"] = None
        data["dimensions"] = None

    try:
        data["volume"] = _safe_float(float(mesh.volume))
    except Exception:
        data["volume"] = None

    try:
        data["surface_area"] = _safe_float(float(mesh.area))
    except Exception:
        data["surface_area"] = None

    try:
        c = mesh.centroid
        data["centroid"] = {"x": _safe_float(float(c[0])), "y": _safe_float(float(c[1])), "z": _safe_float(float(c[2]))}
    except Exception:
        data["centroid"] = None

    try:
        data["face_count"] = int(len(mesh.faces))
    except Exception:
        data["face_count"] = None

    try:
        data["edge_count"] = int(len(mesh.edges_unique))
    except Exception:
        data["edge_count"] = None

    try:
        data["vertex_count"] = int(len(mesh.vertices))
    except Exception:
        data["vertex_count"] = None

    try:
        areas = mesh.area_faces
        data["degenerate_faces"] = int(np.sum(areas < 1e-6))
    except Exception:
        data["degenerate_faces"] = None

    try:
        edges_sorted = np.sort(mesh.edges, axis=1)
        unique_edges, edge_counts = np.unique(edges_sorted, axis=0, return_counts=True)
        data["non_manifold_edges"] = int(np.sum(edge_counts > 2))
    except Exception:
        data["non_manifold_edges"] = None

    return data


def extract_cad_features(filepath, file_type_detail):
    """Route to the correct CAD extraction backend."""
    if file_type_detail == "stl":
        if not HAS_TRIMESH:
            raise ImportError("trimesh is not installed")
        return _extract_stl_features(filepath)
    elif file_type_detail in ("step", "iges"):
        if not HAS_OCC:
            raise ImportError("pythonocc-core is not installed")
        if file_type_detail == "step":
            shape = _load_step(filepath)
        else:
            shape = _load_iges(filepath)
        return _extract_brep_features(shape, filepath)
    else:
        raise ValueError(f"Unsupported CAD type: {file_type_detail}")


def process_cad_file(filepath):
    """Full CAD pipeline: load → extract → validate → score."""
    ext = os.path.splitext(filepath)[1].lower()
    type_map = {".stl": "stl", ".step": "step", ".stp": "step", ".iges": "iges", ".igs": "iges"}
    file_type_detail = type_map.get(ext, "unknown")

    try:
        data = extract_cad_features(filepath, file_type_detail)
    except Exception as e:
        data = {
            "file_name": Path(filepath).name,
            "file_type": Path(filepath).suffix,
            "file_size": os.path.getsize(filepath) if os.path.isfile(filepath) else None,
            "error": str(e),
        }
        for m in CAD_METRICS:
            data.setdefault(m, None)
        data["flags"] = ["read_failure"]
        data["confidence_score"] = calculate_confidence_score(data["flags"])
        return _safe_serialize(data)

    flags = validate_results(data, "cad")
    data["flags"] = flags
    data["confidence_score"] = calculate_confidence_score(flags)
    return _safe_serialize(data)


# ===========================================================================
# MESH ENGINE
# ===========================================================================

def _summarize(arr):
    """Return min/max/mean/std summary dict from a numpy array."""
    arr = np.array(arr)
    return {
        "min":  float(arr.min()),
        "max":  float(arr.max()),
        "mean": float(arr.mean()),
        "std":  float(arr.std())
    }


def _compute_edge_lengths(mesh):
    """Compute all pairwise edge lengths from mesh element connectivity."""
    lengths = []
    for cell_block in mesh.cells:
        for element in cell_block.data:
            pts = mesh.points[element]
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    lengths.append(np.linalg.norm(pts[i] - pts[j]))
    return np.array(lengths)


def _count_edge_type(grid, boundary=False, non_manifold=False):
    """Count edges of a specific type using a fresh vtkFeatureEdges instance."""
    fe = vtk.vtkFeatureEdges()
    fe.SetInputData(grid)
    fe.BoundaryEdgesOn() if boundary else fe.BoundaryEdgesOff()
    fe.NonManifoldEdgesOn() if non_manifold else fe.NonManifoldEdgesOff()
    fe.FeatureEdgesOff()
    fe.ManifoldEdgesOff()
    fe.Update()
    return fe.GetOutput().GetNumberOfCells()


def _grade_mesh(sj_mean, skew_mean, ar_mean):
    """Return mesh quality grade based on industry thresholds."""
    if sj_mean >= 0.7 and skew_mean <= 0.3 and ar_mean <= 3.0:
        return "GOOD"
    elif sj_mean >= 0.4 and skew_mean <= 0.6 and ar_mean <= 5.0:
        return "MODERATE"
    else:
        return "POOR"


def extract_mesh_features(file_path):
    """Extract all mesh metrics from a mesh file."""
    flags = []

    try:
        mesh = meshio.read(file_path)
    except Exception:
        return {"flags": ["read_failure"]}

    grid = pv.wrap(mesh)

    # Basic structure
    node_count = len(mesh.points)
    element_count = sum(len(c.data) for c in mesh.cells)
    element_types = [c.type for c in mesh.cells]

    # Geometry
    bounding_box = {
        "min": mesh.points.min(axis=0).tolist(),
        "max": mesh.points.max(axis=0).tolist()
    }

    try:
        mesh_volume = grid.volume
    except Exception:
        mesh_volume = None
        flags.append("missing_metrics")

    try:
        surface_area = grid.area
    except Exception:
        surface_area = None
        flags.append("missing_metrics")

    # Quality — summarized as {min, max, mean, std}
    try:
        skew_arr = grid.compute_cell_quality(quality_measure='skew')["CellQuality"]
        skewness = _summarize(skew_arr)
    except Exception:
        skewness = None
        skew_arr = None
        flags.append("missing_metrics")

    try:
        ar_arr = grid.compute_cell_quality(quality_measure='aspect_ratio')["CellQuality"]
        aspect_ratio = _summarize(ar_arr)
    except Exception:
        aspect_ratio = None
        ar_arr = None
        flags.append("missing_metrics")

    try:
        jac_arr = grid.compute_cell_quality(quality_measure='jacobian')["CellQuality"]
        jacobian = _summarize(jac_arr)
        if np.any(jac_arr < 0):
            flags.append("invalid_elements")
    except Exception:
        jacobian = None
        jac_arr = None
        flags.append("missing_metrics")

    try:
        sj = grid.compute_cell_quality(quality_measure='scaled_jacobian')["CellQuality"]
        scaled_jacobian = _summarize(sj)
        min_quality = float(sj.min())
        max_quality = float(sj.max())
        mean_quality = float(sj.mean())
        std_quality = float(sj.std())
    except Exception:
        scaled_jacobian = None
        sj = None
        min_quality = max_quality = mean_quality = std_quality = None
        flags.append("missing_metrics")

    # Element size — computed from actual edge lengths
    try:
        edge_lengths = _compute_edge_lengths(mesh)
        min_element_size = float(edge_lengths.min())
        max_element_size = float(edge_lengths.max())
        average_element_size = float(edge_lengths.mean())
    except Exception:
        min_element_size = max_element_size = average_element_size = None
        flags.append("missing_metrics")

    # Integrity — duplicate nodes
    try:
        duplicate_nodes = int(len(mesh.points) - len(np.unique(mesh.points, axis=0)))
    except Exception:
        duplicate_nodes = None
        flags.append("missing_metrics")

    # Integrity — isolated nodes
    try:
        all_indices = np.concatenate([c.data.flatten() for c in mesh.cells])
        isolated_nodes = node_count - len(set(all_indices.tolist()))
    except Exception:
        isolated_nodes = None
        flags.append("missing_metrics")

    # Integrity — duplicate elements
    try:
        rows = [tuple(row) for c in mesh.cells for row in c.data.tolist()]
        duplicate_elements = len(rows) - len(set(rows))
    except Exception:
        duplicate_elements = None
        flags.append("missing_metrics")

    # Integrity — non-manifold and boundary edges via VTK (separate instances)
    try:
        boundary_edges = _count_edge_type(grid, boundary=True)
        non_manifold_edges = _count_edge_type(grid, non_manifold=True)
    except Exception:
        non_manifold_edges = boundary_edges = None
        flags.append("missing_metrics")

    # Connectivity
    try:
        conn = grid.connectivity()
        region_ids = conn["RegionId"]
        connected_components = int(np.unique(region_ids).size)
        region_count = connected_components
    except Exception:
        connected_components = region_count = None
        flags.append("missing_metrics")

    # Mesh quality grade
    try:
        sj_mean = scaled_jacobian["mean"] if scaled_jacobian else None
        skew_mean = skewness["mean"] if skewness else None
        ar_mean = aspect_ratio["mean"] if aspect_ratio else None
        if sj_mean is not None and skew_mean is not None and ar_mean is not None:
            mesh_quality_grade = _grade_mesh(sj_mean, skew_mean, ar_mean)
        else:
            mesh_quality_grade = None
    except Exception:
        mesh_quality_grade = None

    # Deduplicate flags
    flags = list(dict.fromkeys(flags))

    return {
        "file_name": Path(file_path).name,
        "file_type": Path(file_path).suffix,
        "file_size": os.path.getsize(file_path),
        "node_count": node_count,
        "element_count": element_count,
        "element_types": element_types,
        "bounding_box": bounding_box,
        "mesh_volume": mesh_volume,
        "surface_area": surface_area,
        "skewness": skewness,
        "aspect_ratio": aspect_ratio,
        "jacobian": jacobian,
        "scaled_jacobian": scaled_jacobian,
        "min_quality": min_quality,
        "max_quality": max_quality,
        "mean_quality": mean_quality,
        "std_quality": std_quality,
        "min_element_size": min_element_size,
        "max_element_size": max_element_size,
        "average_element_size": average_element_size,
        "duplicate_nodes": int(duplicate_nodes) if duplicate_nodes is not None else None,
        "duplicate_elements": duplicate_elements,
        "isolated_nodes": isolated_nodes,
        "non_manifold_edges": non_manifold_edges,
        "boundary_edges": boundary_edges,
        "connected_components": connected_components,
        "region_count": region_count,
        "mesh_quality_grade": mesh_quality_grade,
        "flags": flags,
    }


def process_mesh_file(filepath):
    """Full mesh pipeline: load → extract → validate → score."""
    if not HAS_MESHIO or not HAS_PYVISTA:
        result = {
            "file_name": Path(filepath).name,
            "file_type": Path(filepath).suffix,
            "file_size": os.path.getsize(filepath) if os.path.isfile(filepath) else None,
            "error": "meshio or pyvista not installed",
        }
        for m in MESH_METRICS:
            result.setdefault(m, None)
        result["flags"] = ["read_failure"]
        result["confidence_score"] = calculate_confidence_score(result["flags"])
        return _safe_serialize(result)

    try:
        result = extract_mesh_features(filepath)
    except Exception as e:
        result = {
            "file_name": Path(filepath).name,
            "file_type": Path(filepath).suffix,
            "file_size": os.path.getsize(filepath) if os.path.isfile(filepath) else None,
            "error": f"Feature extraction failed: {e}",
        }
        for m in MESH_METRICS:
            result.setdefault(m, None)
        result["flags"] = ["read_failure"]
        result["confidence_score"] = calculate_confidence_score(result["flags"])
        return _safe_serialize(result)

    if "flags" in result and result["flags"] == ["read_failure"]:
        for m in MESH_METRICS:
            result.setdefault(m, None)
        result["confidence_score"] = calculate_confidence_score(result["flags"])
        return _safe_serialize(result)

    flags = validate_results(result, "mesh")
    existing_flags = result.get("flags", [])
    combined = list(dict.fromkeys(existing_flags + flags))
    result["flags"] = combined
    result["confidence_score"] = calculate_confidence_score(combined)

    print(json.dumps(_safe_serialize(result), indent=2, default=str))
    return _safe_serialize(result)


# ===========================================================================
# SOLVER ENGINE
# ===========================================================================

def extract_solver_features(text):
    """Extract solver metrics from raw log/output text using regex."""
    data = {}

    warning_patterns = [
        re.compile(r'\*\*\*\s*WARNING\b[:\s]*(.*)', re.IGNORECASE),
        re.compile(r'WARNING[:\s]+(.*)', re.IGNORECASE),
        re.compile(r'WARN[:\s]+(.*)', re.IGNORECASE),
    ]
    error_patterns = [
        re.compile(r'\*\*\*\s*ERROR\b[:\s]*(.*)', re.IGNORECASE),
        re.compile(r'ERROR[:\s]+(.*)', re.IGNORECASE),
        re.compile(r'FATAL[:\s]+(.*)', re.IGNORECASE),
    ]

    warnings_found = []
    errors_found = []
    for line in text.splitlines():
        stripped = line.strip()
        for pat in warning_patterns:
            m = pat.search(stripped)
            if m:
                msg = m.group(1).strip() if m.group(1).strip() else "unspecified"
                warnings_found.append(msg)
                break
        for pat in error_patterns:
            m = pat.search(stripped)
            if m:
                msg = m.group(1).strip() if m.group(1).strip() else "unspecified"
                errors_found.append(msg)
                break

    data["warning_count"] = len(warnings_found)
    data["error_count"] = len(errors_found)

    warning_types = {}
    for w in warnings_found:
        category = w[:80] if len(w) > 80 else w
        warning_types[category] = warning_types.get(category, 0) + 1
    data["warning_types"] = warning_types if warning_types else None

    error_types = {}
    for e in errors_found:
        category = e[:80] if len(e) > 80 else e
        error_types[category] = error_types.get(category, 0) + 1
    data["error_types"] = error_types if error_types else None

    residual_patterns = [
        re.compile(r'[Rr]esidual\s*[=:]\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)', re.IGNORECASE),
        re.compile(r'[Rr]es\s*[=:]\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)', re.IGNORECASE),
        re.compile(r'RMS\s*[=:]\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)', re.IGNORECASE),
        re.compile(r'[Nn]orm\s*[=:]\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)', re.IGNORECASE),
    ]

    residual_values = []
    for line in text.splitlines():
        for pat in residual_patterns:
            for m in pat.finditer(line):
                try:
                    val = float(m.group(1))
                    residual_values.append(val)
                except (ValueError, TypeError):
                    pass

    data["residual_values"] = residual_values if residual_values else None

    if residual_values:
        data["initial_residual"] = residual_values[0]
        data["final_residual"] = residual_values[-1]
        if len(residual_values) >= 2 and residual_values[0] != 0:
            data["residual_decay_rate"] = _safe_float(residual_values[-1] / residual_values[0])
        else:
            data["residual_decay_rate"] = None
    else:
        data["initial_residual"] = None
        data["final_residual"] = None
        data["residual_decay_rate"] = None

    iteration_patterns = [
        re.compile(r'[Ii]teration\s*[=:#]\s*(\d+)', re.IGNORECASE),
        re.compile(r'[Ss]tep\s*[=:#]\s*(\d+)', re.IGNORECASE),
        re.compile(r'[Ii]ncrement\s*[=:#]\s*(\d+)', re.IGNORECASE),
        re.compile(r'TIME STEP\s*[=:#]\s*(\d+)', re.IGNORECASE),
    ]
    iteration_counts = []
    for line in text.splitlines():
        for pat in iteration_patterns:
            m = pat.search(line)
            if m:
                try:
                    iteration_counts.append(int(m.group(1)))
                except (ValueError, TypeError):
                    pass

    data["iteration_count"] = max(iteration_counts) if iteration_counts else None

    convergence_patterns = [
        (re.compile(r'CONVERGED', re.IGNORECASE), True),
        (re.compile(r'CONVERGENCE\s+ACHIEVED', re.IGNORECASE), True),
        (re.compile(r'SOLUTION\s+CONVERGED', re.IGNORECASE), True),
        (re.compile(r'NOT\s+CONVERGED', re.IGNORECASE), False),
        (re.compile(r'DIVERGED', re.IGNORECASE), False),
        (re.compile(r'DIVERGENCE', re.IGNORECASE), False),
        (re.compile(r'FAILED\s+TO\s+CONVERGE', re.IGNORECASE), False),
    ]

    convergence_status = None
    for line in text.splitlines():
        for pat, status in convergence_patterns:
            if pat.search(line):
                convergence_status = "converged" if status else "not_converged"

    data["convergence_status"] = convergence_status

    divergence_patterns = [
        re.compile(r'DIVERG', re.IGNORECASE),
        re.compile(r'SOLUTION\s+BLEW\s+UP', re.IGNORECASE),
        re.compile(r'NUMERICAL\s+SINGULARITY', re.IGNORECASE),
        re.compile(r'OVERFLOW', re.IGNORECASE),
    ]
    divergence_detected = False
    for line in text.splitlines():
        for pat in divergence_patterns:
            if pat.search(line):
                divergence_detected = True
                break
        if divergence_detected:
            break

    if not divergence_detected and residual_values and len(residual_values) >= 3:
        recent = residual_values[-3:]
        if all(recent[i+1] > recent[i] * 1.5 for i in range(len(recent)-1)):
            divergence_detected = True

    data["divergence_detection"] = divergence_detected

    if residual_values and len(residual_values) >= 2:
        rates = []
        for i in range(1, len(residual_values)):
            if residual_values[i-1] != 0:
                rates.append(residual_values[i] / residual_values[i-1])
        data["convergence_rate"] = _safe_float(np.mean(rates)) if rates else None
    else:
        data["convergence_rate"] = None

    return data


def process_solver_file(filepath):
    """Full solver pipeline: read → extract → validate → score."""
    result = {
        "file_name": os.path.basename(filepath),
        "file_type": os.path.splitext(filepath)[1].lower(),
    }
    try:
        result["file_size"] = os.path.getsize(filepath)
    except OSError:
        result["file_size"] = None

    try:
        with open(filepath, "r", errors="replace") as f:
            text = f.read()
    except Exception as e:
        result["error"] = f"File read failed: {e}"
        for m in SOLVER_METRICS:
            result.setdefault(m, None)
        result["flags"] = ["read_failure"]
        result["confidence_score"] = calculate_confidence_score(result["flags"])
        return _safe_serialize(result)

    try:
        metrics = extract_solver_features(text)
        result.update(metrics)
    except Exception as e:
        result["error"] = f"Solver extraction failed: {e}"
        for m in SOLVER_METRICS:
            result.setdefault(m, None)
        result["flags"] = ["read_failure"]
        result["confidence_score"] = calculate_confidence_score(result["flags"])
        return _safe_serialize(result)

    flags = validate_results(result, "solver")
    result["flags"] = flags
    result["confidence_score"] = calculate_confidence_score(flags)
    return _safe_serialize(result)


# ===========================================================================
# VALIDATION & CONFIDENCE
# ===========================================================================

def validate_results(data, file_category):
    """Apply validation flags based on computed metric values."""
    flags = []

    if data.get("error") and "read" in str(data.get("error", "")).lower():
        flags.append("read_failure")

    if file_category == "cad":
        if data.get("degenerate_faces") is not None and data["degenerate_faces"] > 0:
            flags.append("degenerate_geometry")
        if data.get("non_manifold_edges") is not None and data["non_manifold_edges"] > 0:
            flags.append("non_manifold_geometry")

        required = ["volume", "surface_area", "bounding_box", "dimensions", "centroid",
                     "face_count", "edge_count", "vertex_count", "solid_count"]
        if any(data.get(k) is None for k in required):
            flags.append("missing_metrics")

    elif file_category == "mesh":
        if data.get("duplicate_nodes") is not None and data["duplicate_nodes"] > 0:
            flags.append("duplicate_nodes")
        if data.get("duplicate_elements") is not None and data["duplicate_elements"] > 0:
            flags.append("duplicate_elements")
        if data.get("isolated_nodes") is not None and data["isolated_nodes"] > 0:
            flags.append("isolated_nodes")
        if data.get("non_manifold_edges") is not None and data["non_manifold_edges"] > 0:
            flags.append("non_manifold_geometry")
        if data.get("mean_quality") is not None and data["mean_quality"] < 0.3:
            flags.append("low_quality_mesh")

        required = ["node_count", "element_count", "mesh_volume", "surface_area",
                     "min_element_size", "connected_components"]
        missing = sum(1 for k in required if data.get(k) is None)
        if missing > 0:
            flags.append("missing_metrics")

    elif file_category == "solver":
        if data.get("divergence_detection") is True:
            flags.append("solver_divergence")

        required = ["warning_count", "error_count", "iteration_count",
                     "convergence_status", "residual_values"]
        missing = sum(1 for k in required if data.get(k) is None)
        if missing > 0:
            flags.append("missing_metrics")

    return flags


def calculate_confidence_score(flags):
    """Compute confidence score per the spec: start at 1.0, apply penalties, clamp [0,1]."""
    score = 1.0
    if "read_failure" in flags:
        score -= 0.3
    if "degenerate_geometry" in flags:
        score -= 0.2
    if "non_manifold_geometry" in flags:
        score -= 0.2
    if "low_quality_mesh" in flags:
        score -= 0.2
    if "solver_divergence" in flags:
        score -= 0.3
    if "missing_metrics" in flags:
        score -= 0.1
    score = max(0.0, min(1.0, score))
    return round(score, 2)


# ===========================================================================
# MARKDOWN REPORT
# ===========================================================================

def generate_markdown_report(results, output_path):
    """Generate the analysis_report.md file."""
    cad_results = [r for r in results if r.get("_category") == "cad"]
    mesh_results = [r for r in results if r.get("_category") == "mesh"]
    solver_results = [r for r in results if r.get("_category") == "solver"]

    lines = []
    lines.append("# Engineering Analysis Report\n")

    lines.append("## Summary")
    lines.append(f"- CAD files processed: {len(cad_results)}")
    lines.append(f"- Mesh files processed: {len(mesh_results)}")
    lines.append(f"- Solver files processed: {len(solver_results)}")
    lines.append("")

    def _format_metrics(result):
        display = {k: v for k, v in result.items() if k not in ("_category",)}
        return json.dumps(display, indent=2, default=str)

    if cad_results:
        lines.append("## CAD Analysis")
        for r in cad_results:
            lines.append(f"### {r.get('file_name', 'unknown')}")
            lines.append("```json")
            lines.append(_format_metrics(r))
            lines.append("```")
            lines.append("")

    if mesh_results:
        lines.append("## Mesh Analysis")
        for r in mesh_results:
            lines.append(f"### {r.get('file_name', 'unknown')}")
            lines.append("```json")
            lines.append(_format_metrics(r))
            lines.append("```")
            lines.append("")

    if solver_results:
        lines.append("## Solver Analysis")
        for r in solver_results:
            lines.append(f"### {r.get('file_name', 'unknown')}")
            lines.append("```json")
            lines.append(_format_metrics(r))
            lines.append("```")
            lines.append("")

    flagged = [r for r in results if r.get("flags")]
    lines.append("## Observations")
    if flagged:
        for r in flagged:
            fname = r.get("file_name", "unknown")
            for flag in r.get("flags", []):
                desc = _flag_description(flag, r)
                lines.append(f"- **{fname}**: `{flag}` — {desc}")
    else:
        lines.append("- No issues detected in any processed files.")
    lines.append("")

    lines.append("## Confidence Summary")
    lines.append("| File | Type | Score | Flags |")
    lines.append("|---|---|---|---|")
    for r in results:
        fname = r.get("file_name", "unknown")
        ftype = r.get("_category", "unknown").upper()
        score = r.get("confidence_score", "N/A")
        flags_str = ", ".join(r.get("flags", [])) if r.get("flags") else "none"
        lines.append(f"| {fname} | {ftype} | {score} | {flags_str} |")
    lines.append("")

    report_text = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    return report_text


def _flag_description(flag, result):
    """Return a brief factual description for a given flag."""
    descriptions = {
        "read_failure": "File could not be parsed or loaded",
        "degenerate_geometry": f"Found {result.get('degenerate_faces', '?')} degenerate face(s)",
        "non_manifold_geometry": f"Detected {result.get('non_manifold_edges', '?')} non-manifold edge(s)",
        "not_watertight": "Mesh is not watertight (has open boundaries)",
        "duplicate_nodes": f"Found {result.get('duplicate_nodes', '?')} duplicate node(s)",
        "duplicate_elements": f"Found {result.get('duplicate_elements', '?')} duplicate element(s)",
        "isolated_nodes": f"Found {result.get('isolated_nodes', '?')} isolated node(s)",
        "low_quality_mesh": f"Mean quality {result.get('mean_quality', '?')} is below 0.3 threshold",
        "solver_divergence": "Solver divergence detected in output",
        "missing_metrics": "One or more expected metrics could not be computed",
    }
    return descriptions.get(flag, "Validation issue detected")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    """AutoCheck 360 pipeline entry point."""
    # script_dir = str(Path(__file__).resolve().parent)
    script_dir = r"C:\Users\Shreyas Durge\Downloads\bar-subjected-to-axial-loads-1.snapshot.1\pansoft_files"
    print(f"AutoCheck 360 — Scanning: {script_dir}")
    print("=" * 72)

    files = scan_directory(script_dir)
    if not files:
        print("No supported engineering files found in the working directory.")
        return

    print(f"Found {len(files)} supported file(s).\n")

    results = []
    for filepath in files:
        category = detect_file_type(filepath)
        if category is None:
            continue

        print(f"Processing [{category.upper()}]: {os.path.basename(filepath)} ...")

        if category == "cad":
            result = process_cad_file(filepath)
        elif category == "mesh":
            result = process_mesh_file(filepath)
        elif category == "solver":
            result = process_solver_file(filepath)
        else:
            continue

        result["_category"] = category

        display = {k: v for k, v in result.items() if k != "_category"}
        print(json.dumps(display, indent=2, default=str))
        print()

        results.append(result)

    if not results:
        print("No files could be processed.")
        return

    report_path = os.path.join(script_dir, "analysis_report_02_pansoft.md")
    generate_markdown_report(results, report_path)
    print("=" * 72)
    print(f"Report saved to: {report_path}")

    cad_count = sum(1 for r in results if r.get("_category") == "cad")
    mesh_count = sum(1 for r in results if r.get("_category") == "mesh")
    solver_count = sum(1 for r in results if r.get("_category") == "solver")
    print(f"Summary: {cad_count} CAD, {mesh_count} Mesh, {solver_count} Solver files analyzed.")


if __name__ == "__main__":
    main()
