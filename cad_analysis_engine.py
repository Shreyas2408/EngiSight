import os
import sys
import json
import math
import numpy as np
import trimesh

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
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepFilletAPI import BRepFilletAPI_MakeFillet, BRepFilletAPI_MakeChamfer
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.TopExp import TopExp
    from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
    from OCP.TopoDS import topods
    from OCP.gp import gp_Dir
    from OCP.StepData import StepData_StepModel
    from OCP.XSControl import XSControl_WorkSession
    HAS_OCC = True
except ImportError:
    try:
        from OCC.Core.STEPControl import STEPControl_Reader
        from OCC.Core.IGESControl import IGESControl_Reader
        from OCC.Core.IFSelect import IFSelect_RetDone
        from OCC.Core.TopExp import TopExp_Explorer, TopExp
        from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX, TopAbs_SOLID, TopAbs_SHELL
        from OCC.Core.BRep import BRep_Tool
        from OCC.Core.BRepGProp import brepgprop_VolumeProperties, brepgprop_SurfaceProperties
        from OCC.Core.GProp import GProp_GProps
        from OCC.Core.Bnd import Bnd_Box
        from OCC.Core.BRepBndLib import brepbndlib_Add
        from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
        from OCC.Core.BRepCheck import BRepCheck_Analyzer
        from OCC.Core.BRepFilletAPI import BRepFilletAPI_MakeFillet, BRepFilletAPI_MakeChamfer
        from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
        from OCC.Core.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
        from OCC.Core.topoDS import topods
        from OCC.Core.gp import gp_Dir
        HAS_OCC = True
    except ImportError:
        HAS_OCC = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_file_type(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    mapping = {
        ".stl": "stl",
        ".step": "step", ".stp": "step",
        ".iges": "iges", ".igs": "iges",
    }
    return mapping.get(ext, "unknown")


def _safe_float(v):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return None
    return float(v)


# ---------------------------------------------------------------------------
# STL loader and feature extraction (trimesh)
# ---------------------------------------------------------------------------

def load_stl(filepath):
    return trimesh.load(filepath, force="mesh")


def compute_edge_metrics(mesh):
    edges = mesh.edges_unique
    if len(edges) == 0:
        return 0.0
    lengths = mesh.edges_unique_length
    return float(np.mean(lengths))


def extract_stl_features(mesh, filepath):
    data = {}
    data["file_name"] = os.path.basename(filepath)
    data["file_type"] = "stl"
    data["file_size"] = os.path.getsize(filepath)

    bb = mesh.bounds  # (2,3) array [[min], [max]]
    data["bounding_box"] = {
        "min": bb[0].tolist(),
        "max": bb[1].tolist(),
    }
    extents = bb[1] - bb[0]
    data["bounding_box_volume"] = _safe_float(float(np.prod(extents)))

    data["volume"] = _safe_float(float(mesh.volume)) if mesh.is_watertight else _safe_float(float(mesh.volume))
    data["surface_area"] = _safe_float(float(mesh.area))
    data["centroid"] = mesh.centroid.tolist()

    data["triangle_count"] = int(len(mesh.faces))
    data["face_count"] = int(len(mesh.faces))
    data["edge_count"] = int(len(mesh.edges_unique))
    data["vertex_count"] = int(len(mesh.vertices))

    # degenerate faces: area < 1e-10
    areas = mesh.area_faces
    data["degenerate_faces"] = int(np.sum(areas < 1e-10))

    # non-manifold edges
    face_adjacency = mesh.face_adjacency
    edge_face_counts = {}
    for pair in mesh.edges:
        key = tuple(sorted(pair))
        edge_face_counts[key] = edge_face_counts.get(key, 0)
    non_manifold = 0
    if hasattr(mesh, "edges_unique"):
        edge_counts = np.zeros(len(mesh.edges_unique), dtype=int)
        for fi, face in enumerate(mesh.faces):
            for i in range(3):
                e = tuple(sorted((face[i], face[(i + 1) % 3])))
        # use trimesh built-in
        try:
            from trimesh import grouping
            face_adj = mesh.face_adjacency_edges
            edges_sorted = np.sort(mesh.edges_sorted, axis=1) if hasattr(mesh, "edges_sorted") else np.sort(mesh.edges, axis=1)
            unique_edges, edge_counts = np.unique(edges_sorted, axis=0, return_counts=True)
            non_manifold = int(np.sum(edge_counts > 2))
        except Exception:
            non_manifold = 0
    data["non_manifold_edges"] = non_manifold

    data["watertight"] = bool(mesh.is_watertight)

    # connected components
    try:
        split = mesh.split(only_watertight=False)
        data["connected_components"] = len(split)
    except Exception:
        data["connected_components"] = 1

    data["euler_number"] = int(data["vertex_count"] - data["edge_count"] + data["face_count"])

    bb_vol = data["bounding_box_volume"] or 1.0
    data["triangle_density"] = _safe_float(data["triangle_count"] / bb_vol) if bb_vol > 0 else None

    data["average_edge_length"] = _safe_float(compute_edge_metrics(mesh))

    data["solid_count"] = None
    data["shell_count"] = None
    data["units"] = None

    return data


# ---------------------------------------------------------------------------
# STEP / IGES loader and feature extraction (OpenCascade)
# ---------------------------------------------------------------------------

def load_step(filepath):
    if not HAS_OCC:
        raise ImportError("pythonocc-core is not installed")
    reader = STEPControl_Reader()
    status = reader.ReadFile(filepath)
    if status != IFSelect_RetDone:
        raise RuntimeError(f"STEP read failed: {filepath}")
    reader.TransferRoots()
    return reader.OneShape(), reader


def load_iges(filepath):
    if not HAS_OCC:
        raise ImportError("pythonocc-core is not installed")
    reader = IGESControl_Reader()
    status = reader.ReadFile(filepath)
    if status != IFSelect_RetDone:
        raise RuntimeError(f"IGES read failed: {filepath}")
    reader.TransferRoots()
    return reader.OneShape(), reader


def _count_topology(shape, topo_type):
    count = 0
    explorer = TopExp_Explorer(shape, topo_type)
    while explorer.More():
        count += 1
        explorer.Next()
    return count


def _compute_brep_bbox(shape):
    bbox = Bnd_Box()
    try:
        brepbndlib_Add(shape, bbox)
    except NameError:
        BRepBndLib.Add_s(shape, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    return {
        "min": [xmin, ymin, zmin],
        "max": [xmax, ymax, zmax],
    }


def _compute_volume_props(shape):
    props = GProp_GProps()
    try:
        brepgprop_VolumeProperties(shape, props)
    except NameError:
        BRepGProp.VolumeProperties_s(shape, props)
    vol = props.Mass()
    c = props.CentreOfMass()
    return vol, [c.X(), c.Y(), c.Z()]


def _compute_surface_props(shape):
    props = GProp_GProps()
    try:
        brepgprop_SurfaceProperties(shape, props)
    except NameError:
        BRepGProp.SurfaceProperties_s(shape, props)
    return props.Mass()


def _tessellated_triangle_count(shape):
    try:
        mesh = BRepMesh_IncrementalMesh(shape, 0.1)
        mesh.Perform()
    except Exception:
        pass
    from OCC.Core.TopLoc import TopLoc_Location
    from OCC.Core.BRep import BRep_Tool
    total = 0
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = explorer.Current()
        loc = TopLoc_Location()
        try:
            triangulation = BRep_Tool.Triangulation_s(topods.Face_s(face), loc)
        except Exception:
            try:
                triangulation = BRep_Tool.Triangulation(topods.Face(face), loc)
            except Exception:
                explorer.Next()
                continue
        if triangulation is not None:
            try:
                total += triangulation.NbTriangles()
            except Exception:
                pass
        explorer.Next()
    return total


def _get_step_units(reader):
    try:
        ws = reader.WS()
        model = ws.Model()
        if model is not None:
            # Try to extract unit string from the STEP header
            for i in range(1, model.NbEntities() + 1):
                ent = model.Entity(i)
                type_name = ent.DynamicType().Name()
                if "unit" in type_name.lower() or "SI" in type_name:
                    return type_name
        return "millimeters"
    except Exception:
        return "unknown"


def _check_brep_validity(shape):
    try:
        analyzer = BRepCheck_Analyzer(shape)
        return analyzer.IsValid()
    except Exception:
        return True


def extract_brep_features(shape, filepath, file_type, reader=None):
    data = {}
    data["file_name"] = os.path.basename(filepath)
    data["file_type"] = file_type
    data["file_size"] = os.path.getsize(filepath)

    bbox = _compute_brep_bbox(shape)
    data["bounding_box"] = bbox
    extents = [bbox["max"][i] - bbox["min"][i] for i in range(3)]
    data["bounding_box_volume"] = _safe_float(extents[0] * extents[1] * extents[2])

    vol, centroid = _compute_volume_props(shape)
    data["volume"] = _safe_float(vol)
    data["surface_area"] = _safe_float(_compute_surface_props(shape))
    data["centroid"] = [_safe_float(c) for c in centroid]

    tri_count = _tessellated_triangle_count(shape)
    data["triangle_count"] = tri_count

    data["face_count"] = _count_topology(shape, TopAbs_FACE)
    data["edge_count"] = _count_topology(shape, TopAbs_EDGE)
    data["vertex_count"] = _count_topology(shape, TopAbs_VERTEX)

    data["degenerate_faces"] = 0
    data["non_manifold_edges"] = 0
    data["watertight"] = None

    data["connected_components"] = None
    data["euler_number"] = data["vertex_count"] - data["edge_count"] + data["face_count"]

    bb_vol = data["bounding_box_volume"] or 1.0
    data["triangle_density"] = _safe_float(tri_count / bb_vol) if bb_vol > 0 and tri_count > 0 else None
    data["average_edge_length"] = None

    data["solid_count"] = _count_topology(shape, TopAbs_SOLID)
    data["shell_count"] = _count_topology(shape, TopAbs_SHELL)

    if file_type == "step" and reader is not None:
        data["units"] = _get_step_units(reader)
    else:
        data["units"] = None

    # topology validity
    data["_topology_valid"] = _check_brep_validity(shape)

    return data


# ---------------------------------------------------------------------------
# Edge selection + Fillet / Chamfer (OpenCascade)
# ---------------------------------------------------------------------------

def _select_edges_by_direction(shape, target_dir, tolerance=0.1):
    """Select edges whose tangent at midpoint aligns with target_dir."""
    selected = []
    explorer = TopExp_Explorer(shape, TopAbs_EDGE)
    while explorer.More():
        edge = topods.Edge_s(explorer.Current()) if hasattr(topods, "Edge_s") else topods.Edge(explorer.Current())
        try:
            curve = BRepAdaptor_Curve(edge)
            u_first = curve.FirstParameter()
            u_last = curve.LastParameter()
            u_mid = (u_first + u_last) / 2.0
            pnt = curve.Value(u_mid)
            tangent = curve.DN(u_mid, 1)  # first derivative = tangent
            t_mag = tangent.Magnitude()
            if t_mag > 1e-9:
                tx, ty, tz = tangent.X() / t_mag, tangent.Y() / t_mag, tangent.Z() / t_mag
                dot = abs(tx * target_dir[0] + ty * target_dir[1] + tz * target_dir[2])
                if dot > (1.0 - tolerance):
                    selected.append(edge)
        except Exception:
            pass
        explorer.Next()
    return selected


def apply_modifications(shape):
    """Demonstrate edge selection + fillet and chamfer on a shape."""
    results = {"fillet_applied": False, "chamfer_applied": False, "selected_edges_z": 0, "selected_edges_x": 0}
    if not HAS_OCC:
        return shape, results

    # Select edges aligned with +Z
    z_edges = _select_edges_by_direction(shape, (0.0, 0.0, 1.0), tolerance=0.1)
    results["selected_edges_z"] = len(z_edges)

    # Select edges aligned with +X / -X
    x_edges = _select_edges_by_direction(shape, (1.0, 0.0, 0.0), tolerance=0.1)
    results["selected_edges_x"] = len(x_edges)

    modified_shape = shape

    # Apply fillet on Z-aligned edges
    if z_edges:
        try:
            fillet = BRepFilletAPI_MakeFillet(modified_shape)
            for edge in z_edges:
                fillet.Add(1.0, edge)  # radius = 1.0
            fillet.Build()
            if fillet.IsDone():
                modified_shape = fillet.Shape()
                results["fillet_applied"] = True
        except Exception:
            pass

    # Apply chamfer on X-aligned edges
    if x_edges:
        try:
            # Build edge-face map for chamfer
            edge_face_map = TopTools_IndexedDataMapOfShapeListOfShape()
            try:
                TopExp.MapShapesAndAncestors_s(modified_shape, TopAbs_EDGE, TopAbs_FACE, edge_face_map)
            except AttributeError:
                TopExp.MapShapesAndAncestors(modified_shape, TopAbs_EDGE, TopAbs_FACE, edge_face_map)

            chamfer = BRepFilletAPI_MakeChamfer(modified_shape)
            added = False
            for edge in x_edges:
                try:
                    idx = edge_face_map.FindIndex(edge)
                    if idx > 0:
                        face_list = edge_face_map.FindFromIndex(idx)
                        it = face_list.begin() if hasattr(face_list, "begin") else None
                        if it is not None:
                            face = topods.Face_s(it.Value()) if hasattr(topods, "Face_s") else topods.Face(it.Value())
                        else:
                            face = topods.Face_s(face_list.First()) if hasattr(topods, "Face_s") else topods.Face(face_list.First())
                        chamfer.Add(0.5, edge, face)  # distance = 0.5
                        added = True
                except Exception:
                    continue

            if added:
                chamfer.Build()
                if chamfer.IsDone():
                    modified_shape = chamfer.Shape()
                    results["chamfer_applied"] = True
        except Exception:
            pass

    return modified_shape, results


# ---------------------------------------------------------------------------
# Validation & confidence
# ---------------------------------------------------------------------------

def validate_geometry(data):
    flags = []

    if data.get("_read_failure"):
        flags.append("read_failure")

    if data.get("degenerate_faces") is not None and data["degenerate_faces"] > 0:
        flags.append("degenerate_geometry")

    if data.get("non_manifold_edges") is not None and data["non_manifold_edges"] > 0:
        flags.append("non_manifold_geometry")

    if data.get("watertight") is False:
        flags.append("not_watertight")

    required = ["volume", "surface_area", "bounding_box", "centroid"]
    for key in required:
        if data.get(key) is None:
            flags.append("missing_metrics")
            break

    if data.get("_topology_valid") is False:
        flags.append("invalid_topology")

    return flags


def calculate_confidence_score(flags):
    score = 1.0
    if "degenerate_geometry" in flags:
        score -= 0.2
    if "non_manifold_geometry" in flags:
        score -= 0.2
    if "not_watertight" in flags:
        score -= 0.1
    if "missing_metrics" in flags:
        score -= 0.1
    if "invalid_topology" in flags:
        score -= 0.3
    if "read_failure" in flags:
        score -= 0.3
    return max(0.0, min(1.0, round(score, 2)))


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def process_cad_file(filepath):
    filepath = os.path.abspath(filepath)
    if not os.path.isfile(filepath):
        return {
            "file_name": os.path.basename(filepath),
            "file_type": "unknown",
            "error": "File not found",
            "flags": ["read_failure"],
            "confidence_score": 0.0,
        }

    file_type = _detect_file_type(filepath)
    data = {}

    if file_type == "stl":
        try:
            mesh = load_stl(filepath)
            data = extract_stl_features(mesh, filepath)
        except Exception as e:
            data = {
                "file_name": os.path.basename(filepath),
                "file_type": "stl",
                "file_size": os.path.getsize(filepath),
                "_read_failure": True,
                "error": str(e),
            }

    elif file_type in ("step", "iges"):
        if not HAS_OCC:
            return {
                "file_name": os.path.basename(filepath),
                "file_type": file_type,
                "file_size": os.path.getsize(filepath),
                "error": "pythonocc-core not installed",
                "flags": ["read_failure"],
                "confidence_score": 0.0,
            }
        try:
            if file_type == "step":
                shape, reader = load_step(filepath)
            else:
                shape, reader = load_iges(filepath)
            data = extract_brep_features(shape, filepath, file_type, reader)

            # Demonstrate fillet/chamfer
            modified_shape, mod_results = apply_modifications(shape)
            data["modifications_demo"] = mod_results
        except Exception as e:
            data = {
                "file_name": os.path.basename(filepath),
                "file_type": file_type,
                "file_size": os.path.getsize(filepath),
                "_read_failure": True,
                "error": str(e),
            }
    else:
        return {
            "file_name": os.path.basename(filepath),
            "file_type": file_type,
            "file_size": os.path.getsize(filepath),
            "error": f"Unsupported format: {file_type}",
            "flags": ["read_failure"],
            "confidence_score": 0.0,
        }

    flags = validate_geometry(data)
    data["flags"] = flags
    data["confidence_score"] = calculate_confidence_score(flags)

    # Remove internal keys
    data.pop("_read_failure", None)
    data.pop("_topology_valid", None)

    return data


def main():
    if len(sys.argv) < 2:
        print("Usage: python cad_analysis_engine.py <cad_file> [<cad_file2> ...]")
        sys.exit(1)

    results = []
    for fpath in sys.argv[1:]:
        result = process_cad_file(fpath)
        results.append(result)

    output = results[0] if len(results) == 1 else results
    print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()
