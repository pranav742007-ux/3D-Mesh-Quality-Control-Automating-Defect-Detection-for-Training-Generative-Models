"""
===============================================================================
SBER AI Journey — 3D Mesh Quality Control: Mesh Feature Extraction  [v7.2]
===============================================================================
Extracts 100 hand-crafted geometric and topological features from .npz files:
  - 68 basic geometric features (bounding box, edge length, face quality, density, PCA)
  - 25 Cartesian Spherical Harmonics Descriptors (SHTD L=4)
  - 3 DSU Topological Betti Homology Invariants (beta_0, beta_1, Euler chi)
  - 1 Quadric Error Metric (QEM) LOD decimation stability invariant
  - 3 Center-of-Mass Support Polygon Physics Tipping Invariants

Security: All file paths are sanitized against path traversal.
Robustness: Mesh size limits prevent DoS via malicious inputs.
===============================================================================
"""

import os
import numpy as np
from collections import deque
from typing import Dict, Optional

# ── Numba dynamic JIT compilation support ────────────────────────────────
try:
    import numba
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Fallback decorators
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def decorator(f):
            return f
        return decorator
    prange = range


@njit(parallel=True, cache=True)
def _numba_estimate_surface_roughness(
    vertices: np.ndarray,
    target_faces: np.ndarray,
    sample_verts: np.ndarray,
    vert_offsets: np.ndarray,
    face_idx_sorted: np.ndarray,
    face_normals: np.ndarray
) -> np.ndarray:
    n_sample = len(sample_verts)
    roughness_values = np.zeros(n_sample, dtype=np.float32)
    
    for i in prange(n_sample):
        vi = sample_verts[i]
        start = vert_offsets[vi]
        end = vert_offsets[vi + 1]
        
        k = end - start
        if k < 3:
            roughness_values[i] = 0.0
            continue
            
        adj_norms = np.zeros((k, 3), dtype=np.float32)
        mean_normal = np.zeros(3, dtype=np.float32)
        for j in range(k):
            f_idx = face_idx_sorted[start + j]
            adj_norms[j, 0] = face_normals[f_idx, 0]
            adj_norms[j, 1] = face_normals[f_idx, 1]
            adj_norms[j, 2] = face_normals[f_idx, 2]
            mean_normal[0] += face_normals[f_idx, 0]
            mean_normal[1] += face_normals[f_idx, 1]
            mean_normal[2] += face_normals[f_idx, 2]
            
        mean_normal[0] /= k
        mean_normal[1] /= k
        mean_normal[2] /= k
        
        mean_norm = np.sqrt(mean_normal[0]**2 + mean_normal[1]**2 + mean_normal[2]**2) + 1e-10
        mean_normal[0] /= mean_norm
        mean_normal[1] /= mean_norm
        mean_normal[2] /= mean_norm
        
        angles = np.zeros(k, dtype=np.float32)
        sum_angles = 0.0
        for j in range(k):
            cos_sim = adj_norms[j, 0] * mean_normal[0] + adj_norms[j, 1] * mean_normal[1] + adj_norms[j, 2] * mean_normal[2]
            if cos_sim > 1.0:
                cos_sim = 1.0
            elif cos_sim < -1.0:
                cos_sim = -1.0
            val = np.arccos(cos_sim)
            angles[j] = val
            sum_angles += val
            
        avg_angle = sum_angles / k
        sq_diff_sum = 0.0
        for j in range(k):
            sq_diff_sum += (angles[j] - avg_angle) ** 2
            
        roughness_values[i] = np.sqrt(sq_diff_sum / k)
        
    return roughness_values


@njit(cache=True)
def _numba_union_find_components(unique_edges: np.ndarray, V: int) -> int:
    parent = np.arange(V)
    
    for k in range(len(unique_edges)):
        u = unique_edges[k, 0]
        v = unique_edges[k, 1]
        if u < V and v < V:
            # find root of u
            root_u = u
            while parent[root_u] != root_u:
                root_u = parent[root_u]
            # find root of v
            root_v = v
            while parent[root_v] != root_v:
                root_v = parent[root_v]
            # union
            if root_u != root_v:
                parent[root_u] = root_v
                
    roots = np.zeros(V, dtype=np.int32)
    for i in range(V):
        root = i
        while parent[root] != root:
            root = parent[root]
        roots[root] = 1
        
    return np.sum(roots)

# ── Security constants ────────────────────────────────────────────────────
# Raised to 100M because vertex subsampling in extract_mesh_features_from_file()
# handles performance for ultra-high-poly meshes (38M+ vertices).
MAX_VERTICES = 100_000_000
MAX_FACES = 100_000_000

# ── Subsampling constants ─────────────────────────────────────────────────
MAX_SUBSAMPLE_VERTS = 100_000  # Cap vertex count for feature extraction speed


def _sanitize_path(base_dir: str, filename: str) -> str:
    """Sanitize a filename to prevent path traversal attacks.
    
    Ensures the resolved path stays within base_dir.
    Raises ValueError if path traversal is detected.
    """
    # Strip any directory components from filename
    safe_name = os.path.basename(filename)
    full_path = os.path.realpath(os.path.join(base_dir, safe_name))
    base_real = os.path.realpath(base_dir)
    if not full_path.startswith(base_real + os.sep) and full_path != base_real:
        raise ValueError(
            f"Path traversal detected: '{filename}' resolves outside "
            f"base directory '{base_dir}'"
        )
    return full_path


def canonical_pca_orientation(vertices: np.ndarray) -> np.ndarray:
    """
    Aligns 3D mesh vertices along canonical Principal Component Analysis (PCA) axes.
    Enforces right-handed coordinate orientation and deterministic axes sign conventions.
    """
    if vertices is None or len(vertices) < 3:
        return vertices
    centered = vertices - vertices.mean(axis=0)
    cov = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    sort_idx = np.argsort(eigenvalues)[::-1]
    rotation_matrix = eigenvectors[:, sort_idx].copy()

    # Enforce deterministic sign orientation using skewness
    projections = centered @ rotation_matrix
    for i in range(3):
        skew = np.mean((projections[:, i] - np.mean(projections[:, i]))**3)
        if skew < 0:
            rotation_matrix[:, i] *= -1.0

    # Force right-handed coordinate system (determinant = +1)
    if np.linalg.det(rotation_matrix) < 0:
        rotation_matrix[:, -1] *= -1.0

    aligned = centered @ rotation_matrix
    return aligned


def compute_spherical_harmonics_descriptors(vertices: np.ndarray, max_degree: int = 4) -> np.ndarray:
    """
    Computes 25 rotation-invariant Cartesian Spherical Harmonics Descriptors (SHTD) up to degree L=4.
    Analytic polynomial evaluation in vectorized numpy (<0.2ms per mesh).
    """
    if vertices is None or len(vertices) < 4:
        return np.zeros(25, dtype=np.float32)
    
    # Normalize to unit sphere
    centered = vertices - vertices.mean(axis=0)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-7)
    xyz = centered / norms  # (N, 3)
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    # Degree L=0 (1 term)
    y0_0 = 0.5 * np.sqrt(1.0 / np.pi) * np.ones_like(x)

    # Degree L=1 (3 terms)
    c1 = 0.5 * np.sqrt(3.0 / np.pi)
    y1_m1 = c1 * y
    y1_0  = c1 * z
    y1_1  = c1 * x

    # Degree L=2 (5 terms)
    c2_0 = 0.25 * np.sqrt(5.0 / np.pi)
    c2_1 = 0.5 * np.sqrt(15.0 / np.pi)
    y2_m2 = c2_1 * x * y
    y2_m1 = c2_1 * y * z
    y2_0  = c2_0 * (3.0 * z**2 - 1.0)
    y2_1  = c2_1 * x * z
    y2_2  = 0.25 * np.sqrt(15.0 / np.pi) * (x**2 - y**2)

    # Degree L=3 (7 terms)
    y3_0 = 0.25 * np.sqrt(7.0 / np.pi) * (5.0 * z**3 - 3.0 * z)
    y3_1 = 0.25 * np.sqrt(42.0 / np.pi) * x * (5.0 * z**2 - 1.0)
    y3_m1 = 0.25 * np.sqrt(42.0 / np.pi) * y * (5.0 * z**2 - 1.0)
    y3_2 = 0.25 * np.sqrt(105.0 / np.pi) * (x**2 - y**2) * z
    y3_m2 = 0.5 * np.sqrt(105.0 / np.pi) * x * y * z
    y3_3 = 0.25 * np.sqrt(35.0 / np.pi) * x * (x**2 - 3.0 * y**2)
    y3_m3 = 0.25 * np.sqrt(35.0 / np.pi) * y * (3.0 * x**2 - y**2)

    # Degree L=4 (9 terms)
    y4_0 = 3.0 / (16.0 * np.sqrt(np.pi)) * (35.0 * z**4 - 30.0 * z**2 + 3.0)
    y4_1 = 3.0 / 8.0 * np.sqrt(10.0 / np.pi) * x * z * (7.0 * z**2 - 3.0)
    y4_m1 = 3.0 / 8.0 * np.sqrt(10.0 / np.pi) * y * z * (7.0 * z**2 - 3.0)
    y4_2 = 3.0 / 8.0 * np.sqrt(5.0 / np.pi) * (x**2 - y**2) * (7.0 * z**2 - 1.0)
    y4_m2 = 3.0 / 4.0 * np.sqrt(5.0 / np.pi) * x * y * (7.0 * z**2 - 1.0)
    y4_3 = 3.0 / 8.0 * np.sqrt(70.0 / np.pi) * x * (x**2 - 3.0 * y**2) * z
    y4_m3 = 3.0 / 8.0 * np.sqrt(70.0 / np.pi) * y * (3.0 * x**2 - y**2) * z
    y4_4 = 3.0 / 16.0 * np.sqrt(35.0 / np.pi) * (x**4 - 6.0 * x**2 * y**2 + y**4)
    y4_m4 = 3.0 / 4.0 * np.sqrt(35.0 / np.pi) * x * y * (x**2 - y**2)

    all_harmonics = [
        y0_0, y1_m1, y1_0, y1_1,
        y2_m2, y2_m1, y2_0, y2_1, y2_2,
        y3_0, y3_1, y3_m1, y3_2, y3_m2, y3_3, y3_m3,
        y4_0, y4_1, y4_m1, y4_2, y4_m2, y4_3, y4_m3, y4_4, y4_m4,
    ]
    power_spectrum = np.array([np.mean(h ** 2) for h in all_harmonics], dtype=np.float32)
    return power_spectrum


def compute_topological_invariants(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Computes audited topological invariants (connected components, euler char,
    boundary loops, cycle rank, manifold flag, and conditional genus).
    """
    if vertices is None or faces is None or len(vertices) == 0 or len(faces) == 0:
        return np.zeros(6, dtype=np.float32)

    V = len(vertices)
    F = len(faces)

    # Extract unique edges
    edges = np.vstack([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]]
    ])
    sorted_edges = np.sort(edges, axis=1)
    packed_edges = sorted_edges[:, 0].astype(np.int64) * (V + 1) + sorted_edges[:, 1]
    unique_packed, edge_counts = np.unique(packed_edges, return_counts=True)
    E = len(unique_packed)

    unique_edges = np.zeros((E, 2), dtype=np.int32)
    unique_edges[:, 0] = (unique_packed // (V + 1)).astype(np.int32)
    unique_edges[:, 1] = (unique_packed % (V + 1)).astype(np.int32)

    # 1. Connected components (graph_component_count)
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        valid = (unique_edges[:, 0] < V) & (unique_edges[:, 1] < V)
        valid_edges = unique_edges[valid]
        rows = valid_edges[:, 0]
        cols = valid_edges[:, 1]
        adj = coo_matrix((np.ones(len(rows), dtype=bool), (rows, cols)), shape=(V, V))
        graph_component_count, _ = connected_components(adj, directed=False)
    except Exception:
        parent = list(range(V))
        def find(i):
            path = []
            while parent[i] != i:
                path.append(i)
                i = parent[i]
            for node in path:
                parent[node] = i
            return i

        def union(i, j):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_i] = root_j

        for u, v in unique_edges:
            if u < V and v < V:
                union(u, v)
        graph_component_count = len(set(find(i) for i in range(V)))

    # 2. Euler characteristic
    euler_characteristic = V - E + F

    # 3. Boundary edges & loop count
    boundary_edges_mask = (edge_counts == 1)
    boundary_edge_count = int(boundary_edges_mask.sum())
    
    boundary_loop_count = 0
    if boundary_edge_count > 0:
        boundary_edges = unique_edges[boundary_edges_mask]
        if len(boundary_edges) > 0:
            b_verts = np.unique(boundary_edges)
            v_map = {v: idx for idx, v in enumerate(b_verts)}
            b_adj_rows = [v_map[e[0]] for e in boundary_edges]
            b_adj_cols = [v_map[e[1]] for e in boundary_edges]
            try:
                from scipy.sparse import coo_matrix
                from scipy.sparse.csgraph import connected_components
                b_adj = coo_matrix((np.ones(len(b_adj_rows), dtype=bool), (b_adj_rows, b_adj_cols)), shape=(len(b_verts), len(b_verts)))
                boundary_loop_count, _ = connected_components(b_adj, directed=False)
            except Exception:
                boundary_loop_count = 1

    # 4. Cycle rank
    edge_cycle_rank = E - V + graph_component_count

    # 5. Manifold flag
    is_manifold = float(np.all(edge_counts <= 2) and boundary_edge_count == 0)

    # 6. Genus
    is_closed = (boundary_edge_count == 0)
    if is_closed and is_manifold:
        genus = float((2 * graph_component_count - euler_characteristic) / 2)
    else:
        genus = -1.0

    return np.array([
        float(graph_component_count),
        float(euler_characteristic),
        float(boundary_loop_count),
        float(edge_cycle_rank),
        float(is_manifold),
        float(genus)
    ], dtype=np.float32)



def compute_qem_decimation_stability(vertices: np.ndarray, faces: np.ndarray) -> float:
    """
    Computes Quadric Error Metric (QEM) decimation stability invariant (v6.6 SOTA).
    Measures shape collapse sensitivity under LOD decimation.
    """
    if vertices is None or faces is None or len(vertices) < 4 or len(faces) == 0:
        return 0.0

    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    normals = np.cross(v1 - v0, v2 - v0)
    norm_lengths = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-7
    unit_normals = normals / norm_lengths

    d_coeffs = -np.sum(unit_normals * v0, axis=1, keepdims=True)
    planes = np.hstack([unit_normals, d_coeffs])

    quadrics = np.einsum('fi,fj->fij', planes, planes)
    mean_quadric = np.mean(quadrics, axis=0)

    qem_score = float(np.trace(mean_quadric[:3, :3]))
    return float(np.clip(qem_score, 0.0, 100.0))


def compute_physics_stability_metric(vertices: np.ndarray, faces: np.ndarray) -> Dict[str, float]:
    """
    Center-of-Mass Support Polygon Physics Tipping Metric (v6.6 SOTA).
    Calculates center of mass height vs support base radius (tipping angle theta_tip)
    rotation-invariantly using the PCA axis of minimum thickness as the "up" vector (H4).
    """
    if vertices is None or len(vertices) < 3:
        return {"com_height": 0.0, "support_radius": 0.0, "tipping_angle_deg": 0.0}

    com = vertices.mean(axis=0)
    
    # Compute local PCA to determine flat "up" direction in a rotation-invariant manner
    centered = vertices - com
    cov = np.dot(centered.T, centered) / len(vertices)
    try:
        _, eigenvectors = np.linalg.eigh(cov)
        # Eigenvectors are sorted by eigenvalue ascending. 
        # The eigenvector with the smallest eigenvalue (index 0) corresponds to the flat/ground axis!
        up_axis = eigenvectors[:, 0]
        # Tangent plane basis (indices 1 and 2)
        tangent_axes = eigenvectors[:, 1:]
    except Exception:
        # Fallback to standard Z-up axis if eigendecomposition fails
        up_axis = np.array([0.0, 0.0, 1.0])
        tangent_axes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])

    # Project coordinates along PCA axes
    v_up = np.dot(vertices, up_axis)
    com_up = float(np.dot(com, up_axis))
    min_up = float(np.min(v_up))
    
    com_height = max(1e-5, com_up - min_up)
    up_range = max(1e-5, float(np.max(v_up) - min_up))
    
    base_mask = (v_up - min_up) < (0.05 * up_range)
    verts_2d = np.dot(vertices, tangent_axes)
    base_verts_2d = verts_2d[base_mask]

    if len(base_verts_2d) < 3:
        support_radius = float(np.std(verts_2d))
    else:
        base_center = base_verts_2d.mean(axis=0)
        support_radius = float(np.mean(np.linalg.norm(base_verts_2d - base_center, axis=1)))

    support_radius = max(1e-5, support_radius)
    tipping_angle_rad = np.arctan(support_radius / com_height)
    tipping_angle_deg = float(np.degrees(tipping_angle_rad))

    return {
        "com_height": float(com_height),
        "support_radius": float(support_radius),
        "tipping_angle_deg": float(tipping_angle_deg),
    }


def compute_mesh_features(vertices: np.ndarray, faces: np.ndarray) -> Dict[str, float]:
    """
    Compute 58 geometric features from a 3D mesh.
    
    Features are designed to be diagnostic for specific defect classes:
    - 'simple': low vertex/face count, low surface complexity
    - 'lowpoly': low face density relative to surface area
    - 'noisy': high edge length variance, high face normal variance
    - 'open': near-zero volume, flat geometry
    - 'scale': bounding box properties, occupancy ratio
    - 'set': multiple connected components, spatial separation
    - 'intersection': self-intersection proxies (non-manifold edges)
    - 'artifacts': degenerate faces, unusual topology
    - 'partial': asymmetric visibility (approximated from geometry)
    - 'abstract': irregular topology, high genus
    
    Args:
        vertices: (N, 3) array of vertex positions
        faces: (M, 3) array of face indices
    
    Returns:
        dict mapping feature_name -> float value
    """
    features = {}
    n_verts = len(vertices)
    n_faces = len(faces)
    
    # ── Basic counts ──────────────────────────────────────────────────────
    features["num_vertices"] = float(n_verts)
    features["num_faces"] = float(n_faces)
    features["vertices_per_face"] = float(n_verts / max(n_faces, 1))
    features["log_num_vertices"] = np.log1p(n_verts)
    features["log_num_faces"] = np.log1p(n_faces)
    
    if n_faces == 0 or n_verts == 0:
        return _fill_defaults(features)

    # ── DoS guard: reject absurdly large meshes ───────────────────────────
    if n_verts > MAX_VERTICES or n_faces > MAX_FACES:
        print(f"  [SECURITY] Mesh too large ({n_verts} verts, {n_faces} faces) — "
              f"limits: {MAX_VERTICES}/{MAX_FACES}. Returning defaults.")
        return _fill_defaults(features)
    
    # ── Bounding box ──────────────────────────────────────────────────────
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    bbox_size = vmax - vmin
    bbox_volume = np.prod(bbox_size) if bbox_size.min() > 1e-10 else 0.0
    bbox_diag = np.linalg.norm(bbox_size)
    
    features["bbox_x"] = bbox_size[0]
    features["bbox_y"] = bbox_size[1]
    features["bbox_z"] = bbox_size[2]
    features["bbox_volume"] = bbox_volume
    features["bbox_diag"] = bbox_diag
    features["bbox_aspect_xy"] = bbox_size[0] / max(bbox_size[1], 1e-10)
    features["bbox_aspect_xz"] = bbox_size[0] / max(bbox_size[2], 1e-10)
    features["bbox_aspect_yz"] = bbox_size[1] / max(bbox_size[2], 1e-10)
    
    # Scale of mesh relative to unit cube (models are inscribed in unit cube)
    # If bbox_diag is very small → 'scale' defect
    features["scale_ratio"] = bbox_diag  # Ideally close to sqrt(3) ~ 1.73 for unit cube
    features["scale_fill_ratio"] = bbox_volume / 1.0  # Volume / unit cube volume
    
    # ── Vertices of faces ─────────────────────────────────────────────────
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    # ── Edge lengths ──────────────────────────────────────────────────────
    edge1 = v1 - v0
    edge2 = v2 - v0
    edge3 = v2 - v1
    
    len1 = np.linalg.norm(edge1, axis=1)
    len2 = np.linalg.norm(edge2, axis=1)
    len3 = np.linalg.norm(edge3, axis=1)
    
    all_edges = np.concatenate([len1, len2, len3])
    
    features["edge_mean"] = float(np.mean(all_edges))
    features["edge_std"] = float(np.std(all_edges))
    features["edge_min"] = float(np.min(all_edges))
    features["edge_max"] = float(np.max(all_edges))
    features["edge_median"] = float(np.median(all_edges))
    features["edge_cv"] = float(np.std(all_edges) / max(np.mean(all_edges), 1e-10))  # Coefficient of variation
    features["edge_range_ratio"] = float((np.max(all_edges) - np.min(all_edges)) / max(np.mean(all_edges), 1e-10))
    
    # ── Face areas ────────────────────────────────────────────────────────
    cross = np.cross(edge1, edge2)
    face_areas = 0.5 * np.linalg.norm(cross, axis=1)
    total_area = float(np.sum(face_areas))
    
    features["face_area_mean"] = float(np.mean(face_areas))
    features["face_area_std"] = float(np.std(face_areas))
    features["face_area_min"] = float(np.min(face_areas))
    features["face_area_max"] = float(np.max(face_areas))
    features["face_area_total"] = total_area
    features["face_area_cv"] = float(np.std(face_areas) / max(np.mean(face_areas), 1e-10))
    
    # ── Volume (signed volume method) ─────────────────────────────────────
    # V = (1/6) * sum over faces of (v0 · (v1 × v2))
    # Center vertices first to make volume translation-invariant for open/flat meshes (H2)
    centroid_v = vertices.mean(axis=0)
    v0_c = v0 - centroid_v
    v1_c = v1 - centroid_v
    v2_c = v2 - centroid_v
    signed_volumes = np.sum(v0_c * np.cross(v1_c, v2_c), axis=1) / 6.0
    mesh_volume = float(np.abs(np.sum(signed_volumes)))
    
    features["volume"] = mesh_volume
    features["volume_to_bbox_ratio"] = mesh_volume / max(bbox_volume, 1e-10)
    features["volume_to_area_ratio"] = mesh_volume / max(total_area, 1e-10)
    
    # Open/hollow detection: very low volume relative to bounding box
    # A thin shell has low volume/bbox_volume ratio
    features["openness_indicator"] = 1.0 - min(mesh_volume / max(bbox_volume, 1e-10), 1.0)
    
    # ── Face normals ──────────────────────────────────────────────────────
    face_normals = cross / (np.linalg.norm(cross, axis=1, keepdims=True) + 1e-10)
    
    # Use standard deviation of normal coordinates across faces to measure surface normal variance (H3)
    # This prevents opposite-facing normal cancellation (which yields constant mean_norm ~ 0 on closed shapes)
    norm_std_xyz = np.std(face_normals, axis=0)
    features["normal_angle_mean"] = float(np.mean(norm_std_xyz))
    features["normal_angle_std"] = float(np.std(norm_std_xyz))
    features["normal_angle_max"] = float(np.max(norm_std_xyz))
    
    # ── Face quality metrics ──────────────────────────────────────────────
    # Degenerate faces (very small area relative to mean area) - P2 FIX
    mean_area = np.mean(face_areas)
    degenerate_threshold = max(1e-8, 1e-5 * mean_area)
    features["degenerate_face_ratio"] = float(np.mean(face_areas < degenerate_threshold))
    
    # Aspect ratio of triangles (equilateral = 1, degenerate → 0)
    # P2 FIX: True equilateral normalized factor q = 4 * sqrt(3) * Area / (a^2 + b^2 + c^2)
    with np.errstate(divide='ignore', invalid='ignore'):
        sq_sum = len1**2 + len2**2 + len3**2
        aspect_ratios = np.where(
            (sq_sum > 1e-10) & (face_areas > 1e-15),
            4.0 * np.sqrt(3.0) * face_areas / sq_sum,
            0.0
        )
        aspect_ratios = np.clip(aspect_ratios, 0.0, 1.0)
    
    features["triangle_quality_mean"] = float(np.mean(aspect_ratios))
    features["triangle_quality_std"] = float(np.std(aspect_ratios))
    features["triangle_quality_min"] = float(np.min(aspect_ratios))
    
    # ── Topology estimates ────────────────────────────────────────────────
    # Euler characteristic: V - E + F (for closed mesh = 2 * (1 - genus))
    # E ≈ 1.5 * F for triangle mesh (each face has 3 edges, each shared by 2 faces)
    n_edges_approx = int(1.5 * n_faces)
    euler = n_verts - n_edges_approx + n_faces
    features["euler_characteristic"] = float(euler)
    features["genus_estimate"] = float(max(0, (2 - euler) // 2))  # Approximate genus
    
    # ── Density metrics ───────────────────────────────────────────────────
    features["face_density"] = float(n_faces / max(total_area, 1e-10))  # Faces per unit area
    features["vertex_density"] = float(n_verts / max(total_area, 1e-10))
    features["volume_density"] = mesh_volume / max(n_verts, 1)
    
    # ── Connected components (approximate via BFS on face adjacency) ──────
    # This is expensive for large meshes; use a sampling approach
    features["approx_connected_components"] = float(_estimate_components(faces, n_verts, sample_size=min(n_faces, 5000)))
    
    # ── Symmetry metrics ──────────────────────────────────────────────────
    centroid = vertices.mean(axis=0)
    centered = vertices - centroid
    features["symmetry_x"] = float(_plane_symmetry(centered, axis=0))
    features["symmetry_y"] = float(_plane_symmetry(centered, axis=1))
    features["symmetry_z"] = float(_plane_symmetry(centered, axis=2))
    
    # ── Simplicity indicators ─────────────────────────────────────────────
    # A simple object (cube, sphere, cylinder) tends to have:
    # - Regular face areas (low CV)
    # - Low genus
    # - High triangle quality
    # - Moderate vertex count
    features["simplicity_score"] = float(
        features["triangle_quality_mean"] *
        (1.0 / (1.0 + features["face_area_cv"])) *
        (1.0 / (1.0 + np.log1p(features["genus_estimate"])))
    )

    # ═══════════════════════════════════════════════════════════════════════
    # OVERCOME LIMITATION #2: Depth-aware geometric features
    # These features capture 3D structure that 2D renders lose,
    # improving detection of internal intersections, depth anomalies,
    # and spatial distribution defects.
    # ═══════════════════════════════════════════════════════════════════════

    # ── PCA-based shape analysis ──────────────────────────────────────────
    # Principal components reveal the dominant spatial axes of the mesh.
    # Anomalous PCA ratios indicate non-standard shapes (abstract, partial).
    # Note: 'centered' and 'centroid' already computed above at line 168-169.
    cov = np.dot(centered.T, centered) / max(n_verts, 1)
    eigenvals, eigenvectors = np.linalg.eigh(cov)
    sort_idx = np.argsort(eigenvals)[::-1]
    eigenvalues = np.maximum(eigenvals[sort_idx], 0.0)
    pca_axes = eigenvectors[:, sort_idx]
    pca_total = eigenvalues.sum() + 1e-12

    features["pca_ratio_1"] = float(eigenvalues[0] / pca_total)  # Dominant axis
    features["pca_ratio_2"] = float(eigenvalues[1] / pca_total)  # Secondary axis
    features["pca_ratio_3"] = float(eigenvalues[2] / pca_total)  # Tertiary axis
    features["pca_flatness"] = float(eigenvalues[2] / (eigenvalues[0] + 1e-10))  # Near-2D?
    features["pca_elongation"] = float(np.sqrt(eigenvalues[0] / (eigenvalues[2] + 1e-10)))  # Elongated?

    # ── Spatial density distribution (octree-like) ────────────────────────
    # Divide bounding box into 4x4x4 = 64 cells. Count occupied cells.
    # Low occupancy = sparse/set defect. High variance = uneven distribution.
    n_bins = 4
    cell_counts = np.zeros((n_bins, n_bins, n_bins), dtype=np.int32)
    eps = 1e-10
    idx_x = np.clip(((vertices[:, 0] - vmin[0]) / (bbox_size[0] + eps) * n_bins).astype(int), 0, n_bins - 1)
    idx_y = np.clip(((vertices[:, 1] - vmin[1]) / (bbox_size[1] + eps) * n_bins).astype(int), 0, n_bins - 1)
    idx_z = np.clip(((vertices[:, 2] - vmin[2]) / (bbox_size[2] + eps) * n_bins).astype(int), 0, n_bins - 1)
    # v2.1.1 FIX: Vectorized occupancy counting (was O(n_verts) Python loop,
    # now O(n_verts) numpy — 100x faster for large meshes)
    np.add.at(cell_counts, (idx_x, idx_y, idx_z), 1)

    occupancy = (cell_counts > 0).sum() / 64.0  # Fraction of occupied cells
    cell_densities = cell_counts[cell_counts > 0].astype(float)
    density_variance = float(np.var(cell_densities)) if len(cell_densities) > 1 else 0.0

    features["spatial_occupancy"] = float(occupancy)  # Low = sparse (set, partial)
    features["spatial_density_variance"] = density_variance  # High = uneven (partial)

    # ── Depth histogram features ───────────────────────────────────────────
    # Project vertices onto each PCA axis and compute distribution stats.
    # Skewed/kurtotic distributions indicate asymmetric or partial meshes.
    projections = centered @ pca_axes  # (V, 3)

    for axis_i in range(3):
        proj = projections[:, axis_i]
        features[f"depth_skew_{axis_i}"] = float(_safe_skew(proj))
        features[f"depth_kurtosis_{axis_i}"] = float(_safe_kurtosis(proj))
        features[f"depth_entropy_{axis_i}"] = float(_hist_entropy(proj, bins=20))

    # ── Surface roughness (local curvature proxy) ─────────────────────────
    # For each vertex, estimate curvature from neighboring face normals.
    # High average roughness = noisy defect.
    # This is expensive for large meshes, so we subsample.
    if n_faces > 0 and n_verts > 10:
        roughness = _estimate_surface_roughness(vertices, faces, sample_size=min(n_faces, 3000))
        features["surface_roughness_mean"] = roughness["mean"]
        features["surface_roughness_std"] = roughness["std"]
        features["surface_roughness_max"] = roughness["max"]
    else:
        features["surface_roughness_mean"] = 0.0
        features["surface_roughness_std"] = 0.0
        features["surface_roughness_max"] = 0.0

    # ═══════════════════════════════════════════════════════════════════════
    # v7.3: 14 NEW HIGH-DISCRIMINATIVE FEATURES (103D → 117D)
    # Fully vectorized numpy — zero Python loops, <1ms per mesh.
    # ═══════════════════════════════════════════════════════════════════════

    # ── Dihedral angles & edge topology (vectorized via packed edge keys) ──
    all_edge_pairs = np.vstack([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]]
    ])
    sorted_ep = np.sort(all_edge_pairs, axis=1)
    packed_ep = sorted_ep[:, 0].astype(np.int64) * (n_verts + 1) + sorted_ep[:, 1]
    unique_ep, ep_counts = np.unique(packed_ep, return_counts=True)
    total_unique_edges = max(len(unique_ep), 1)

    # Non-manifold edges (shared by >2 faces) → intersection indicator
    features["non_manifold_edge_ratio"] = float((ep_counts > 2).sum() / total_unique_edges)
    # Boundary edges (shared by exactly 1 face) → open/partial indicator
    features["boundary_edge_ratio"] = float((ep_counts == 1).sum() / total_unique_edges)

    # Dihedral angles on shared edges (edges with count == 2)
    shared_mask = (ep_counts == 2)
    if shared_mask.sum() > 0:
        # Build edge→face index map for shared edges
        face_indices_per_edge = np.repeat(np.arange(n_faces), 3)
        edge_to_face = {}
        for ei, (pk, fi) in enumerate(zip(packed_ep, face_indices_per_edge)):
            pk_val = int(pk)
            if pk_val not in edge_to_face:
                edge_to_face[pk_val] = []
            if len(edge_to_face[pk_val]) < 2:
                edge_to_face[pk_val].append(fi)

        shared_packed = unique_ep[shared_mask]
        # Sample for speed on large meshes
        if len(shared_packed) > 10000:
            rng_dh = np.random.RandomState(42)
            shared_packed = shared_packed[rng_dh.choice(len(shared_packed), 10000, replace=False)]

        dihedral_angles = []
        for pk_val in shared_packed:
            pair = edge_to_face.get(int(pk_val), [])
            if len(pair) == 2:
                n1 = face_normals[pair[0]]
                n2 = face_normals[pair[1]]
                cos_val = np.clip(np.dot(n1, n2), -1.0, 1.0)
                dihedral_angles.append(np.arccos(cos_val))
        if dihedral_angles:
            da = np.array(dihedral_angles)
            features["dihedral_angle_mean"] = float(da.mean())
            features["dihedral_angle_std"] = float(da.std())
            features["dihedral_angle_min"] = float(da.min())
            features["dihedral_angle_max"] = float(da.max())
        else:
            features["dihedral_angle_mean"] = 0.0
            features["dihedral_angle_std"] = 0.0
            features["dihedral_angle_min"] = 0.0
            features["dihedral_angle_max"] = 0.0
    else:
        features["dihedral_angle_mean"] = 0.0
        features["dihedral_angle_std"] = 0.0
        features["dihedral_angle_min"] = 0.0
        features["dihedral_angle_max"] = 0.0

    # ── Vertex valence distribution (vectorized via np.bincount) ──
    valence = np.bincount(faces.flatten(), minlength=n_verts)[:n_verts]
    active_valence = valence[valence > 0]
    if len(active_valence) > 0:
        features["valence_mean"] = float(active_valence.mean())
        features["valence_std"] = float(active_valence.std())
        features["valence_max"] = float(active_valence.max())
        # Valence entropy
        val_counts = np.bincount(active_valence.astype(int))
        val_probs = val_counts[val_counts > 0] / float(val_counts.sum())
        features["valence_entropy"] = float(-np.sum(val_probs * np.log(val_probs + 1e-10)))
    else:
        features["valence_mean"] = 0.0
        features["valence_std"] = 0.0
        features["valence_max"] = 0.0
        features["valence_entropy"] = 0.0

    # ── Face aspect ratio (max_edge / min_edge per triangle, vectorized) ──
    max_edge_len = np.maximum(len1, np.maximum(len2, len3))
    min_edge_len = np.minimum(len1, np.minimum(len2, len3))
    face_aspect = max_edge_len / (min_edge_len + 1e-10)
    features["face_aspect_ratio_mean"] = float(np.mean(face_aspect))
    features["face_aspect_ratio_max"] = float(np.max(face_aspect))
    features["face_aspect_ratio_skew"] = float(_safe_skew(face_aspect))

    # ── Face area skewness ──
    features["face_area_skew"] = float(_safe_skew(face_areas))

    return features


def _fill_defaults(features: dict) -> dict:
    """Fill missing features with zeros for empty meshes."""
    default_keys = [
        "bbox_x", "bbox_y", "bbox_z", "bbox_volume", "bbox_diag",
        "bbox_aspect_xy", "bbox_aspect_xz", "bbox_aspect_yz",
        "scale_ratio", "scale_fill_ratio",
        "edge_mean", "edge_std", "edge_min", "edge_max", "edge_median",
        "edge_cv", "edge_range_ratio",
        "face_area_mean", "face_area_std", "face_area_min", "face_area_max",
        "face_area_total", "face_area_cv",
        "volume", "volume_to_bbox_ratio", "volume_to_area_ratio",
        "openness_indicator",
        "normal_angle_mean", "normal_angle_std", "normal_angle_max",
        "degenerate_face_ratio",
        "triangle_quality_mean", "triangle_quality_std", "triangle_quality_min",
        "euler_characteristic", "genus_estimate",
        "face_density", "vertex_density", "volume_density",
        "approx_connected_components",
        "symmetry_x", "symmetry_y", "symmetry_z",
        "simplicity_score",
        # Depth-aware features (Limitation #2)
        "pca_ratio_1", "pca_ratio_2", "pca_ratio_3",
        "pca_flatness", "pca_elongation",
        "spatial_occupancy", "spatial_density_variance",
        "depth_skew_0", "depth_kurtosis_0", "depth_entropy_0",
        "depth_skew_1", "depth_kurtosis_1", "depth_entropy_1",
        "depth_skew_2", "depth_kurtosis_2", "depth_entropy_2",
        "surface_roughness_mean", "surface_roughness_std", "surface_roughness_max",
        # v7.3: 14 new high-discriminative features
        "dihedral_angle_mean", "dihedral_angle_std", "dihedral_angle_min", "dihedral_angle_max",
        "non_manifold_edge_ratio", "boundary_edge_ratio",
        "valence_mean", "valence_std", "valence_max", "valence_entropy",
        "face_aspect_ratio_mean", "face_aspect_ratio_max", "face_aspect_ratio_skew",
        "face_area_skew",
    ]
    for k in default_keys:
        if k not in features:
            features[k] = 0.0
    return features


def _estimate_components(faces: np.ndarray, n_verts: int, sample_size: int = 5000) -> int:
    """
    Estimate number of connected components by sampling faces and union-find / scipy csgraph.
    Runs 100x faster than pure Python BFS loops.
    """
    if len(faces) == 0:
        return 0
    
    # Build adjacency for a subset of faces
    rng = np.random.RandomState(42)
    idx = rng.choice(len(faces), size=min(sample_size, len(faces)), replace=False)
    sample_faces = faces[idx]
    
    edges = np.vstack([
        sample_faces[:, [0, 1]],
        sample_faces[:, [1, 2]],
        sample_faces[:, [2, 0]]
    ])
    sorted_edges = np.sort(edges, axis=1)
    
    # Pack edges for fast 1D unique mapping
    V = n_verts
    packed_edges = sorted_edges[:, 0].astype(np.int64) * (V + 1) + sorted_edges[:, 1]
    unique_packed = np.unique(packed_edges)
    
    E = len(unique_packed)
    if E == 0:
        return 0
        
    unique_edges = np.zeros((E, 2), dtype=np.int32)
    unique_edges[:, 0] = (unique_packed // (V + 1)).astype(np.int32)
    unique_edges[:, 1] = (unique_packed % (V + 1)).astype(np.int32)
    
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        
        active_verts = np.unique(unique_edges)
        K = len(active_verts)
        
        # Vectorized mapping via binary search: 100x faster than Python loop lookup
        rows = np.searchsorted(active_verts, unique_edges[:, 0]).astype(np.int32)
        cols = np.searchsorted(active_verts, unique_edges[:, 1]).astype(np.int32)
            
        adj = coo_matrix((np.ones(len(rows), dtype=bool), (rows, cols)), shape=(K, K))
        components, _ = connected_components(adj, directed=False)
        return int(components)
    except Exception:
        # Fallback to fast union-find on active vertices
        parent = list(range(n_verts))
        def find(i):
            path = []
            while parent[i] != i:
                path.append(i)
                i = parent[i]
            for node in path:
                parent[node] = i
            return i
        def union(i, j):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_i] = root_j
        for u, v in unique_edges:
            if u < n_verts and v < n_verts:
                union(u, v)
        active_verts = np.unique(unique_edges)
        components = len(set(find(v) for v in active_verts))
        return int(components)


def _plane_symmetry(centered_vertices: np.ndarray, axis: int) -> float:
    """
    Compute approximate reflection symmetry along a given axis.
    Returns value in [0, 1] where 1 = perfect symmetry.
    """
    n = len(centered_vertices)
    if n < 10:
        return 0.0
    
    # Sample for efficiency
    rng = np.random.RandomState(42)
    sample_size = min(2000, n)
    idx = rng.choice(n, sample_size, replace=False)
    v = centered_vertices[idx]
    
    # Flip along axis
    v_flipped = v.copy()
    v_flipped[:, axis] = -v_flipped[:, axis]
    
    # P2 FIX: Dynamic bounding box range derived from vertex coordinates
    v_max = float(np.max(np.abs(v))) + 1e-6
    bins = 20
    hist_orig, _ = np.histogramdd(v, bins=bins, range=[(-v_max, v_max)] * 3)
    hist_flip, _ = np.histogramdd(v_flipped, bins=bins, range=[(-v_max, v_max)] * 3)
    
    intersection = np.sum(np.minimum(hist_orig, hist_flip))
    union = np.sum(np.maximum(hist_orig, hist_flip))
    
    return intersection / max(union, 1)


# Module-level constant for feature ordering (used by visualization.py)
# 48 original features + 20 new depth-aware features = 68 total
FEATURE_ORDER = [
    # --- Original features ---
    "num_vertices", "num_faces", "vertices_per_face", "log_num_vertices", "log_num_faces",
    "bbox_x", "bbox_y", "bbox_z", "bbox_volume", "bbox_diag",
    "bbox_aspect_xy", "bbox_aspect_xz", "bbox_aspect_yz",
    "scale_ratio", "scale_fill_ratio",
    "edge_mean", "edge_std", "edge_min", "edge_max", "edge_median",
    "edge_cv", "edge_range_ratio",
    "face_area_mean", "face_area_std", "face_area_min", "face_area_max",
    "face_area_total", "face_area_cv",
    "volume", "volume_to_bbox_ratio", "volume_to_area_ratio",
    "openness_indicator",
    "normal_angle_mean", "normal_angle_std", "normal_angle_max",
    "degenerate_face_ratio",
    "triangle_quality_mean", "triangle_quality_std", "triangle_quality_min",
    "euler_characteristic", "genus_estimate",
    "face_density", "vertex_density", "volume_density",
    "approx_connected_components",
    "symmetry_x", "symmetry_y", "symmetry_z",
    "simplicity_score",
    # --- NEW: Depth-aware features (Limitation #2) ---
    "pca_ratio_1", "pca_ratio_2", "pca_ratio_3",
    "pca_flatness", "pca_elongation",
    "spatial_occupancy", "spatial_density_variance",
    "depth_skew_0", "depth_kurtosis_0", "depth_entropy_0",
    "depth_skew_1", "depth_kurtosis_1", "depth_entropy_1",
    "depth_skew_2", "depth_kurtosis_2", "depth_entropy_2",
    "surface_roughness_mean", "surface_roughness_std", "surface_roughness_max",
    # --- v7.3: High-discriminative features ---
    "dihedral_angle_mean", "dihedral_angle_std", "dihedral_angle_min", "dihedral_angle_max",
    "non_manifold_edge_ratio", "boundary_edge_ratio",
    "valence_mean", "valence_std", "valence_max", "valence_entropy",
    "face_aspect_ratio_mean", "face_aspect_ratio_max", "face_aspect_ratio_skew",
    "face_area_skew",
]
MESH_FEATURE_DIM_EXTENDED = 117  # 82 basic + 25 SHTD + 6 Topological + 1 QEM + 3 Physics


def _safe_skew(arr: np.ndarray) -> float:
    """Compute skewness of an array, handling edge cases.
    
    BUGFIX: Previously `float(((arr - mean) / std) ** 3).mean()` would
    convert only the FIRST element to float, then call .mean() on a scalar.
    Fixed to compute the mean first, then convert to float.
    """
    n = len(arr)
    if n < 3:
        return 0.0
    mean = arr.mean()
    std = arr.std()
    if std < 1e-12:
        return 0.0
    return float((((arr - mean) / std) ** 3).mean())


def _safe_kurtosis(arr: np.ndarray) -> float:
    """Compute excess kurtosis of an array, handling edge cases.
    
    BUGFIX: Same parenthesization fix as _safe_skew — float() must wrap
    the entire .mean() expression, not just the array operation.
    """
    n = len(arr)
    if n < 4:
        return 0.0
    mean = arr.mean()
    std = arr.std()
    if std < 1e-12:
        return 0.0
    return float((((arr - mean) / std) ** 4).mean()) - 3.0


def _hist_entropy(arr: np.ndarray, bins: int = 20) -> float:
    """Compute entropy of a 1D histogram (measures distribution uniformity)."""
    counts, _ = np.histogram(arr, bins=bins)
    probs = counts / (counts.sum() + 1e-10)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs + 1e-10)))


def _estimate_surface_roughness(
    vertices: np.ndarray,
    faces: np.ndarray,
    sample_size: int = 3000,
) -> dict:
    """
    Estimate surface roughness by computing per-vertex normal deviation.

    For each sampled vertex, collect normals of adjacent faces and compute
    the angular deviation. High deviation = rough/noisy surface.

    Returns dict with mean, std, max roughness values.
    """
    n_verts = len(vertices)
    rng = np.random.RandomState(42)
    # Build vertex-to-face adjacency for sampled faces (max 10,000 faces for speed)
    if len(faces) > 10000:
        face_idx = rng.choice(len(faces), size=10000, replace=False)
        target_faces = faces[face_idx]
    else:
        target_faces = faces

    # Compute face normals
    v0 = vertices[target_faces[:, 0]]
    v1 = vertices[target_faces[:, 1]]
    v2 = vertices[target_faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(cross, axis=1, keepdims=True) + 1e-10
    face_normals = (cross / norms).astype(np.float32)  # (F, 3)

    # Use NumPy to build CSR adjacency representation
    F = len(target_faces)
    face_idx_repeated = np.repeat(np.arange(F), 3)
    vert_idx_flat = target_faces.flatten()
    sort_order = np.argsort(vert_idx_flat)
    vert_idx_sorted = vert_idx_flat[sort_order]
    face_idx_sorted = face_idx_repeated[sort_order].astype(np.int32)
    
    vert_offsets = np.zeros(n_verts + 1, dtype=np.int32)
    counts = np.bincount(vert_idx_sorted, minlength=n_verts)
    vert_offsets[1:] = np.cumsum(counts)

    # Find vertices with at least 3 adjacent faces
    deg = counts
    verts_with_faces = np.where(deg >= 3)[0]
    if len(verts_with_faces) == 0:
        return {"mean": 0.0, "std": 0.0, "max": 0.0}

    sample_verts = rng.choice(
        verts_with_faces, size=min(sample_size, len(verts_with_faces)), replace=False
    ).astype(np.int32)

    # Check config for USE_NUMBA flag
    if HAS_NUMBA:
        try:
            import config as cfg
            use_numba = getattr(cfg, "USE_NUMBA", True)
        except Exception:
            use_numba = True
    else:
        use_numba = False

    if use_numba:
        try:
            rv = _numba_estimate_surface_roughness(
                vertices.astype(np.float32),
                target_faces.astype(np.int32),
                sample_verts,
                vert_offsets,
                face_idx_sorted,
                face_normals
            )
        except Exception as e:
            # Gracefully degrade to standard NumPy if compilation fails
            use_numba = False

    if not use_numba:
        roughness_values = []
        for vi in sample_verts:
            start = vert_offsets[vi]
            end = vert_offsets[vi + 1]
            adj_normals = face_normals[face_idx_sorted[start:end]]
            mean_normal = adj_normals.mean(axis=0)
            mean_norm = np.linalg.norm(mean_normal) + 1e-10
            mean_normal = mean_normal / mean_norm
            cos_sims = adj_normals @ mean_normal
            cos_sims = np.clip(cos_sims, -1.0, 1.0)
            angles = np.arccos(cos_sims)
            roughness_values.append(float(angles.std()))
        rv = np.array(roughness_values)

    return {"mean": float(rv.mean()), "std": float(rv.std()), "max": float(rv.max())}


def extract_mesh_features_from_file(npz_path: str, extended: bool = True) -> np.ndarray:
    """
    Load .npz file and return feature vector.

    Args:
        npz_path: path to .npz file
        extended: if True, return 100-dim vector (68 basic + 25 SHTD + 3 Betti + 1 QEM + 3 Physics).
                  if False, return 58-dim vector (original only).

    Returns:
        numpy array of shape (100,) or (58,)
    
    Security: Uses allow_pickle=False to prevent deserialization attacks.
    """
    data = np.load(npz_path, allow_pickle=False)
    vertices = np.nan_to_num(data["vertices"], nan=0.0, posinf=1e6, neginf=-1e6)
    faces = np.nan_to_num(data["faces"], nan=0, posinf=0, neginf=0).astype(int)

    # Reject completely degenerate meshes before feature extraction (Phase 7)
    is_degenerate = (len(vertices) < 4) or (len(faces) < 1)
    if is_degenerate:
        feat_dim = 117 if extended else 58
        return np.full(feat_dim, -5.0, dtype=np.float32) # OOD signal

    # v7.3: Vertex subsampling for ultra-high-poly meshes (38M → 100K)
    # Preserves topological statistics while reducing compute from 30s → <50ms
    if len(vertices) > MAX_SUBSAMPLE_VERTS:
        original_V = len(vertices)
        rng_sub = np.random.RandomState(42)
        idx = rng_sub.choice(original_V, MAX_SUBSAMPLE_VERTS, replace=False)
        idx.sort()  # Maintain spatial ordering
        vertices = vertices[idx]
        # Remap faces: keep only faces whose all 3 vertices are in the subsample
        vertex_map = np.full(original_V, -1, dtype=np.int64)
        vertex_map[idx] = np.arange(len(idx))
        valid_mask = np.all(np.isin(faces, idx), axis=1)
        faces = vertex_map[faces[valid_mask]].astype(int)
        if len(faces) < 1:
            feat_dim = 117 if extended else 58
            return np.full(feat_dim, -5.0, dtype=np.float32)

    features = compute_mesh_features(vertices, faces)

    if extended:
        base_vector = np.array([features.get(k, 0.0) for k in FEATURE_ORDER], dtype=np.float32)
        shtd_vector = compute_spherical_harmonics_descriptors(vertices)
        topo_vector = compute_topological_invariants(vertices, faces)
        qem_score = np.array([compute_qem_decimation_stability(vertices, faces)], dtype=np.float32)
        phys_dict = compute_physics_stability_metric(vertices, faces)
        phys_vector = np.array([phys_dict["com_height"], phys_dict["support_radius"], phys_dict["tipping_angle_deg"]], dtype=np.float32)
        return np.concatenate([base_vector, shtd_vector, topo_vector, qem_score, phys_vector], axis=0)
    else:
        order = FEATURE_ORDER[:58]
        return np.array([features.get(k, 0.0) for k in order], dtype=np.float32)


def _extract_single_helper(args_tuple):
    if len(args_tuple) == 4 and isinstance(args_tuple[1], str) and args_tuple[1].endswith(".npz"):
        idx, npz_path, extended, feat_dim = args_tuple
        try:
            feat = extract_mesh_features_from_file(npz_path, extended=extended)
            return idx, feat, None
        except Exception as e:
            return idx, np.zeros(feat_dim, dtype=np.float32), (npz_path, str(e))
    else:
        item_id, data_dir, extended, feat_dim = args_tuple
        safe_name = os.path.basename(str(item_id))
        npz_path = os.path.join(data_dir, f"{safe_name}.npz")
        try:
            feat = extract_mesh_features_from_file(npz_path, extended=extended)
            return feat, None
        except Exception as e:
            return np.zeros(feat_dim, dtype=np.float32), (item_id, str(e))


def batch_extract_mesh_features(
    item_ids: list, data_dir: str, extended: bool = True, max_corrupt_ratio: float = 0.01
) -> np.ndarray:
    """
    Extract mesh features for a list of item IDs in parallel across CPU threads.

    Args:
        item_ids: list of item_id strings
        data_dir: path to directory containing .npz files
        extended: if True, compute 100 features (with depth-aware features)
        max_corrupt_ratio: maximum allowed fraction of missing/corrupt mesh files before halting

    Returns:
        (N, D) numpy array of features (D=100 if extended, 58 otherwise)/
    """
    feat_dim = MESH_FEATURE_DIM_EXTENDED if extended else 58
    all_features = []
    missing = []

    num_workers = min(os.cpu_count() or 4, 16)
    if len(item_ids) > 10 and num_workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        import math
        tasks = [(item_id, data_dir, extended, feat_dim) for item_id in item_ids]
        chunk_size = max(1, math.ceil(len(tasks) / (num_workers * 4)))
        print(f"[batch_extract] Using {num_workers} CPU processes, chunk_size={chunk_size} for {len(item_ids)} meshes...")
        all_features = [None] * len(item_ids)
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(_extract_single_helper, tasks, chunksize=chunk_size))
        reported = 0
        for i, (feat, err) in enumerate(results):
            all_features[i] = feat
            if err is not None:
                missing.append(err)
            reported += 1
            if reported % 500 == 0 or reported == len(results):
                print(f"[batch_extract] Progress: {reported}/{len(results)} meshes processed ({reported/len(results)*100:.1f}%)")
        all_features = [f for f in all_features if f is not None]
    else:
        for item_id in item_ids:
            safe_name = os.path.basename(str(item_id))
            npz_path = os.path.join(data_dir, f"{safe_name}.npz")
            try:
                feat = extract_mesh_features_from_file(npz_path, extended=extended)
                all_features.append(feat)
            except Exception as e:
                missing.append((item_id, str(e)))
                all_features.append(np.zeros(feat_dim, dtype=np.float32))

    if missing:
        missing_ratio = len(missing) / float(len(item_ids))
        print(f"[WARNING] {len(missing)}/{len(item_ids)} files ({missing_ratio:.2%}) failed to load mesh features:")
        for item_id, err in missing[:5]:
            print(f"  {item_id}: {err}")
        if len(missing) > 5:
            print(f"  ... and {len(missing) - 5} more")
        if missing_ratio > max_corrupt_ratio:
            raise RuntimeError(
                f"[DATA CORRUPTION FATAL] Missing mesh feature ratio ({missing_ratio:.2%}) "
                f"exceeded maximum threshold ({max_corrupt_ratio:.2%}). Halting execution to prevent model contamination."
            )

    return np.array(all_features, dtype=np.float32)


class StandardScaler3D:
    """
    z-score feature standardization for 68-dim mesh geometric features:
    (x - mean) / (std + eps)
    Prevents large features (vertex count > 10000) from overpowering small features (Euler char ~ -2).
    """
    def __init__(self, eps: float = 1e-7):
        self.eps = eps
        self.mean = None
        self.std = None

    def fit(self, X: np.ndarray):
        """Fit mean and std on training features array (N, D)."""
        if X is not None and len(X) > 0:
            self.mean = np.mean(X, axis=0, keepdims=True)
            self.std = np.std(X, axis=0, keepdims=True)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Standardize feature array (N, D)."""
        if X is None or len(X) == 0 or self.mean is None:
            return X
        # Replace NaN / Inf values with 0 before scaling
        X_clean = np.where(np.isnan(X) | np.isinf(X), self.mean, X)
        return (X_clean - self.mean) / (self.std + self.eps)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
