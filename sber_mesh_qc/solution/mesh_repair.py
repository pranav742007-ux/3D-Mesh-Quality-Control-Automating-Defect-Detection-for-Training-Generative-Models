"""
===============================================================================
SBER AI Journey — 3D Mesh Quality Control: Mesh Repair Engine  [v7.2]
===============================================================================
Automated 3D Geometric Mesh Repair & Topological Sanitization Engine:
  - Ear-clipping boundary loop triangulation to close non-manifold holes ('open' fix)
  - Degenerate face & unreferenced floating vertex purging ('artifacts' & 'simple' fix)
  - Master auto_repair_mesh entrypoint with detailed repair reporting
===============================================================================
"""

import numpy as np
from typing import Tuple, Dict, Any, List


def purge_degenerate_faces(vertices: np.ndarray, faces: np.ndarray, min_area: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Remove degenerate (zero-area) faces and unreferenced floating vertices.
    """
    if vertices is None or faces is None or len(vertices) == 0 or len(faces) == 0:
        return vertices, faces, 0

    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    
    valid_mask = areas > min_area
    purged_count = int(np.sum(~valid_mask))
    clean_faces = faces[valid_mask]

    if len(clean_faces) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=int), len(faces)

    # Remap vertices to remove floating unreferenced nodes
    unique_v_idx, remapped_faces = np.unique(clean_faces, return_inverse=True)
    clean_vertices = vertices[unique_v_idx]
    clean_faces = remapped_faces.reshape(-1, 3)

    return clean_vertices, clean_faces, purged_count


def find_boundary_edge_loops(faces: np.ndarray) -> List[List[int]]:
    """
    Find non-manifold boundary edge loops (edges referenced by exactly 1 face).
    """
    if faces is None or len(faces) == 0:
        return []

    edges = np.vstack([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]]
    ])
    sorted_edges = np.sort(edges, axis=1)
    unique_edges, counts = np.unique(sorted_edges, axis=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1]

    if len(boundary_edges) == 0:
        return []

    from collections import defaultdict
    adj = defaultdict(list)
    for u, v in boundary_edges:
        adj[u].append(v)
        adj[v].append(u)

    visited = set()
    loops = []

    for start_node in adj:
        if start_node in visited:
            continue

        loop = [start_node]
        visited.add(start_node)
        curr = start_node

        while True:
            # Prioritize closing the loop back to the start node (H7)
            if len(loop) >= 3 and start_node in adj.get(curr, []):
                loops.append(loop)
                break
                
            neighbors = [n for n in adj.get(curr, []) if n not in visited]
            if not neighbors:
                break
            nxt = neighbors[0]
            loop.append(nxt)
            visited.add(nxt)
            curr = nxt

    return loops


def triangulate_loop_3d(vertices: np.ndarray, loop: list) -> list:
    """
    Robust 3D boundary loop triangulation using 2D projection and ear-clipping.
    Falls back to fan triangulation if ear-clipping fails to clip all ears.
    """
    n_pts = len(loop)
    if n_pts < 3:
        return []

    # 1. Project 3D loop vertices onto their PCA plane of best fit
    loop_pts = vertices[loop]
    centroid = loop_pts.mean(axis=0)
    centered = loop_pts - centroid
    cov = np.dot(centered.T, centered) / n_pts
    try:
        _, eigenvectors = np.linalg.eigh(cov)
        # The first two eigenvectors form the 2D basis
        basis_x = eigenvectors[:, 2]
        basis_y = eigenvectors[:, 1]
        pts_2d = np.zeros((n_pts, 2))
        pts_2d[:, 0] = np.dot(centered, basis_x)
        pts_2d[:, 1] = np.dot(centered, basis_y)
    except Exception:
        # Fallback to simple XY projection
        pts_2d = loop_pts[:, :2]

    # 2. Ear-clipping algorithm in 2D
    def is_convex(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]) >= 0

    def in_triangle(p, a, b, c):
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
        d1 = sign(p, a, b)
        d2 = sign(p, b, c)
        d3 = sign(p, c, a)
        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        return not (has_neg and has_pos)

    # Check overall orientation (clockwise vs counter-clockwise)
    orient_sum = 0.0
    for i in range(n_pts):
        p_curr = pts_2d[i]
        p_next = pts_2d[(i + 1) % n_pts]
        orient_sum += (p_next[0] - p_curr[0]) * (p_next[1] + p_curr[1])
    is_ccw = orient_sum < 0

    remaining_indices = list(range(n_pts))
    triangles = []
    
    max_iters = n_pts * 10
    iter_count = 0
    
    while len(remaining_indices) >= 3 and iter_count < max_iters:
        iter_count += 1
        clipped = False
        L = len(remaining_indices)
        for i in range(L):
            idx_u = remaining_indices[i - 1]
            idx_v = remaining_indices[i]
            idx_w = remaining_indices[(i + 1) % L]
            
            u = loop[idx_u]
            v = loop[idx_v]
            w = loop[idx_w]
            
            pu, pv, pw = pts_2d[idx_u], pts_2d[idx_v], pts_2d[idx_w]
            
            convex = is_convex(pu, pv, pw) == is_ccw
            if not convex:
                continue
                
            ear = True
            for j in range(L):
                idx_j = remaining_indices[j]
                if idx_j in (idx_u, idx_v, idx_w):
                    continue
                pj = pts_2d[idx_j]
                if in_triangle(pj, pu, pv, pw):
                    ear = False
                    break
                    
            if ear:
                triangles.append([u, v, w])
                remaining_indices.pop(i)
                clipped = True
                break
                
        if not clipped:
            break

    # 3. Fallback to fan triangulation for remaining vertices
    if len(remaining_indices) >= 3:
        v0 = loop[remaining_indices[0]]
        for i in range(1, len(remaining_indices) - 1):
            v1 = loop[remaining_indices[i]]
            v2 = loop[remaining_indices[i + 1]]
            triangles.append([v0, v1, v2])
            
    return triangles


def repair_open_holes(vertices: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Ear-clipping triangulation algorithm closing non-manifold boundary edge loops.
    """
    if vertices is None or faces is None or len(faces) == 0:
        return vertices, faces, 0

    loops = find_boundary_edge_loops(faces)
    if not loops:
        return vertices, faces, 0

    existing_edges = set()
    for f in faces:
        existing_edges.add((f[0], f[1]))
        existing_edges.add((f[1], f[2]))
        existing_edges.add((f[2], f[0]))

    new_faces = list(faces)
    holes_filled = 0

    for loop in loops:
        if len(loop) < 3:
            continue
        
        loop_triangles = triangulate_loop_3d(vertices, loop)
        added_any = False
        
        for v0, v1, v2 in loop_triangles:
            # Skip if tri edge already exists in mesh (prevents duplicate non-manifold edges)
            if (v0, v1) in existing_edges or (v1, v2) in existing_edges or (v2, v0) in existing_edges:
                continue
            
            # Geometric validation: check if the new triangle is degenerate (zero-area)
            tri_pts = vertices[[v0, v1, v2]]
            cross_prod = np.cross(tri_pts[1] - tri_pts[0], tri_pts[2] - tri_pts[0])
            area = 0.5 * np.linalg.norm(cross_prod)
            if area < 1e-8:
                continue

            new_faces.append([v0, v1, v2])
            existing_edges.add((v0, v1))
            existing_edges.add((v1, v2))
            existing_edges.add((v2, v0))
            added_any = True
            
        if added_any:
            holes_filled += 1

    return vertices, np.array(new_faces, dtype=int), holes_filled


def auto_repair_mesh(vertices: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Executes degenerate face purging and boundary hole filling.
    """
    if vertices is None or faces is None:
        return vertices, faces, {"repaired": False, "reason": "Empty inputs"}

    vertices, faces, purged_count = purge_degenerate_faces(vertices, faces)
    vertices, faces, holes_filled = repair_open_holes(vertices, faces)

    report = {
        "repaired": bool(purged_count > 0 or holes_filled > 0),
        "degenerate_faces_purged": purged_count,
        "boundary_holes_filled": holes_filled,
        "final_vertex_count": len(vertices),
        "final_face_count": len(faces),
    }

    return vertices, faces, report
