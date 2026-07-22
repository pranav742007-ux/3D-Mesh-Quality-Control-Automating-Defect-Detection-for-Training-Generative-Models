"""
===============================================================================
SBER AI Journey — 3D Mesh Quality Control: Mesh Validity Analyzer [v7.3]
===============================================================================
Phase 3 Deterministic Mesh Validation. Extracts exact topological validity flags,
non-manifold edges, open boundaries, connected components, and self-intersections.
===============================================================================
"""
import numpy as np

class MeshValidityAnalyzer:
    """
    Calculates deterministic, rule-based 3D mesh structure validation parameters.
    Returns clean structural descriptors that bypass image/coordinate variance.
    """
    @staticmethod
    def analyze_mesh(vertices: np.ndarray, faces: np.ndarray) -> dict:
        report = {
            "degenerate_face_ratio": 0.0,
            "duplicate_vertex_count": 0,
            "duplicate_face_count": 0,
            "boundary_edge_count": 0,
            "non_manifold_edge_count": 0,
            "connected_components": 1,
            "inconsistent_normal_edges": 0,
            "watertight": 1.0,
            "signed_volume": 0.0,
            "surface_area": 0.0,
            "self_intersections": 0,
        }

        if vertices is None or faces is None or len(vertices) < 3 or len(faces) == 0:
            report["watertight"] = 0.0
            return report

        V = len(vertices)
        F = len(faces)

        # 1. Scale computation
        bbox_min = vertices.min(axis=0)
        bbox_max = vertices.max(axis=0)
        scale = np.linalg.norm(bbox_max - bbox_min) + 1e-8

        # 2. Degenerate faces check (area < epsilon * scale^2)
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        areas = 0.5 * np.linalg.norm(cross, axis=1)
        degenerate_mask = (areas < (1e-12 * scale * scale))
        report["degenerate_face_ratio"] = float(degenerate_mask.sum() / F)
        report["surface_area"] = float(areas.sum())

        # 3. Duplicate vertices
        _, unique_verts = np.unique(np.round(vertices, 6), axis=0, return_index=True)
        report["duplicate_vertex_count"] = int(V - len(unique_verts))

        # 4. Duplicate faces
        sorted_faces = np.sort(faces, axis=1)
        _, unique_faces = np.unique(sorted_faces, axis=0, return_index=True)
        report["duplicate_face_count"] = int(F - len(unique_faces))

        # 5. Boundary & Non-Manifold Edges
        # Stack edges canonically (v_min, v_max)
        edges = np.vstack([
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]]
        ])
        sorted_edges = np.sort(edges, axis=1)
        packed_edges = sorted_edges[:, 0].astype(np.int64) * (V + 1) + sorted_edges[:, 1]
        _, edge_counts = np.unique(packed_edges, return_counts=True)
        
        boundary_edges = int((edge_counts == 1).sum())
        non_manifold_edges = int((edge_counts > 2).sum())
        
        report["boundary_edge_count"] = boundary_edges
        report["non_manifold_edge_count"] = non_manifold_edges

        # 6. Connected Components via BFS Face Adjacency
        # Map canonical edges to face indices
        edge_to_faces = {}
        for f_idx, f in enumerate(faces):
            e1 = tuple(sorted(f[[0, 1]]))
            e2 = tuple(sorted(f[[1, 2]]))
            e3 = tuple(sorted(f[[2, 0]]))
            for e in (e1, e2, e3):
                edge_to_faces.setdefault(e, []).append(f_idx)

        # Build adjacency graph: face to adjacent faces
        face_adj = [[] for _ in range(F)]
        for shared_faces in edge_to_faces.values():
            if len(shared_faces) > 1:
                for idx_i in shared_faces:
                    for idx_j in shared_faces:
                        if idx_i != idx_j:
                            face_adj[idx_i].append(idx_j)

        # BFS component labeling
        visited = np.zeros(F, dtype=bool)
        n_components = 0
        for i in range(F):
            if not visited[i]:
                n_components += 1
                queue = [i]
                visited[i] = True
                # Standard BFS loop
                head = 0
                while head < len(queue):
                    curr = queue[head]
                    head += 1
                    for neighbor in face_adj[curr]:
                        if not visited[neighbor]:
                            visited[neighbor] = True
                            queue.append(neighbor)

        report["connected_components"] = n_components

        # 7. Inconsistent normal windings
        # Winding check: if two faces share an edge, they must traverse it in opposite directions
        inconsistent_edges = 0
        for edge, shared in edge_to_faces.items():
            if len(shared) == 2:
                f1_idx, f2_idx = shared[0], shared[1]
                # Check orientation of edge in face 1 and face 2
                f1 = faces[f1_idx].tolist()
                f2 = faces[f2_idx].tolist()
                
                # Helper to find orientation of directed edge in face list
                def get_dir(f, e):
                    for idx in range(3):
                        if f[idx] == e[0] and f[(idx+1)%3] == e[1]:
                            return 1
                        if f[idx] == e[1] and f[(idx+1)%3] == e[0]:
                            return -1
                    return 0
                
                dir1 = get_dir(f1, edge)
                dir2 = get_dir(f2, edge)
                if dir1 != 0 and dir2 != 0 and dir1 == dir2:
                    inconsistent_edges += 1
        report["inconsistent_normal_edges"] = inconsistent_edges

        # 8. Watertightness & Volume
        is_watertight = (boundary_edges == 0 and non_manifold_edges == 0)
        report["watertight"] = 1.0 if is_watertight else 0.0

        if is_watertight:
            # Compute signed volume of closed mesh
            # Vol = 1/6 * sum( v0 . (v1 x v2) )
            volume = float(np.sum(np.sum(v0 * np.cross(v1, v2), axis=1)) / 6.0)
            report["signed_volume"] = volume
        else:
            report["signed_volume"] = 0.0

        # 9. Self-intersections approximation using AABB overlaps
        report["self_intersections"] = MeshValidityAnalyzer._compute_aabb_overlaps(vertices, faces)

        return report

    @staticmethod
    def _compute_aabb_overlaps(vertices: np.ndarray, faces: np.ndarray) -> int:
        F = len(faces)
        tri_points = vertices[faces]
        mins = np.min(tri_points, axis=1)
        maxs = np.max(tri_points, axis=1)

        if F > 200:
            rng = np.random.RandomState(42)
            idx = rng.choice(F, 200, replace=False)
            mins = mins[idx]
            maxs = maxs[idx]
            faces_subset = faces[idx]
            F = 200
        else:
            faces_subset = faces

        overlap_count = 0
        for i in range(F):
            if i + 1 >= F:
                continue
            # Overlap along all three coordinate axes
            overlap_mask = np.all(mins[i] <= maxs[i+1:], axis=1) & np.all(maxs[i] >= mins[i+1:], axis=1)
            overlapping_indices = np.where(overlap_mask)[0] + (i + 1)
            
            for j in overlapping_indices:
                # Exclude neighbors sharing vertices/edges
                f_i = faces_subset[i]
                f_j = faces_subset[j]
                if len(set(f_i).intersection(f_j)) > 0:
                    continue
                overlap_count += 1
                # Early stop: we only care if overlaps exist (> 0)
                return overlap_count
                
        return overlap_count
