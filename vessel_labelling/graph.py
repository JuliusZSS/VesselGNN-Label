"""Build a vessel graph with 9 node features from unlabelled centerline segments.

Node features (in this order):
    0-2  xyz coordinates (case translated so that its bounding-box minimum is at the origin)
    3    path distance (depth-first traversal depth, in nodes) from the root of each vessel tree
    4    bifurcation level (incremented when passing through a node of degree >= 3)
    5    mean angle between edges incident to the node (radians, -1 if undefined)
    6-8  segment orientation vector (last point - first point of the segment)
"""
import numpy as np
import torch
from scipy.interpolate import interp1d
from torch_geometric.data import Data

N_POINTS_PER_SEGMENT = 300


def resample_centerline(points, num_points=N_POINTS_PER_SEGMENT):
    """Resample a polyline to a fixed number of points (linear interpolation over point index)."""
    if points.shape[0] == num_points:
        return points
    src = np.linspace(0, 1, points.shape[0])
    dst = np.linspace(0, 1, num_points)
    return np.stack([interp1d(src, points[:, i], kind='linear')(dst) for i in range(3)], axis=1)


def _dfs_distance_level(n_nodes, neighbours):
    """Depth-first traversal from node 0 (then from any unvisited node), as used in training."""
    dist = -np.ones(n_nodes, dtype=np.float32)
    level = -np.ones(n_nodes, dtype=np.float32)
    visited = np.zeros(n_nodes, dtype=bool)

    for root in range(n_nodes):
        if visited[root]:
            continue
        visited[root] = True
        dist[root], level[root] = 0, 0
        stack = [(root, 0, int(len(neighbours[root]) >= 3), iter(neighbours[root]))]
        while stack:
            node, d, lvl_out, it = stack[-1]
            nxt = next(it, None)
            if nxt is None:
                stack.pop()
                continue
            if visited[nxt]:
                continue
            visited[nxt] = True
            dist[nxt], level[nxt] = d + 1, lvl_out
            stack.append((nxt, d + 1, lvl_out + int(len(neighbours[nxt]) >= 3), iter(neighbours[nxt])))
    return dist, level


def _mean_edge_angles(points, neighbours):
    """Mean pairwise angle between the unit vectors to a node's neighbours (float32, as in training)."""
    pts = torch.tensor(points, dtype=torch.float)
    ang = torch.zeros(len(points), dtype=torch.float)
    for i, nbrs in enumerate(neighbours):
        if len(nbrs) < 2:
            continue
        vecs = [v / v.norm() for v in (pts[nbrs] - pts[i]) if v.norm() > 1e-6]
        if len(vecs) < 2:
            continue
        a = [torch.acos(torch.clamp(torch.dot(vecs[j], vecs[k]), -1.0, 1.0))
             for j in range(len(vecs)) for k in range(j + 1, len(vecs))]
        ang[i] = torch.mean(torch.stack(a))
    ang[ang == 0] = -1
    return ang.numpy()


def topological_order(n_segments, connections):
    """Order segments so that every parent precedes its children, otherwise keeping the input
    order (the first listed root stays first). Graph traversal restarts at the first unvisited
    segment, so each vessel tree must be entered at its root."""
    parents = {i: set() for i in range(n_segments)}
    for p, c in connections:
        if p != c:
            parents[c].add(p)
    order, placed = [], set()
    while len(order) < n_segments:
        ready = [i for i in range(n_segments) if i not in placed and parents[i] <= placed]
        nxt = ready[0] if ready else min(i for i in range(n_segments) if i not in placed)  # cycle
        order.append(nxt)
        placed.add(nxt)
    return order


def canonical_connections(n_segments, connections, order):
    """Sort connections so that, at each branching, the child with the most downstream segments
    is traversed first. This makes the features independent of the order connections are listed."""
    children = {i: [] for i in range(n_segments)}
    for p, c in connections:
        children[p].append(c)

    def descendants(s, seen):
        for c in children[s]:
            if c not in seen:
                seen.add(c)
                descendants(c, seen)
        return seen

    size = {i: len(descendants(i, set())) for i in range(n_segments)}
    rank = {s: k for k, s in enumerate(order)}
    return sorted(connections, key=lambda pc: (rank[pc[0]], -size[pc[1]], rank[pc[1]]))


def build_graph(segments, connections, double_secondary_root_edges=False):
    """Build a PyG graph from centerline segments and their connectivity.

    segments: list of (N_i, 3) arrays, each ordered from proximal to distal. The first segment
        should be the root from which path distances are measured.
    connections: list of (parent_index, child_index); each links the last point of the parent to
        the first point of the child.
    double_secondary_root_edges: add every connection leaving the root of an additional vessel
        tree twice (coronary convention: the right coronary tree was built this way in training).
    Returns (data, order): graph segment k is input segment order[k], and `data.region_batch`
    assigns every node to its graph segment index.
    """
    connections = list(dict.fromkeys((p, c) for p, c in connections if p != c))
    if double_secondary_root_edges:
        has_parent = {c for _, c in connections}
        connections += [(p, c) for p, c in connections if p != 0 and p not in has_parent]
    order = topological_order(len(segments), connections)
    connections = canonical_connections(len(segments), connections, order)
    rank = {s: k for k, s in enumerate(order)}
    connections = [(rank[p], rank[c]) for p, c in connections]

    # Translate the case so that its bounding-box minimum is at the origin.
    origin = np.min(np.vstack(segments), axis=0)
    res = [resample_centerline(np.asarray(segments[s], dtype=np.float64) - origin) for s in order]
    counts = [len(r) for r in res]
    offsets = np.concatenate([[0], np.cumsum(counts)[:-1]])
    points = np.vstack(res)
    n = len(points)

    edges = []
    for off, cnt in zip(offsets, counts):
        for i in range(cnt - 1):
            edges += [[off + i, off + i + 1], [off + i + 1, off + i]]
    for p, c in connections:
        a, b = offsets[p] + counts[p] - 1, offsets[c]
        edges += [[a, b], [b, a]]

    neighbours = [[] for _ in range(n)]
    for a, b in edges:
        neighbours[a].append(b)

    dist, level = _dfs_distance_level(n, neighbours)
    angle = _mean_edge_angles(points, neighbours)
    orient = np.vstack([np.repeat((r[-1] - r[0])[None], len(r), 0) for r in res])

    x = np.column_stack([points, dist, level, angle, orient]).astype(np.float32)
    data = Data(x=torch.from_numpy(x),
                edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(),
                region_batch=torch.from_numpy(np.repeat(np.arange(len(res)), counts)).long())
    return data, order
