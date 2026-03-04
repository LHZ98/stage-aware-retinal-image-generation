"""
Arc length and path decomposition along full vessel paths.
- Build graph from segment endpoints; find endpoints (degree=1); DFS from each endpoint to another to get paths.
- Each path = sequence of (seg_id, direction); assign path_id and path arc-length start s_start per segment.
"""

import numpy as np
from typing import List, Tuple, Dict

from . import metrics


def _node_key(rc: Tuple[int, int]) -> Tuple[int, int]:
    return rc


def build_segment_graph(
    segments: List[List[Tuple[Tuple[int, int], float]]],
) -> Tuple[Dict[Tuple[int, int], List[Tuple[int, int]]], List[float]]:
    """
    node -> list of (seg_id, end_which): end_which 0 = this node is seg[0], 1 = seg[-1].
    Returns (adj: node -> [(seg_id, end_which), ...], seg_lengths).
    """
    adj: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    seg_lengths = []
    for seg_id, seg in enumerate(segments):
        if len(seg) < 2:
            seg_lengths.append(0.0)
            continue
        L = metrics.segment_arc_length(seg)
        seg_lengths.append(L)
        end0 = seg[0][0]
        end1 = seg[-1][0]
        for node, ew in [(end0, 0), (end1, 1)]:
            key = _node_key(node)
            if key not in adj:
                adj[key] = []
            adj[key].append((seg_id, ew))
    return adj, seg_lengths


def get_endpoints(adj: Dict[Tuple[int, int], List[Tuple[int, int]]]) -> List[Tuple[int, int]]:
    """Endpoints = nodes with exactly one incident segment."""
    return [node for node, ne in adj.items() if len(ne) == 1]


def dfs_path(
    start_node: Tuple[int, int],
    adj: Dict[Tuple[int, int], List[Tuple[int, int]]],
    seg_lengths: List[float],
) -> Tuple[List[Tuple[int, int]], float]:
    """
    From start_node (endpoint), DFS along segments until we hit another endpoint.
    Returns (path as list of (seg_id, direction), total_length).
    direction: 0 = segment order as stored, 1 = reverse (so we walk from start to end of path).
    """
    path: List[Tuple[int, int]] = []
    total_len = 0.0
    visited_seg = set()
    cur = start_node
    cur_key = _node_key(cur)
    if cur_key not in adj or len(adj[cur_key]) != 1:
        return path, total_len
    while True:
        neighbors = adj[cur_key]
        next_seg = None
        for (seg_id, end_which) in neighbors:
            if seg_id in visited_seg:
                continue
            next_seg = (seg_id, end_which)
            break
        if next_seg is None:
            break
        seg_id, end_which = next_seg
        visited_seg.add(seg_id)
        L = seg_lengths[seg_id] if seg_id < len(seg_lengths) else 0.0
        total_len += L
        # direction: walk from cur toward the other end. If cur is end 0, walk forward -> 0; if end 1, backward -> 1.
        direction = 0 if end_which == 0 else 1
        path.append((seg_id, direction))
        # move to other end of segment - we need segments list to get the other node; we don't have it here. So we need to pass segments to get the other endpoint. Refactor: return segment list and we look up.
        break
    # We need segments to get the other end. So change signature to pass segments and use seg_id to get next node.
    return path, total_len


def build_paths(
    segments: List[List[Tuple[Tuple[int, int], float]]],
) -> Tuple[List[List[Tuple[int, int]]], List[float], Dict[int, Tuple[int, float, float]]]:
    """
    Build root-to-leaf style paths. Each path = list of (seg_id, direction).
    direction 0 = segment from first to last point; 1 = reverse.
    Returns (paths, path_L_totals, seg_to_path: seg_id -> (path_id, s_start, L_path)).
    """
    adj, seg_lengths = build_segment_graph(segments)
    endpoints = get_endpoints(adj)
    if not endpoints:
        return [], [], {}

    paths: List[List[Tuple[int, int]]] = []
    path_L_totals: List[float] = []
    seg_to_path: Dict[int, Tuple[int, float, float]] = {}

    def other_end(seg_id: int, this_end: Tuple[int, int]) -> Tuple[int, int]:
        seg = segments[seg_id]
        e0, e1 = seg[0][0], seg[-1][0]
        return e1 if this_end == e0 else e0

    for start_node in endpoints:
        path: List[Tuple[int, int]] = []
        total_len = 0.0
        cur = start_node
        cur_key = _node_key(cur)
        visited_seg = set()
        while True:
            if cur_key not in adj:
                break
            neighbors = adj[cur_key]
            next_item = None
            for (seg_id, end_which) in neighbors:
                if seg_id in visited_seg:
                    continue
                next_item = (seg_id, end_which)
                break
            if next_item is None:
                break
            seg_id, end_which = next_item
            visited_seg.add(seg_id)
            L = seg_lengths[seg_id] if seg_id < len(seg_lengths) else 0.0
            direction = 0 if end_which == 0 else 1
            path.append((seg_id, direction))
            total_len += L
            cur = other_end(seg_id, cur)
            cur_key = _node_key(cur)
            if cur_key not in adj or len(adj[cur_key]) == 0:
                break
        if not path:
            continue
        path_id = len(paths)
        paths.append(path)
        path_L_totals.append(total_len)
        s_start = 0.0
        for (seg_id, _) in path:
            if seg_id not in seg_to_path:
                seg_to_path[seg_id] = (path_id, s_start, total_len)
            s_start += seg_lengths[seg_id] if seg_id < len(seg_lengths) else 0.0

    return paths, path_L_totals, seg_to_path
