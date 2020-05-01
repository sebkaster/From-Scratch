import numpy as np
from scipy.spatial import KDTree
from typing import List, Optional, Tuple
from queue import LifoQueue


def expand_cluster(next_cluster: List[int], cluster_idx: int, data_kd_tree: KDTree, data: np.ndarray,
                   neighbor_idxs: List[int], visited: List[bool], status: List[Optional[int]], eps: float,
                   min_pts: int) -> None:
    """
    Find the connected components of core points on the neighbor graph, ignoring all non-core points.
    Assign each non-core point to a nearby cluster if the cluster is an ε (eps) neighbor, otherwise assign it to noise (-1).
    :param next_cluster: current indices list
    :param cluster_idx: cluster index
    :param data_kd_tree: KD-Tree of whole data
    :param data: raw data
    :param neighbor_idxs: neighbors in range epsilon of initial points
    :param visited: list which keeps track of already processed points
    :param status: list which keeps track to which cluster a point belongs to
    :param eps: radius of a neighborhood with respect to some point
    :param min_pts: minimum number of neighbors for a point to be considered as a core instance
    """
    stack = LifoQueue()
    stack.queue = neighbor_idxs
    while not stack.empty():
        current_idx = stack.get()
        if not visited[current_idx]:
            visited[current_idx] = True
            _, neighbor_idxs = data_kd_tree.query(x=data[current_idx], k=data.shape[0], distance_upper_bound=eps)
            cleaned_neighbor_idxs = [idx for idx in neighbor_idxs if idx < data.shape[0] and idx != current_idx]
            if len(cleaned_neighbor_idxs) >= min_pts:
                stack.queue.extend(cleaned_neighbor_idxs)
        if status[current_idx] == -1:
            status[current_idx] = cluster_idx
            next_cluster.append(current_idx)


def train_dbscan(data: np.ndarray, eps: float = 0.5, min_pts: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Density-based spatial clustering of applications with noise (DBSCAN)
    :param data: raw data
    :param eps: radius of a neighborhood with respect to some point
    :param min_pts: minimum number of neighbors for a point to be considered as a core instance
    :return: clusters with indices, labels
    """
    clusters = list()
    visited = [False] * data.shape[0]
    status: List[int] = [-1] * data.shape[0]
    data_kd_tree = KDTree(data)
    for i in range(data.shape[0]):
        if not visited[i]:
            visited[i] = True
            _, neighbor_idxs = data_kd_tree.query(x=data[i], k=data.shape[0], distance_upper_bound=eps)
            cleaned_neighbor_idxs = [idx for idx in neighbor_idxs if idx < data.shape[0] and idx != i]
            if len(cleaned_neighbor_idxs) >= min_pts:
                status[i] = len(clusters)
                next_cluster = [i]
                expand_cluster(next_cluster, status[i], data_kd_tree, data, cleaned_neighbor_idxs, visited, status, eps,
                               min_pts)
                clusters.append(next_cluster)

    return np.asarray(clusters), np.asarray(status)
