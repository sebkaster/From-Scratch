import numpy as np
from typing import Optional
from scipy.spatial import KDTree


def randomly_selected_points(data: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    :param data:
    :param n_clusters:
    :return:
    """
    indices = np.random.choice(data.shape[0], n_clusters, replace=False)
    return data[indices]


def centroids_of_random_sub_samples(data: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    :param data:
    :param n_clusters:
    :return:
    """
    sub_samples = np.array_split(data, n_clusters)
    return np.asarray([np.mean(sub_sample, axis=0) for sub_sample in sub_samples])


def simple_farthest_point(data: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    :param data:
    :param n_clusters:
    :return:
    """
    clusters = [data[np.random.choice(data.shape[0])].tolist()]

    for _ in range(n_clusters - 1):
        next_idx = None
        max_sum = None
        for i in range(data.shape[0]):
            sum_dist = np.sum(np.linalg.norm([cluster - data[i] for cluster in clusters]))
            if (next_idx or max_sum) is None or sum_dist > max_sum:
                next_idx, max_sum = i, sum_dist
        clusters.append(data[next_idx])

    return np.asarray(clusters)


def kmeans_plus_plus(data: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    :param data: dataset
    :param n_clusters: number of clusters
    :return: clusters
    """
    clusters = [data[np.random.choice(data.shape[0])].tolist()]

    for _ in range(n_clusters - 1):
        cluster_kd_tree = KDTree(clusters)
        min_distances, _ = cluster_kd_tree.query(data)
        weights_squared = np.square(min_distances)
        weights_normalized = weights_squared / np.sum(weights_squared)
        idx = np.random.choice(data.shape[0], p=weights_normalized)
        clusters.append(data[idx])
    return np.asarray(clusters)


def initialize_clusters(data: np.ndarray, n_clusters: int, method: Optional[str] = 'RP') -> np.ndarray:
    switch_case = {
        'RP': randomly_selected_points,
        'RGC': centroids_of_random_sub_samples,
        'KMPP': kmeans_plus_plus,
        'SIMFP': simple_farthest_point
    }
    return switch_case[method](data, n_clusters)


def train_kmeans(data: np.ndarray, n_clusters: int, n_epochs: int, verbose: Optional[bool] = True) -> np.ndarray:
    centroids = initialize_clusters(data, n_clusters, method='RGC')

    for epoch in range(n_epochs):
        print('epoch', epoch)
        cluster_kd_tree = KDTree(centroids)
        data_centroids_mapping = [{'coords_sum': 0.0, 'n_points': 0} for _ in range(n_clusters)]
        for point in data:
            _, centroid_idx = cluster_kd_tree.query(point)

            data_centroids_mapping[centroid_idx]['coords_sum'] += point
            data_centroids_mapping[centroid_idx]['n_points'] += 1

        for i in range(centroids.shape[0]):
            centroids[i] = data_centroids_mapping[i]['coords_sum'] / data_centroids_mapping[i]['n_points']

    return centroids
