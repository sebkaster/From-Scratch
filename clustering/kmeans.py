import numpy as np
from typing import Optional, Tuple
from scipy.spatial import KDTree
from sklearn.metrics import silhouette_score as sklearn_silhouette_score

# global variables
epsilon = 1e-5


def randomly_selected_points(data: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    Distinct points are randomly selected as initial centers
    :param data: raw data
    :param n_clusters: desired number of cluster
    :return: initialized clusters
    """
    indices = np.random.choice(data.shape[0], n_clusters, replace=False)
    return data[indices]


def centroids_of_random_sub_samples(data: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    The data is partitioned into k random sub samples of the same size.
    The initial clusters are the mean of these sub samples.
    :param data: raw data
    :param n_clusters: desired number of cluster
    :return: initialized clusters
    """
    sub_samples = np.array_split(data, n_clusters)
    return np.asarray([np.mean(sub_sample, axis=0) for sub_sample in sub_samples])


def simple_farthest_point(data: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    The first center is selected randomly.
    The remaining centers are selected as points maximally distant from the nearest cluster.
    :param data: raw data
    :param n_clusters: desired number of cluster
    :return: initialized clusters
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
    The first center is selected randomly.
    The remaining centers are selected randomly with a probability of the squared distance from the nearest cluster.
    :param data: raw data
    :param n_clusters: desired number of cluster
    :return: initialized clusters
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
    """
    Initialize cluster by the specified method.
    :param data: raw data
    :param n_clusters: desired number of clusters
    :param method: method used for initialization
    :return: initialized clusters
    """
    switch_case = {
        'RP': randomly_selected_points,
        'RGC': centroids_of_random_sub_samples,
        'KMPP': kmeans_plus_plus,
        'SIMFP': simple_farthest_point
    }
    return switch_case[method](data, n_clusters)


def mean_silhouette_score(data: np.ndarray, clusters: np.ndarray) -> float:
    """
    The silhouette score is a measure of how similar an object is to its own cluster compared to other clusters.
    This function returns the mean silhouette score of all data points
    :param data: raw data
    :param clusters: trained clusters
    :return: mean silhouette score
    """
    cluster_kd_tree = KDTree(clusters)
    _, labels = cluster_kd_tree.query(data)

    scores = list()
    for cluster_idx in range(len(clusters)):

        cluster_points = np.take(data, np.where(labels == cluster_idx)[0], axis=0)

        for i in range(cluster_points.shape[0]):
            sum_dist_inner = 0.0

            for j in range(cluster_points.shape[0]):
                if i == j:
                    continue
                sum_dist_inner += np.linalg.norm(cluster_points[i] - cluster_points[j])

            mean_dist_inner = 0
            if cluster_points.shape[0] - 1 > 0:
                mean_dist_inner = sum_dist_inner / (cluster_points.shape[0] - 1)

            min_mean_dist_outer = np.inf

            for other_cluster_idx in range(len(clusters)):
                current_sum = 0.0
                if other_cluster_idx == cluster_idx:
                    continue
                other_cluster_points = np.take(data, np.where(labels == other_cluster_idx)[0], axis=0)

                for j in range(other_cluster_points.shape[0]):
                    current_sum += np.linalg.norm(cluster_points[i] - other_cluster_points[j])
                mean_of_elems = current_sum / other_cluster_points.shape[0]
                min_mean_dist_outer = min(mean_of_elems, min_mean_dist_outer)

            if mean_dist_inner < min_mean_dist_outer:
                scores.append(1.0 - (mean_dist_inner / min_mean_dist_outer))
            elif mean_dist_inner > min_mean_dist_outer:
                scores.append((min_mean_dist_outer / mean_dist_inner) - 1.0)
            else:
                scores.append(0.0)

    mean_score = float(np.mean(scores))
    assert (sklearn_silhouette_score(data, labels) - mean_score < epsilon)

    return mean_score


def train_kmeans(data: np.ndarray, max_clusters: int, n_epochs: int, verbose: Optional[bool] = True) -> Tuple[
    np.ndarray, float]:
    """
    Selects the optimal number of clusters based on the silhouette score and returns the trained model.
    :param data: raw data
    :param max_clusters: upper limit for number of clusters
    :param n_epochs: number epochs to train model
    :param verbose: print logging information
    :return: trained clusters, silhouette score
    """
    best_silhouette_score, best_centroids = None, None
    for n_clusters in range(2, max_clusters+1):
        centroids = initialize_clusters(data, n_clusters, method='KMPP')
        for epoch in range(n_epochs):
            if verbose:
                print('epoch', epoch, n_clusters)
            cluster_kd_tree = KDTree(centroids)
            data_centroids_mapping = [{'coords_sum': 0.0, 'n_points': 0} for _ in range(n_clusters)]
            for point in data:
                _, centroid_idx = cluster_kd_tree.query(point)

                data_centroids_mapping[centroid_idx]['coords_sum'] += point
                data_centroids_mapping[centroid_idx]['n_points'] += 1

            for i in range(centroids.shape[0]):
                if data_centroids_mapping[i]['n_points'] > 0:
                    centroids[i] = data_centroids_mapping[i]['coords_sum'] / data_centroids_mapping[i]['n_points']
            silhouette_score = mean_silhouette_score(data, centroids)
            if best_silhouette_score is None or silhouette_score > best_silhouette_score:
                best_silhouette_score = silhouette_score
                best_centroids = centroids

    return best_centroids, best_silhouette_score
