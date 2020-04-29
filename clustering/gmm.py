import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import KMeans

# global variables
epsilon = 1e-5


def gaussian(X, mu, cov):
    n = X.shape[1]
    diff = (X - mu).T
    return np.diagonal(1 / ((2 * np.pi) ** (n / 2) * np.linalg.det(cov) ** 0.5) * np.exp(
        -0.5 * np.dot(np.dot(diff.T, np.linalg.inv(cov)), diff))).reshape(-1, 1)


def initialize_clusters(data_: np.ndarray, num_clusters: int) -> List[Dict]:
    clusters = list()

    kmeans = KMeans().fit(data_)
    mu_k = kmeans.cluster_centers_

    for i in range(num_clusters):
        clusters.append({
            'pi_k': 1.0 / num_clusters,
            'mu_k': mu_k[i],
            'cov_k': np.identity(data_.shape[1], dtype=np.float64)
        })

    return clusters


def expectation_step(data_: np.ndarray, clusters: List[Dict]) -> None:
    totals = np.zeros((data_.shape[0], 1), dtype=np.float64)

    for cluster in clusters:
        pi_k = cluster['pi_k']
        mu_k = cluster['mu_k']
        cov_k = cluster['cov_k']

        gamma_nk = (pi_k * gaussian(data_, mu_k, cov_k)).astype(np.float64)

        for i in range(data_.shape[0]):
            totals[i] += gamma_nk[i]

        cluster['gamma_nk'] = gamma_nk
        cluster['totals'] = totals

    for cluster in clusters:
        cluster['gamma_nk'] /= cluster['totals']


def maximization_step(data_: np.ndarray, cluster: List[Dict]) -> None:
    total_num = data_.shape[0]

    for cluster in cluster:
        gamma_nk = cluster['gamma_nk']
        cov_k = np.zeros((data_.shape[1], data_.shape[1]))

        N_k = np.sum(gamma_nk, axis=0)

        pi_k = N_k / float(total_num)
        mu_k = np.sum(gamma_nk * data_, axis=0) / N_k

        for j in range(data_.shape[0]):
            diff = (data_[j] - mu_k).reshape(-1, 1)
            cov_k += gamma_nk[j] * np.dot(diff, diff.T)

        cov_k /= N_k

        cluster['pi_k'] = pi_k
        cluster['mu_k'] = mu_k
        cluster['cov_k'] = cov_k


def get_likelihood(clusters_: List[Dict]) -> Tuple[np.ndarray, np.float64]:
    sample_likelihoods_ = np.log(np.array([cluster['totals'] for cluster in clusters_]))
    return np.sum(sample_likelihoods_), sample_likelihoods_


def train_gmm(data_: np.ndarray, n_clusters: int, n_epochs: int, verbose: Optional[bool] = True) -> Tuple[
    List, np.ndarray, np.ndarray, np.ndarray, List]:
    assert (n_epochs > 0)

    clusters = initialize_clusters(data_, n_clusters)
    likelihoods = np.zeros((n_epochs,))
    sample_likelihoods = np.ndarray([])
    scores = np.zeros((data_.shape[0], n_clusters))

    history = list()  # for plotting

    for i in range(n_epochs):

        # for plotting animation
        clusters_snapshot = list()
        for cluster in clusters:
            clusters_snapshot.append({
                'mu_k': cluster['mu_k'].copy(),
                'cov_k': cluster['cov_k'].copy()
            })
        history.append(clusters_snapshot)

        # EM-Steps
        expectation_step(data_, clusters)
        maximization_step(data_, clusters)

        likelihood, sample_likelihoods = get_likelihood(clusters)
        likelihoods[i] = likelihood

        if verbose:
            print('Epoch: ', i + 1, 'Likelihood: ', likelihood)

    for i, cluster in enumerate(clusters):
        scores[:, i] = np.log(cluster['gamma_nk'] + epsilon).reshape(-1)

    return clusters, likelihoods, scores, sample_likelihoods, history
