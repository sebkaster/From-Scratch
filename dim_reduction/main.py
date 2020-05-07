from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt


def load_dataset() -> np.ndarray:
    """
    :return: iris dataset with 4 features
    """
    X_ = load_iris()['data']
    return X_


def center_data(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Center data to zero mean.
    :param X: raw data
    :return: centered data
    """
    # X_normalized_ = StandardScaler(with_std=False).fit_transform(X)
    X_normalized_ = X - X.mean(axis=0)
    return X_normalized_, X.mean(axis=0)


def calc_covariance_mat(X_normalized_: np.ndarray) -> np.ndarray:
    """
    The covariance matrix of two variavles measures how correlated they are.
    :param X: normalized data
    :return: covariance matrix
    """

    mean_vec = np.mean(X_normalized_, axis=0)
    cov_mat_ = (X - mean_vec).T.dot((X - mean_vec)) / (X.shape[0] - 1)

    return cov_mat_


def calc_eigen_pairs(cov_mat_: np.ndarray) -> np.ndarray:
    """
    Calculate eigenvectors and eigenvalues of covariance matrix.
    :param cov_mat_: covariance matrix
    :return: pairs of eigenvalues and corresponding eigenvectors; sorted with respect to eigenvalue in descending order
    """
    eig_values, eig_vectors = np.linalg.eig(cov_mat_)

    eig_pairs_ = list(zip(eig_values, eig_vectors))
    eig_pairs_.sort(key=lambda x: x[0], reverse=True)

    return np.asarray(eig_pairs_)


def reduce_dim_of_data(X_normalized_: np.ndarray, eig_pairs_: np.ndarray, threshold: float) -> Tuple[np.ndarray,
                                                                                                     np.ndarray]:
    """
    :param X_normalized_: normalized data
    :param eig_pairs_: eigenvalue-eigenvector pairs of covariance matrix in descending order with respect to eigenvalue
    :param threshold: threshold for preserving variance of the data
    :return: dimensionality reduced dataset
    """
    reduced_dim_size = np.argmax(np.cumsum(eig_pairs_[:, 0] / sum(eig_pairs_[:, 0])) > threshold) + 1
    proj_mat_ = np.column_stack(eig_pairs_[:reduced_dim_size, 1])

    pca_data = X_normalized_.dot(proj_mat_)

    return pca_data, proj_mat_


if __name__ == "__main__":
    X = load_dataset()
    X_normalized, X_mean = center_data(X)

    cov_mat = calc_covariance_mat(X_normalized)
    eig_pairs = calc_eigen_pairs(cov_mat)
    X_reduced, proj_mat = reduce_dim_of_data(X_normalized, eig_pairs, 0.99)

    X_reconstructed = X_reduced.dot(proj_mat.T) + X_mean

    plt.scatter(X[:, 0], X[:, 2])
    plt.scatter(X_reconstructed[:, 0], X_reconstructed[:, 2])
    plt.show()
