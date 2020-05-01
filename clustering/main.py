import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional
from sklearn import datasets
import imageio
import matplotlib.colors as colors
from matplotlib.patches import Ellipse
from gmm import train_gmm
from kmeans import train_kmeans
from dbscan import train_dbscan
from scipy.spatial import KDTree


def generate_data(iris_dataset: Optional[bool] = False) -> np.ndarray:
    if iris_dataset:
        iris = datasets.load_iris()
        x = iris.data
    else:
        x = -2 * np.random.rand(100, 2)
        temp = 1 + 2 * np.random.rand(50, 2)
        x[50:100, :] = temp
    return x


def create_cluster_animation_gmm(data_: np.ndarray, history_: List, scores_: np.ndarray) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    colorset = ['blue', 'red', 'black']
    images = []

    for j, clusters_ in enumerate(history_):

        idx = 0

        if j % 3 != 0:
            continue

        plt.cla()

        for cluster in clusters_:
            mu = cluster['mu_k']
            cov = cluster['cov_k']

            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            order = eigenvalues.argsort()[::-1]
            eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
            vx, vy = eigenvectors[:, 0][0], eigenvectors[:, 0][1]
            theta = np.arctan2(vy, vx)

            color = colors.to_rgba(colorset[idx])

            for cov_factor in range(1, 4):
                ell = Ellipse(xy=mu, width=np.sqrt(eigenvalues[0]) * cov_factor * 2,
                              height=np.sqrt(eigenvalues[1]) * cov_factor * 2, angle=float(np.degrees(theta)),
                              linewidth=2)
                ell.set_facecolor((color[0], color[1], color[2], 1.0 / (cov_factor * 4.5)))
                ax.add_artist(ell)

            ax.scatter(cluster['mu_k'][0], cluster['mu_k'][1], c=colorset[idx], s=1000, marker='+')
            idx += 1

        for i in range(data_.shape[0]):
            ax.scatter(data_[i, 0], data_[i, 1], c=colorset[int(np.argmax(scores_[i]))], marker='o')

        fig.canvas.draw()

        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        images.append(image)

    imageio.mimsave('./gmm.gif', images, fps=1)
    plt.show()


def plot_kmeans(data: np.ndarray, centroids: np.ndarray) -> None:
    cluster_kd_tree = KDTree(centroids)
    _, labels = cluster_kd_tree.query(data)

    plt.scatter(centroids[:, 0], centroids[:, 1])
    plt.scatter(data[:, 0], data[:, 1], c=labels,
                s=50, cmap='viridis')
    plt.show()


def plot_dbscan(data: np.ndarray, labels: np.ndarray) -> None:
    plt.scatter(data[:, 0], data[:, 1], c=labels,
                s=50, cmap='viridis')
    plt.show()


if __name__ == "__main__":
    data = generate_data(True)

    #res, labels = train_dbscan(data)
    #plot_dbscan(data, labels)
    # max_clusters = 20
    n_epochs = 50
    # centroids, score = train_kmeans(data, max_clusters, n_epochs)
    # plot_kmeans(data, centroids)
    clusters, likelihoods, scores, sample_likelihoods, history = train_gmm(data, 3, 50)
    create_cluster_animation_gmm(data, history, scores)
