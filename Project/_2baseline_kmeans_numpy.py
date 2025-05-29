# Name & Study Nr.
# Deniz Isikli s215818
# Mark Nielsen s204434

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from Util.util import Util
import matplotlib.pyplot as plt

class BaselineKmeansNumpy():
    def __init__(self):
        """Run the KMeans clustering algorithm on the dataset."""
        steps = ["Loading Data", "Running KMeans", "Saving Results"]
        progress = tqdm(total=len(steps), desc="KMeans Clustering", ncols=80)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, "Results", "DataProcessing", "rfm_data.csv")
        os.makedirs(os.path.join(script_dir, "Results", "BaselineKMeansNumpy"), exist_ok=True)
        results_dir = os.path.join(script_dir, "Results", "BaselineKMeansNumpy")

        Util.remove_folder_content(results_dir)

        progress.set_postfix_str(steps[0])
        df = pd.read_csv(file_path, index_col=0)
        progress.update(1)

        X = self.normalize(df.values)
        k = 4

        progress.set_postfix_str(steps[1])
        labels, centroids = self.kmeans(X, k)
        self.plot_clusters(X, labels, centroids)
        progress.update(1)

        df['Cluster'] = labels
        df.to_csv(os.path.join(results_dir, "baseline_clusters.csv"), index=False)

        progress.set_postfix_str(steps[2])
        progress.update(1)
        
        progress.close()
        print("Saved cluster assignments to baseline_clusters.csv\n\n")

    def normalize(self, X):
        """Normalize the dataset using z-score normalization.
        
        Args:
            X (ndarray): The input data to be normalized.
        
        Returns:
            ndarray: The normalized data.
        """
        return (X - X.mean(axis=0)) / X.std(axis=0)

    def initialize_centroids(self, X, k):
        """Initialize centroids for KMeans clustering.
        
        Args:
            X (ndarray): The input data.
            k (int): The number of clusters.

        Returns:
            ndarray: Randomly selected centroids from the dataset.
        """
        np.random.seed(42)
        indices = np.random.choice(X.shape[0], k, replace=False)
        return X[indices]

    def compute_distances(self, X, centroids):
        """Compute the distance between each point and each centroid.

        Args:
            X (ndarray): The input data.
            centroids (ndarray): The centroids.

        Returns:
            ndarray: A matrix of distances between each point and each centroid.
        """
        return np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)

    def assign_clusters(self, distances):
        """Assign each point to the nearest centroid.

        Args:
            distances (ndarray): A matrix of distances between each point and each centroid.

        Returns:
            ndarray: An array of cluster labels for each point.
        """
        return np.argmin(distances, axis=1)

    def update_centroids(self, X, labels, k):
        """Update the centroids based on the current cluster assignments.

        Args:
            X (ndarray): The input data.
            labels (ndarray): The current cluster assignments.
            k (int): The number of clusters.

        Returns:
            ndarray: The updated centroids.
        """
        return np.array([X[labels == i].mean(axis=0) for i in range(k)])

    def kmeans(self, X, k, max_iters=100):
        """KMeans clustering algorithm.

        Args:
            X (ndarray): The input data.
            k (int): The number of clusters.
            max_iters (int): The maximum number of iterations.

        Returns:
            tuple: A tuple containing the cluster labels and the final centroids.
        """
        centroids = self.initialize_centroids(X, k)
        for i in range(max_iters):
            distances = self.compute_distances(X, centroids)
            labels = self.assign_clusters(distances)
            new_centroids = self.update_centroids(X, labels, k)
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids
        return labels, centroids

    def plot_clusters(self, X, labels, centroids):
        """Plot the clusters and centroids.

        Args:
            X (ndarray): The input data.
            labels (ndarray): The cluster labels.
            centroids (ndarray): The centroids of the clusters.
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', s=50, label='Data Points')
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, label='Centroids')
        plt.title('KMeans Clustering')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.savefig(os.path.join("Results", "BaselineKMeansNumpy", "baselinekmeansnumpy.png"))
        plt.close()
