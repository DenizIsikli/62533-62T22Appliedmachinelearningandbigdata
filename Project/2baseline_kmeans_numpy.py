import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def normalize(X):
    """Normalize the dataset using z-score normalization.
    
    Args:
        X (ndarray): The input data to be normalized.
    
    Returns:
        ndarray: The normalized data.
    """
    return (X - X.mean(axis=0)) / X.std(axis=0)

def initialize_centroids(X, k):
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

def compute_distances(X, centroids):
    """Compute the distance between each point and each centroid.

    Args:
        X (ndarray): The input data.
        centroids (ndarray): The centroids.

    Returns:
        ndarray: A matrix of distances between each point and each centroid.
    """
    return np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)

def assign_clusters(distances):
    """Assign each point to the nearest centroid.

    Args:
        distances (ndarray): A matrix of distances between each point and each centroid.

    Returns:
        ndarray: An array of cluster labels for each point.
    """
    return np.argmin(distances, axis=1)

def update_centroids(X, labels, k):
    """Update the centroids based on the current cluster assignments.

    Args:
        X (ndarray): The input data.
        labels (ndarray): The current cluster assignments.
        k (int): The number of clusters.

    Returns:
        ndarray: The updated centroids.
    """
    return np.array([X[labels == i].mean(axis=0) for i in range(k)])

def kmeans(X, k, max_iters=100):
    """KMeans clustering algorithm.

    Args:
        X (ndarray): The input data.
        k (int): The number of clusters.
        max_iters (int): The maximum number of iterations.

    Returns:
        tuple: A tuple containing the cluster labels and the final centroids.
    """
    centroids = initialize_centroids(X, k)
    for i in range(max_iters):
        distances = compute_distances(X, centroids)
        labels = assign_clusters(distances)
        new_centroids = update_centroids(X, labels, k)
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return labels, centroids

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "Results", "rfm_data.csv")
    results_dir = os.path.join(script_dir, "Results")

    df = pd.read_csv(file_path, index_col=0)
    X = normalize(df.values)
    k = 4
    labels, centroids = kmeans(X, k)

    df['Cluster'] = labels
    df.to_csv(os.path.join(results_dir, "baseline_clusters.csv"), index=False)
    print("Saved cluster assignments to baseline_clusters.csv")
