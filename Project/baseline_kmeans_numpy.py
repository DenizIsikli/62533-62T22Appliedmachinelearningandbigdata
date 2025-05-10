import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def normalize(X):
    return (X - X.mean(axis=0)) / X.std(axis=0)

def initialize_centroids(X, k):
    np.random.seed(42)
    indices = np.random.choice(X.shape[0], k, replace=False)
    return X[indices]

def compute_distances(X, centroids):
    return np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)

def assign_clusters(distances):
    return np.argmin(distances, axis=1)

def update_centroids(X, labels, k):
    return np.array([X[labels == i].mean(axis=0) for i in range(k)])

def kmeans(X, k, max_iters=100):
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
    df = pd.read_csv("rfm_data.csv", index_col=0)
    X = normalize(df.values)
    k = 4
    labels, centroids = kmeans(X, k)

    df['Cluster'] = labels
    df.to_csv("baseline_clusters.csv")
    print("Saved cluster assignments to baseline_clusters.csv")
