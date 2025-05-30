# Name & Study Nr.
# Deniz Isikli s215818
# Mark Nielsen s204434

import os
import pandas as pd
from tqdm import tqdm
from Util.util import Util
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

class TaskModelSklearn():
    def __init__(self):
        """Run the clustering algorithms on the dataset and save the results."""
        steps = ["Loading & Scaling Data", "Running KMeans", "Running DBSCAN", "Saving Results"]
        progress = tqdm(total=len(steps), desc="Task Model (sklearn)", ncols=80)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, "Results", "DataProcessing", "rfm_data.csv")
        os.makedirs(os.path.join(script_dir, "Results", "TaskModelSklearn"), exist_ok=True)
        results_dir = os.path.join(script_dir, "Results", "TaskModelSklearn")

        Util.remove_folder_content(results_dir)

        progress.set_postfix_str(steps[0])
        df = pd.read_csv(file_path, index_col=0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df)
        progress.update(1)

        progress.set_postfix_str(steps[1])
        kmeans_labels, kmeans_model = self.run_kmeans(X_scaled, n_clusters=4)
        self.plot_clusters(X_scaled, kmeans_labels, centroids=kmeans_model.cluster_centers_, iterations=kmeans_model.n_iter_)
        df['KMeansCluster'] = kmeans_labels
        progress.update(1)

        progress.set_postfix_str(steps[2])
        dbscan_labels, _ = self.run_dbscan(X_scaled, eps=1.5, min_samples=10)
        df['DBSCANCluster'] = dbscan_labels
        progress.update(1)

        progress.set_postfix_str(steps[3])
        df.to_csv(os.path.join(results_dir, "task_model_clusters.csv"), index=False)
        progress.update(1)

        progress.close()
        print("Saved sklearn cluster results to task_model_clusters.csv\n\n")

    def run_kmeans(self, X, n_clusters=4):
        """Run KMeans clustering on the data.

        Args:
            X (ndarray): The input data.
            n_clusters (int): The number of clusters.

        Returns:
            labels (ndarray): The cluster labels for each point.
            model (KMeans): The fitted KMeans model.
        """
        model = KMeans(n_clusters=n_clusters, random_state=42)
        labels = model.fit_predict(X)
        return labels, model

    def run_dbscan(self, X, eps, min_samples):
        """Run DBSCAN clustering on the data.

        Args:
            X (ndarray): The input data.
            eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
            min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.

        Returns:
            labels (ndarray): The cluster labels for each point.
            model (DBSCAN): The fitted DBSCAN model.
        """
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X)
        return labels, model

    def plot_clusters(self, X, labels, centroids=None, iterations=None):
        title = "Clusters Visualization"

        plt.figure(figsize=(10, 6))
        plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', s=50, alpha=0.6)
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, label='Centroids') if centroids is not None else None
        plt.title(title)
        if iterations is not None:
            title += f" (Converged in {iterations} iterations)"
            plt.title(title)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.savefig(os.path.join("Results", "TaskModelSklearn", "taskmodel_clusters_visualization.png"))
        plt.close()
        