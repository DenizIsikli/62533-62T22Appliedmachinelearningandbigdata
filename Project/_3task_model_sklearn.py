import os
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

class TaskModelSklearn():
    def __init__(self):
        """Run the clustering algorithms on the dataset and save the results."""
        steps = ["Loading & Scaling Data", "Running KMeans", "Running DBSCAN", "Saving Results"]
        progress = tqdm(total=len(steps), desc="Task Model (sklearn)", ncols=80)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, "Results", "rfm_data.csv")
        results_dir = os.path.join(script_dir, "Results")

        progress.set_description(steps[0])
        df = pd.read_csv(file_path, index_col=0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df)
        progress.update(1)

        progress.set_description(steps[1])
        kmeans_labels, _ = self.run_kmeans(X_scaled, n_clusters=4)
        df['KMeansCluster'] = kmeans_labels
        progress.update(1)

        progress.set_description(steps[2])
        dbscan_labels, _ = self.run_dbscan(X_scaled, eps=1.5, min_samples=10)
        df['DBSCANCluster'] = dbscan_labels
        progress.update(1)

        progress.set_description(steps[3])
        df.to_csv(os.path.join(results_dir, "task_model_clusters.csv"), index=False)
        progress.update(1)

        progress.close()
        print("Saved sklearn cluster results to task_model_clusters.csv")

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

    def run_dbscan(self, X, eps=0.5, min_samples=5):
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
