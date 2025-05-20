import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def plot_elbow(X):
    """Plot the elbow method to determine the optimal number of clusters for KMeans.

    Args:
        X (ndarray): The input data.

    Outputs:
        - A plot showing the inertia (WCSS) for different numbers of clusters.
    """
    distortions = []
    for k in range(1, 10):
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X)
        distortions.append(km.inertia_)
    plt.plot(range(1, 10), distortions, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia (WCSS)')
    plt.title('Elbow Method')
    plt.show()

def plot_silhouette(X, labels):
    """Plot the silhouette score for the clustering.

    Args:
        X (ndarray): The input data.
        labels (ndarray): The cluster labels.

    Outputs:
        - A plot showing the silhouette score for the clustering.
    """
    score = silhouette_score(X, labels)
    print(f'Silhouette Score: {score:.3f}')

def plot_pca(X, labels, title):
    """Plot the PCA components of the data.

    Args:
        X (ndarray): The input data.
        labels (ndarray): The cluster labels.
        title (str): The title for the plot.

    Outputs:
        - A scatter plot of the PCA components colored by cluster labels.
    """
    pca = PCA(n_components=2)
    components = pca.fit_transform(X)
    plt.scatter(components[:, 0], components[:, 1], c=labels, cmap='tab10', s=40)
    plt.title(title)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.show()

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "Results", "task_model_clusters.csv")
    results_dir = os.path.join(script_dir, "Results")

    df = pd.read_csv(file_path)
    features = ['Recency', 'Frequency', 'Monetary']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])

    # Elbow and Silhouette
    plot_elbow(X_scaled)
    plot_silhouette(X_scaled, df['KMeansCluster'])

    # PCA plots
    plot_pca(X_scaled, df['KMeansCluster'], "KMeans Clusters")
    plot_pca(X_scaled, df['DBSCANCluster'], "DBSCAN Clusters")
