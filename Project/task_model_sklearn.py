import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

def run_kmeans(X, n_clusters=4):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(X)
    return labels, model

def run_dbscan(X, eps=0.5, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    return labels, model

if __name__ == "__main__":
    df = pd.read_csv("rfm_data.csv", index_col=0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    # KMeans
    kmeans_labels, _ = run_kmeans(X_scaled, n_clusters=4)
    df['KMeansCluster'] = kmeans_labels

    # DBSCAN
    dbscan_labels, _ = run_dbscan(X_scaled, eps=1.5, min_samples=10)
    df['DBSCANCluster'] = dbscan_labels

    df.to_csv("task_model_clusters.csv")
    print("Saved sklearn cluster results to task_model_clusters.csv")
