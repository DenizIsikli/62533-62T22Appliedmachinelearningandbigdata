# Name & Study Nr.
# Deniz Isikli s215818
# Mark Nielsen s204434

from DatasetDownloader.download_dataset import DatasetDownloader
from _1data_processing import DataProcessing
from _2baseline_kmeans_numpy import BaselineKmeansNumpy
from _3task_model_sklearn import TaskModelSklearn
from _4visualization_and_evaluation import VisualizationAndEvaluation

def main():
    """Main function to run the entire project pipeline."""
    # Step 1: Download the dataset
    DatasetDownloader()

    # Step 2: Process the data and calculate RFM features
    DataProcessing()

    # Step 3: Run baseline KMeans clustering using NumPy
    BaselineKmeansNumpy()

    # Step 4: Run task model clustering using scikit-learn
    TaskModelSklearn()

    # Step 5: Visualize and evaluate the clustering results
    VisualizationAndEvaluation()
    

if __name__ == "__main__":
    main()
