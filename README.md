# 62533-62T22 Applied Machine Learning & Big Data

This repository contains the code and resources for the course **Applied Machine Learning & Big Data** (62533-62T22) at the **Technical University of Denmark (DTU)**.

## Structure

- **`Coursework/`** – Contains weekly exercises, experiments, and small assignments done throughout the course.
- **`Project/`** – Contains the full pipeline for the final project, focused on customer segmentation using unsupervised learning.

## Project Overview

The project performs customer segmentation using the **Online Retail dataset**, applying:
- Data cleaning and RFM feature engineering
- A baseline K-Means model implemented from scratch using NumPy
- Task models using `scikit-learn` (KMeans and DBSCAN)
- Visual evaluation with PCA, silhouette score, and the elbow method
- Auto-generated plots and PowerPoint slides

All logic is modularized into scripts such as:
- **`_1data_processing.py`** – Cleans the dataset and computes RFM features
- **`_2baseline_kmeans_numpy.py`** – Runs KMeans clustering using NumPy
- **`_3task_model_sklearn.py`** – Applies KMeans and DBSCAN using `scikit-learn`
- **`_4visualization_evaluation.py`** – Generates plots and saves them to a presentation
- **`main.py`** – Entry point to run the full pipeline
- **`Util/`** – Contains helper functions and dataset downloader

## Dataset

The retail dataset is ignored in Git due to its size, but it will be downloaded automatically when running the program. The dataset is a CSV file containing transactions from an online retail store.

The data is originally from the UC Irvine Machine Learning Repository:  
https://archive.ics.uci.edu/ml/datasets/online+retail

## Execution
Simply run the `main.py` script to execute the full pipeline: