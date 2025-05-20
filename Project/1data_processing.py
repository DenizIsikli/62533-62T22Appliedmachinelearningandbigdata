import os
import pandas as pd
import numpy as np

def load_and_clean_data(file_path):
    """Load and clean the Online Retail dataset.
    This function reads the dataset from an Excel file, drops rows with missing values,
    filters out negative quantities and unit prices, and calculates the total price for each transaction.

    Args:
        file_path (str): The path to the Excel file containing the dataset.

    Outputs:
        - df: A cleaned DataFrame containing the relevant data.
    """
    df = pd.read_excel(file_path)
    df.dropna(inplace=True)
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    return df

def calculate_rfm(df, reference_date):
    """Calculate RFM (Recency, Frequency, Monetary) features for customer segmentation.
    This function groups the data by CustomerID and calculates the recency (days since last purchase),
    frequency (number of unique invoices), and monetary value (total spending) for each customer.

    Args:
        df (DataFrame): The cleaned DataFrame containing the dataset.
        reference_date (datetime): The date to calculate recency from.

    Outputs:
        - rfm: A DataFrame containing the RFM features for each customer.
    """
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (reference_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum'
    })

    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    rfm = rfm[rfm['Monetary'] > 0]  # Remove customers with 0 spending
    return rfm

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "DatasetDownloader", "OnlineRetail.xlsx")
    results_dir = os.path.join(script_dir, "Results")

    df = load_and_clean_data(file_path)
    rfm_df = calculate_rfm(df, reference_date=pd.to_datetime("2011-12-10"))
    rfm_df.to_csv(os.path.join(results_dir, "rfm_data.csv"))
    print("RFM features saved to rfm_data.csv")
