# Name & Study Nr.
# Deniz Isikli s215818
# Mark Nielsen s204434

import os
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from Util.util import Util
import matplotlib.pyplot as plt

class DataProcessing():
    def __init__(self):
        """Run the data processing and RFM calculation."""
        steps = ["Loading & Cleaning Data", "Calculating RM", "Saving RFM CSV"]
        progress = tqdm(total=len(steps), desc="Data Processing", ncols=80) 

        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, "DatasetDownloader", "OnlineRetail.xlsx")
        os.makedirs(os.path.join(script_dir, "Results", "DataProcessing"), exist_ok=True)
        results_dir = os.path.join(script_dir, "Results", "DataProcessing")

        Util.remove_folder_content(results_dir)

        progress.set_postfix_str(steps[0])
        df = self.load_and_clean_data(file_path)
        progress.update(1)

        progress.set_postfix_str(steps[1])
        rfm_df = self.calculate_rfm(df, reference_date=pd.to_datetime("2011-12-10"))
        progress.update(1)

        progress.set_postfix_str(steps[2])
        rfm_df.to_csv(os.path.join(results_dir, "rfm_data.csv"))
        self.plot_rfm(rfm_df)
        progress.update(1)

        progress.close()
        print("Saved RFM features to rfm_data.csv\n\n")

    def load_and_clean_data(self, file_path):
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

    def calculate_rfm(self, df, reference_date):
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
    
    # plot the rfm data and save as png to the Results/DataProcessing folder
    def plot_rfm(self, rfm_df):
        """Plot the RFM data and save as a PNG file."""
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=rfm_df, x='Recency', y='Monetary', hue='Frequency', palette='viridis')
        plt.title('RFM Analysis')
        plt.xlabel('Recency (days since last purchase)')
        plt.ylabel('Monetary Value (total spending in GBP)')
        plt.legend(title='Frequency (number of unique invoices)')
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(script_dir, "Results", "DataProcessing")
        plt.savefig(os.path.join(results_dir, "rfm_plot.png"))
        plt.close()
