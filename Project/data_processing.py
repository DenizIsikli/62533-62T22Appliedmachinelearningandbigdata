import pandas as pd
import numpy as np

def load_and_clean_data(file_path):
    df = pd.read_excel(file_path)
    df.dropna(inplace=True)
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    return df

def calculate_rfm(df, reference_date):
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (reference_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum'
    })

    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    rfm = rfm[rfm['Monetary'] > 0]  # Remove customers with 0 spending
    return rfm

if __name__ == "__main__":
    df = load_and_clean_data("OnlineRetail.xlsx")
    rfm_df = calculate_rfm(df, reference_date=pd.to_datetime("2011-12-10"))
    rfm_df.to_csv("rfm_data.csv")
    print("RFM features saved to rfm_data.csv")
