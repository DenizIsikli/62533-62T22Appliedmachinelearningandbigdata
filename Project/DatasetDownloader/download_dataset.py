import requests
import os
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
destination_path = os.path.join(script_dir, "OnlineRetail.xlsx")

def download_online_retail_data(destination=destination_path):
    """This function downloads the Online Retail dataset from UCI Machine Learning Repository.
    The dataset is an Excel file containing transactional data from a UK-based online retailer.

    Args:
        destination (str): The path where the dataset will be saved. Default is "OnlineRetail.xlsx".

    Outputs:
        - OnlineRetail.xlsx: The downloaded dataset file.
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
    print(f"Downloading dataset from {url}...")

    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte

    if response.status_code == 200:
        with open(destination, "wb") as f, tqdm(
            desc=os.path.basename(destination),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                f.write(data)
                bar.update(len(data))
        print(f"\n Dataset saved as '{destination}'")
    else:
        print("Failed to download dataset. Please try manually.")

if __name__ == "__main__":
    if not os.path.exists(destination_path):
        download_online_retail_data()
    else:
        print("Dataset already exists.")
