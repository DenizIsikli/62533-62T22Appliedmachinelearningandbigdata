# Name & Study Nr.
# Deniz Isikli s215818
# Mark Nielsen s204434

import os
import requests
from tqdm import tqdm

class DatasetDownloader:
    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.destination_path = os.path.join(self.script_dir, "OnlineRetail.xlsx")

        self.url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
        self.download_online_retail_data()

    def download_online_retail_data(self):
        """This function downloads the Online Retail dataset from UCI Machine Learning Repository.
        The dataset is an Excel file containing transactional data from a UK-based online retailer.

        Args:
            destination (str): The path where the dataset will be saved. Default is "OnlineRetail.xlsx".

        Outputs:
            - OnlineRetail.xlsx: The downloaded dataset file.
        """

        if os.path.exists(self.destination_path):
            print(f"Dataset already exists at {self.destination_path}. Skipping download.\n\n")
            return

        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
        print(f"Downloading dataset from {url}...")

        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte

        if response.status_code == 200:
            with open(self.destination_path, "wb") as f, tqdm(
                desc=os.path.basename(self.destination_path),
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(block_size):
                    f.write(data)
                    bar.update(len(data))
            print(f"\n Dataset saved as '{self.destination_path}'")
        else:
            print("Failed to download dataset. Please try manually.\n\n")
