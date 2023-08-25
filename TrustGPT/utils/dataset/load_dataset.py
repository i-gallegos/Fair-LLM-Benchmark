import random
import requests
import zipfile
import os
import pandas as pd


class LoadDataset:
    """
    Load a dataset.
    """

    def __init__(self):
        self.dataset_name = None

    def download_dataset(self, dataset_name):
        """
        Download a dataset based on its name.
        """
        self.dataset_name = dataset_name
        if self.dataset_name == 'social101':
            # Determine whether "./social-chem-101/" exists.
            if not os.path.exists("./social-chem-101/"):
                # Define the URL and local file path for the ZIP file to download
                url = "https://storage.googleapis.com/ai2-mosaic-public/projects/social-chemistry/data/social-chem" \
                      "-101.zip "
                zip_file_path = "./social-chem-101.zip"

                # Send a download request
                print("Retrieving URL: {}".format(url))
                response = requests.get(url)

                # Save the downloaded content to a local file
                with open(zip_file_path, "wb") as file:
                    file.write(response.content)

                print("Download completed.")
                # Extract the ZIP file
                with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
                    zip_ref.extractall("./")

                print("Extraction completed.")
                # Remove the original ZIP file
                os.remove(zip_file_path)

                # Return the path to the downloaded dataset
                return "./social-chem-101/"
            else:
                print("Dataset already exists.")
        else:
            raise ValueError("Unsupported dataset_name. Please provide a valid dataset name.")

    def preprocess_social101(self, return_type='all', shuffle=False, size=None, seed=None):
        """
        Determine whether "./social-chem-101/" exists. Then preprocess the dataset.
        """
        if not os.path.exists("./social-chem-101/"):
            self.download_dataset('social101')

        # 读取TSV文件
        df = pd.read_csv('social-chem-101/social-chem-101.v1.0.tsv', sep='\t')

        # 将DataFrame转换为包含字典的列表
        data_list = df.to_dict(orient='records')
        print("preprocessing...")
        if size is not None:
            if seed is not None and shuffle == True:
                random.seed(seed)
                data_list = random.sample(data_list, size)
            elif seed is None and shuffle == True:
                data_list = random.sample(data_list, size)
            else:
                data_list = data_list[:size]
        print("preprocessing completed.")
        if return_type == 'all':
            return data_list
        elif return_type == 'bias':
            bias_data_list = [el['action'] for el in data_list]
            return bias_data_list, data_list
        elif return_type == 'toxicity':
            toxicity_data_list = [el['action'] for el in data_list]
            return toxicity_data_list, data_list
        elif return_type == 'value-alignment':
            value_alignment_data_dict = {}
            for el in data_list:
                value_alignment_data_dict[el['action']] = el['rot-judgement']
            return value_alignment_data_dict, data_list
        else:
            raise ValueError("Unsupported return_type. Please provide a valid return_type.")
