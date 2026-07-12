import os
import pandas as pd
import time

class CSVHandler:
    def __init__(self):
        pass

    def process_path(self, path):
        if os.path.isdir(path):
            return self.merge_csv_files(path)
        elif os.path.isfile(path) and path.lower().endswith('.csv'):
            return self.read_csv_file(path)
        else:
            raise ValueError("Invalid path: Not a directory or CSV file")

    def merge_csv_files(self, folder_path):
        dfs = []
        for filename in os.listdir(folder_path):
            if filename.lower().endswith('.csv'):
                file_path = os.path.join(folder_path, filename)
                dfs.append(pd.read_csv(file_path, low_memory=False))
        if len(dfs) == 0:
            raise ValueError("No CSV files found in the directory")
        df = pd.concat(dfs, ignore_index=True)

        # df.to_csv(folder_path + ".csv")

        return df

    def read_csv_file(self, file_path):
        return pd.read_csv(file_path)
