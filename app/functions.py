import ast
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

class Datasets:
    def __init__(self, directory):
        self.df = pd.read_csv(directory)
        self.log = []
        print("Dataset constructor is called for {}".format(directory))

    def get_dataframe(self):
        return self.df

    def get_dataframe_col(self, col):
        return self.df[col]
    def parse_json_column(self, col):
        try:
            # First try proper JSON parsing
            return pd.io.json.loads(col)
        except:
            try:
                # If that fails, try literal_eval which handles Python-style strings
                return ast.literal_eval(col)
            except:
                # If all fails, return empty list
                return []

    # --- Step logger to document row counts before/after each step ---
    def log_step(self, name, **counts):
        entry = {"step": name}
        entry.update(counts)
        self.log.append(entry)
        print(entry)

class Plot:
    def __init__(self):
        print("Plot constructor is called!")

    def save_histogram(self, dataframe, title, filename):
        plt.hist(dataframe)
        filename = "src/" + filename + ".png"
        plt.savefig(filename)
        print("histogram of {} is saved in {} successfully!".format(title, filename))