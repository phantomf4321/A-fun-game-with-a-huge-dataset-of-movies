import ast
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

class General_Operations:
    def __init__(self):
        self.log = []
        print("General_Operations is called!")

    # --- Step logger to document row counts before/after each step ---
    def log_step(self, name, **counts):
        entry = {"step": name}
        entry.update(counts)
        self.log.append(entry)
        print(entry)

class Datasets:
    def __init__(self, directory):
        self.df = pd.read_csv(directory, low_memory=False)
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

    def tidy_json_list(self, x):
        """Parse a JSON-like list in movies_metadata (e.g., genres, production_countries)."""
        if pd.isna(x):
            return []
        s = str(x)
        try:
            return ast.literal_eval(s)
        except Exception:
            # Sometimes the field is already a list-like string but malformed; fallback:
            try:
                return json.loads(s)
            except Exception:
                return []


    def to_num(self, s, default=np.nan):
        try:
            return pd.to_numeric(s)
        except Exception:
            return default

class Plot:
    def __init__(self):
        print("Plot constructor is called!")

    def save_histogram(self, dataframe, vertex, title, filename):
        # Histogram
        plt.figure()
        bins = np.arange(0.25, 5.51, 0.5)  # centers for 0.5-step bins
        plt.hist(dataframe[vertex], bins=bins, edgecolor="black")
        plt.title(title)
        plt.xlabel(vertex)
        plt.ylabel("Count")
        plt.tight_layout()
        filename = "src/" + filename + ".png"
        plt.savefig(filename)
        print("histogram of {} is saved in {} successfully!".format(title, filename))