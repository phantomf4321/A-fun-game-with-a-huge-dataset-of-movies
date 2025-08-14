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

    # cumulative coverage — what fraction of ratings come from top X% users/movies
    def cumulative_coverage(self, counts, top_frac=0.1, id_col="userId"):
        c = counts.sort_values("n_ratings", ascending=False).reset_index(drop=True)
        cutoff = max(1, int(len(c) * top_frac))
        numer = c.loc[:cutoff - 1, "n_ratings"].sum()
        denom = c["n_ratings"].sum()
        return numer / denom

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

    def save_simple_plot(self, dataframe, vertex, xlabel, ylabel, title, filename):
        plt.figure()
        dataframe[vertex].plot()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        filename = "src/" + filename + ".png"
        plt.savefig(filename)
        print("log log histogram of {} is saved in {} successfully!".format(title, filename))


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

    def save_log_log_histogram(self, dataframe, vertex, xlabel, ylabel, title, filename):
        # Log-log style histograms (count of counts)
        plt.figure()
        u_vals = dataframe[vertex].values
        u_bins = np.logspace(0, np.log10(max(2, u_vals.max())), 50)
        plt.hist(u_vals, bins=u_bins)
        plt.xscale("log");
        plt.yscale("log")
        plt.xlabel(xlabel);
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        filename = "src/" + filename + ".png"
        plt.savefig(filename)
        print("log log histogram of {} is saved in {} successfully!".format(title, filename))

    def save_heatmap(self, dataframe, xlabel, ylabel, topx, topy, filename):
        plt.figure(figsize=(8, 6))
        plt.imshow(dataframe.values, aspect="auto", interpolation="nearest")
        plt.title(f"Sparsity Heatmap (1=rating present) — top {topx} users × top {topy} movies")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.colorbar(label="Present(1) / Missing(0)")
        filename = "src/" + filename + ".png"
        plt.savefig(filename)
        print("log log histogram of {} is saved in {} successfully!".format("heatmap", filename))
