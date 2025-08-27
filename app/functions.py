import ast
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack

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

    def get_logs(self):
        return self.log

    def to_num(self, s, default=np.nan):
        try:
            return pd.to_numeric(s)
        except Exception:
            return default


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

    def save_bar(self, dataframe, xlabel, ylabel, title, filename):
        # Plots (bars) — genres & languages coverage
        plt.figure()
        dataframe.head(15).iloc[::-1].plot(kind="barh")
        plt.title(title)
        plt.xlabel(xlabel);
        plt.ylabel(ylabel)
        plt.tight_layout()
        filename = "src/" + filename + ".png"
        plt.savefig(filename)
        print("log log histogram of {} is saved in {} successfully!".format("heatmap", filename))


class Recommendator:
    def __init__(self, meta_subset):
        print("Recommendator is constructed")
        self.meta_subset = meta_subset

    def setup(self):
        # --- Text features (overview + tagline) ---
        self.meta_subset["overview"] = self.meta_subset["overview"].fillna("")
        self.meta_subset["tagline"] = self.meta_subset["tagline"].fillna("")
        self.meta_subset["text_all"] = self.meta_subset["overview"] + " " + self.meta_subset["tagline"]
        self.tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
        self.tfidf_matrix = self.tfidf.fit_transform(self.meta_subset["text_all"])

        # --- Multi-hot genres ---
        self.mlb_genres = MultiLabelBinarizer()
        self.genres_matrix = self.mlb_genres.fit_transform(
            self.meta_subset["genres"].apply(lambda g: g if isinstance(g, list) else []))

        # --- Multi-hot keywords (if keywords dataset available) ---
        # For now assume we have parsed keywords as a list in meta_clean["keywords_list"]
        # If not, set as empty lists
        if "keywords_list" not in self.meta_subset:
            self.meta_subset["keywords_list"] = [[] for _ in range(len(self.meta_subset))]
        self.mlb_keywords = MultiLabelBinarizer()
        self.keywords_matrix = self.mlb_keywords.fit_transform(self.meta_subset["keywords_list"])

        # --- Top-k cast/crew multi-hot (if credits dataset available) ---
        # Assume meta_subset["top_cast"] and ["top_crew"] are lists of names from preprocessing
        if "top_cast" not in self.meta_subset:
            self.meta_subset["top_cast"] = [[] for _ in range(len(self.meta_subset))]
        if "top_crew" not in self.meta_subset:
            self.meta_subset["top_crew"] = [[] for _ in range(len(self.meta_subset))]

        self.mlb_cast = MultiLabelBinarizer()
        self.cast_matrix = self.mlb_cast.fit_transform(self.meta_subset["top_cast"])

        self.mlb_crew = MultiLabelBinarizer()
        self.crew_matrix = self.mlb_crew.fit_transform(self.meta_subset["top_crew"])

        # --- Concatenate all features ---
        self.item_features = hstack([self.tfidf_matrix, self.genres_matrix, self.keywords_matrix, self.cast_matrix, self.crew_matrix])


        print("Recomendor setup complete!")
