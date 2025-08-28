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

    def dimensionality_reduction(self):
        svd = TruncatedSVD(n_components=300, random_state=42)
        item_features_reduced = svd.fit_transform(self.item_features)
        # Map tmdbId → feature vector
        self.item_vectors = pd.DataFrame(item_features_reduced, index=self.meta_subset["id"])

        print("Recomendor dimensionality_reduction complete!")

    def build_user_profile(self, user_id, ratings_df, item_vecs):
        # Get this user's ratings
        user_ratings = ratings_df[ratings_df["userId"] == user_id]
        if user_ratings.empty:
            return None

        # Mean center ratings
        mean_rating = user_ratings["rating"].mean()
        user_ratings = ratings_df[ratings_df["userId"] == user_id].copy()
        user_ratings["adj_rating"] = user_ratings["rating"] - mean_rating

        # Get feature vectors for rated items
        rated_vecs = item_vecs.loc[user_ratings["tmdbId"]]
        weights = user_ratings["adj_rating"].values.reshape(-1, 1)

        # Weighted average
        self.profile_vec = np.average(rated_vecs, axis=0, weights=weights.flatten())

        return self.profile_vec

    def recommend_content(self, user_id, ratings_df, item_vecs, top_n=10):
        profile = self.build_user_profile(user_id, ratings_df, item_vecs)
        if profile is None:
            # Cold start — fallback to global popularity
            return self.global_wr.head(top_n)[["title", "WR"]]

        # Compute cosine similarity to all items
        sims = cosine_similarity([profile], item_vecs.values)[0]
        self.sim_df = pd.DataFrame({
            "tmdbId": item_vecs.index,
            "similarity": sims
        })

        # Exclude items already rated by the user
        seen = set(ratings_df.loc[ratings_df["userId"] == user_id, "tmdbId"])
        self.sim_df = self.sim_df[~self.sim_df["tmdbId"].isin(seen)]

        # Attach titles
        meta_titles = self.meta_subset.groupby("id")["title"].first()
        self.sim_df["title"] = self.sim_df["tmdbId"].map(meta_titles)
        return self.sim_df.sort_values("similarity", ascending=False).head(top_n)

    def explain_recommendation(self, tmdbId, user_id, ratings_df, meta_df):
        # Find top overlapping genres and cast with items user liked
        rec_genres = set(meta_df.loc[meta_df["id"] == tmdbId, "genres"].values or [])
        rec_cast = set(meta_df.loc[meta_df["id"] == tmdbId, "top_cast"].values or [])

        # Get user's highly rated movies
        liked = ratings_df[(ratings_df["userId"] == user_id) & (ratings_df["rating"] >= 4.0)]
        liked_ids = liked["tmdbId"].tolist()

        liked_genres = set()
        liked_cast = set()
        for mid in liked_ids:
            liked_genres.update(meta_df.loc[meta_df["id"] == mid, "genres"].values or [])
            liked_cast.update(meta_df.loc[meta_df["id"] == mid, "top_cast"].values or [])

        genre_overlap = rec_genres & liked_genres
        cast_overlap = rec_cast & liked_cast

        self.explanation = []
        if genre_overlap:
            self.explanation.append(f"shares genres {', '.join(sorted(genre_overlap))}")
        if cast_overlap:
            self.explanation.append(f"features cast members {', '.join(sorted(cast_overlap))}")

        return " and ".join(self.explanation) if self.explanation else "matches your taste profile"

class KNN:
    def __init__(self, r_full):
        self.r_full = r_full
        print("KNN is called!")

