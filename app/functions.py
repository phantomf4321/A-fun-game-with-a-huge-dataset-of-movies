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
from scipy.sparse import hstack, csr_matrix
from typing import Dict, List, Set, Optional, Tuple
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
    def __init__(self, meta_subset, global_wr):
        print("Recommendator is constructed")
        self.meta_subset = meta_subset
        self.global_wr = global_wr

    def get_item_vectors(self):
        return self.item_vectors
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
        # map ids to indices
        self.user_ids = self.r_full["userId"].astype("category")
        self.item_ids = self.r_full["tmdbId"].astype("category")

        self.uid_map = dict(enumerate(self.user_ids.cat.categories))
        self.iid_map = dict(enumerate(self.item_ids.cat.categories))
        self.uid_inv = {v: k for k, v in self.uid_map.items()}
        self.iid_inv = {v: k for k, v in self.iid_map.items()}

        self.n_users = len(self.uid_map)
        self.n_items = len(self.iid_map)

        # sparse user-item rating matrix
        self.R = csr_matrix(
            (self.r_full["rating"].values,
             (self.user_ids.cat.codes.values, self.item_ids.cat.codes.values)),
            shape=(self.n_users, self.n_items)
        )

        # mean ratings per user (for Pearson)
        self.user_means = np.array(self.R.sum(axis=1)).ravel() / (self.R != 0).sum(axis=1).A1

    def get_R(self):
        return self.R
    def get_iid_map(self):
        return self.iid_map

    def get_uid_inv(self):
        return self.uid_inv

    def get_titles(self):
        return self.titles

    def item_item_knn(self, user_idx, k=50):
        """Score all items for a given user with item–item cosine."""
        user_row = self.R[user_idx, :]
        seen_items = user_row.nonzero()[1]
        if len(seen_items) == 0:
            return None

        # cosine sim between seen items and all items
        sims = cosine_similarity(self.R[:, seen_items].T, self.R.T)  # shape (#seen, n_items)
        ratings_seen = user_row[:, seen_items].toarray().ravel()

        # weighted sum
        scores = ratings_seen @ sims
        norms = np.abs(sims).sum(axis=0)
        scores = scores / np.maximum(norms, 1e-8)

        # zero out seen items
        scores[seen_items] = -np.inf
        return scores

    def user_user_knn(self, user_idx, k=50):
        """Score items for a given user with user–user Pearson similarity."""
        # convert to dense array (safe if dataset <10k users/items; otherwise we can optimize)
        R_dense = self.R.toarray().astype(float)

        # mean-center (ignoring zeros)
        mask = (R_dense != 0)
        user_means = np.divide(
            R_dense.sum(axis=1), mask.sum(axis=1), out=np.zeros_like(R_dense.sum(axis=1)), where=mask.sum(axis=1) != 0
        )
        R_centered = (R_dense - user_means[:, None]) * mask  # subtract mean only where ratings exist

        # similarities
        target = R_centered[user_idx, :].reshape(1, -1)
        sims = cosine_similarity(target, R_centered)[0]
        sims[user_idx] = 0  # ignore self

        # weighted sum of neighbor ratings
        scores = sims @ R_centered
        norms = np.abs(sims).sum()
        scores = scores / np.maximum(norms, 1e-8)

        # add back mean for this user
        scores += user_means[user_idx]

        # filter out seen items
        seen_items = self.R[user_idx, :].nonzero()[1]
        scores[seen_items] = -np.inf
        return scores

    # --------------------------
    # 2) Matrix Factorization (Biased SVD via SGD)
    # --------------------------
    def train_mf(self, n_factors, n_epochs, lr, reg, seed):
        rng = np.random.default_rng(seed)
        n_users, n_items = self.R.shape
        # latent factors
        P = 0.1 * rng.standard_normal((n_users, n_factors))
        Q = 0.1 * rng.standard_normal((n_items, n_factors))
        bu = np.zeros(n_users)
        bi = np.zeros(n_items)
        mu = self.R[self.R.nonzero()].mean()  # global mean

        rows, cols = self.R.nonzero()
        for epoch in range(n_epochs):
            for u, i in zip(rows, cols):
                r_ui = self.R[u, i]
                pred = mu + bu[u] + bi[i] + P[u, :] @ Q[i, :].T
                err = r_ui - pred
                # update biases
                bu[u] += lr * (err - reg * bu[u])
                bi[i] += lr * (err - reg * bi[i])
                # update latent factors
                P[u, :] += lr * (err * Q[i, :] - reg * P[u, :])
                Q[i, :] += lr * (err * P[u, :] - reg * Q[i, :])
        return mu, bu, bi, P, Q

    def mf_scores(self, user_idx):
        """ Predict scores for all items for given user."""
        mu, bu, bi, P, Q = self.train_mf(50, 20, 0.005, 0.02, 42)
        scores = mu + bu[user_idx] + bi + P[user_idx, :] @ Q.T
        seen = self.R[user_idx, :].nonzero()[1]
        scores[seen] = -np.inf
        return scores

    # --------------------------
    # 3) Recommend function
    # --------------------------
    def recommend(self, user_id, method, k, global_wr):
        if user_id not in self.uid_inv:
            # cold-start fallback
            if "global_wr" in globals():
                return global_wr.head(k)[["tmdbId", "title", "WR"]].rename(columns={"WR": "score"})
            return pd.DataFrame(columns=["tmdbId", "title", "score"])

        user_idx = self.uid_inv[user_id]
        if method == "itemknn":
            scores = self.item_item_knn(user_idx)
        elif method == "userknn":
            scores = self.user_user_knn(user_idx)
        else:
            scores = self.mf_scores(user_idx)

        top_idx = np.argpartition(-scores, k)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        tmdb_ids = [self.iid_map[i] for i in top_idx]
        self.titles = [TITLE_BY_ID[int(t)] if "TITLE_BY_ID" in globals() else str(t) for t in tmdb_ids]
        return pd.DataFrame({"tmdbId": tmdb_ids, "title": self.titles, "score": scores[top_idx]})


class Hybrid:
    def __init__(self, R, r_full, item_vectors, iid_map, uid_inv, global_wr, title_by_id):
        self.R = R
        self.r_full = r_full
        self.item_vectors = item_vectors
        self.iid_map = iid_map
        self.uid_inv = uid_inv
        self.global_wr = global_wr
        self.title_by_id = title_by_id
        self.knn = KNN(r_full)
        self.n_items = len(iid_map)
        print("Hybrid model initialized!")

    def _normalize_scores(self, x: np.ndarray) -> np.ndarray:
        """Min-max normalize per-user over finite scores; keep -inf as -inf."""
        if x is None:
            return np.full(self.n_items, -np.inf, dtype=float)

        x = x.astype(float).copy()
        finite_mask = np.isfinite(x)

        if not np.any(finite_mask):
            return np.full_like(x, -np.inf, dtype=float)

        finite_scores = x[finite_mask]
        lo, hi = np.min(finite_scores), np.max(finite_scores)

        if hi - lo < 1e-12:
            x[finite_mask] = 0.5
        else:
            x[finite_mask] = (finite_scores - lo) / (hi - lo)

        return x

    def _seen_items_for_user(self, user_idx: int) -> Set[int]:
        """Get set of seen item indices for a user."""
        return set(self.R[user_idx, :].nonzero()[1])

    def cb_scores_for_user(self, user_id: int, exclude_seen: bool = True) -> Optional[np.ndarray]:
        """Content-based similarity scores for all items for a given user."""
        try:
            # Get user ratings
            user_ratings = self.r_full[self.r_full["userId"] == user_id]
            if user_ratings.empty:
                print(f"User {user_id}: No ratings found")
                return None

            # Calculate mean-centered adjusted ratings
            mean_rating = user_ratings["rating"].mean()
            user_ratings = user_ratings.copy()
            user_ratings["adjusted_rating"] = user_ratings["rating"] - mean_rating

            # Filter to items with available vectors
            available_items = set(self.item_vectors.index.astype(int))
            user_items = user_ratings["tmdbId"].astype(int)
            valid_mask = user_items.isin(available_items)

            if not valid_mask.any():
                print(f"User {user_id}: No valid items with vectors")
                return None

            valid_items = user_items[valid_mask].values
            valid_weights = user_ratings.loc[valid_mask, "adjusted_rating"].values

            # Aggregate weights for duplicate items
            item_weights = {}
            for item_id, weight in zip(valid_items, valid_weights):
                item_weights[item_id] = item_weights.get(item_id, 0) + weight

            # Filter out items with negligible weights
            filtered_items = []
            filtered_weights = []

            for item_id, weight in item_weights.items():
                if abs(weight) > 1e-10:
                    filtered_items.append(item_id)
                    filtered_weights.append(weight)

            # Use uniform weights if all weights are negligible
            if not filtered_items:
                filtered_items = list(item_weights.keys())
                filtered_weights = [1.0] * len(filtered_items)
                print(f"User {user_id}: Using uniform weights")

            # Get vectors and compute weighted average profile
            try:
                item_vectors_subset = self.item_vectors.loc[filtered_items].values
                profile = np.average(item_vectors_subset, axis=0, weights=filtered_weights)
            except:
                profile = np.mean(item_vectors_subset, axis=0)

            # Compute similarities to all items
            all_item_vectors = self.item_vectors.values
            similarities = cosine_similarity([profile], all_item_vectors)[0]

            # Map similarities to CF item ordering
            scores = np.full(self.n_items, -np.inf, dtype=float)
            vector_idx_map = {int(mid): idx for idx, mid in enumerate(self.item_vectors.index.astype(int))}

            for cf_idx in range(self.n_items):
                tmdb_id = int(self.iid_map[cf_idx])
                if tmdb_id in vector_idx_map:
                    vec_idx = vector_idx_map[tmdb_id]
                    scores[cf_idx] = similarities[vec_idx]

            # Exclude already seen items if requested
            if exclude_seen and user_id in self.uid_inv:
                user_idx = self.uid_inv[user_id]
                seen_indices = self._seen_items_for_user(user_idx)
                scores[list(seen_indices)] = -np.inf

            return scores

        except Exception as e:
            print(f"Error in CB for user {user_id}: {e}")
            return None

    def cf_scores_for_user(self, user_id: int, method: str = "mf") -> Optional[np.ndarray]:
        """Get collaborative filtering scores for a user."""
        if user_id not in self.uid_inv:
            print(f"User {user_id} not in user index map")
            return None

        user_idx = self.uid_inv[user_id]

        try:
            if method == "itemknn":
                return self.knn.item_item_knn(user_idx)
            elif method == "userknn":
                return self.knn.user_user_knn(user_idx)
            else:
                return self.knn.mf_scores(user_idx)
        except Exception as e:
            print(f"Error in CF for user {user_id}: {e}")
            return None

    def hybrid_scores(self, user_id: int, alpha: float = 0.5, cf_method: str = "mf") -> Optional[np.ndarray]:
        """Combine CB and CF scores using weighted average."""
        cb_scores = self.cb_scores_for_user(user_id)
        cf_scores = self.cf_scores_for_user(user_id, method=cf_method)

        # Fallback strategies
        if cb_scores is None and cf_scores is None:
            print(f"No scores available for user {user_id}")
            return None
        if cb_scores is None:
            print(f"Using CF only for user {user_id}")
            return self._normalize_scores(cf_scores)
        if cf_scores is None:
            print(f"Using CB only for user {user_id}")
            return self._normalize_scores(cb_scores)

        # Normalize both score vectors
        cb_normalized = self._normalize_scores(cb_scores)
        cf_normalized = self._normalize_scores(cf_scores)

        # Blend only where both scores are valid
        valid_mask = np.isfinite(cb_normalized) & np.isfinite(cf_normalized)
        blended_scores = np.full_like(cb_normalized, -np.inf, dtype=float)

        if np.any(valid_mask):
            blended_scores[valid_mask] = (
                    alpha * cf_normalized[valid_mask] +
                    (1.0 - alpha) * cb_normalized[valid_mask]
            )

        return blended_scores

    def topk_from_scores(self, scores: np.ndarray, k: int = 10) -> np.ndarray:
        """Get top-k item indices from score vector."""
        if scores is None:
            return np.array([], dtype=int)

        valid_scores_mask = np.isfinite(scores)
        if not np.any(valid_scores_mask):
            return np.array([], dtype=int)

        valid_indices = np.where(valid_scores_mask)[0]
        valid_scores = scores[valid_scores_mask]

        if len(valid_indices) <= k:
            return valid_indices[np.argsort(-valid_scores)]

        # Use argpartition for efficiency
        top_k_indices = np.argpartition(-valid_scores, k)[:k]
        top_k_items = valid_indices[top_k_indices]

        # Return sorted by score (descending)
        return top_k_items[np.argsort(-scores[top_k_items])]

    def recommend_hybrid(self, user_id: int, alpha: float = 0.6,
                         cf_method: str = "mf", k: int = 10) -> pd.DataFrame:
        """Generate hybrid recommendations for a user."""
        scores = self.hybrid_scores(user_id, alpha=alpha, cf_method=cf_method)

        # Fallback to global popularity if no valid scores
        if scores is None or not np.any(np.isfinite(scores)):
            print(f"Falling back to global popularity for user {user_id}")
            return self.global_wr.head(k)[["tmdbId", "title", "WR"]].rename(columns={"WR": "score"})

        # Get top-k recommendations
        top_indices = self.topk_from_scores(scores, k=k)
        recommendations = []

        for idx in top_indices:
            tmdb_id = self.iid_map[idx]
            title = self.title_by_id.get(int(tmdb_id), f"Movie_{tmdb_id}")
            score = scores[idx]
            recommendations.append({"tmdbId": tmdb_id, "title": title, "score": score})

        return pd.DataFrame(recommendations)

    def leave_last_one_out(self, ratings_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data using leave-last-one-out validation."""
        ratings_sorted = ratings_df.sort_values(["userId", "timestamp"])
        last_indices = ratings_sorted.groupby("userId")["timestamp"].idxmax()

        test_set = ratings_sorted.loc[last_indices]
        train_set = ratings_sorted.drop(last_indices)

        # Keep only users with sufficient interactions
        user_counts = ratings_sorted["userId"].value_counts()
        valid_users = user_counts[user_counts >= 2].index
        test_set = test_set[test_set["userId"].isin(valid_users)]

        return train_set, test_set

    def tune_alpha(self, ratings_df: pd.DataFrame, cf_method: str = "mf",
                   K: int = 10, alpha_grid: List[float] = None) -> float:
        """Tune alpha parameter using Recall @ K on validation set."""
        if alpha_grid is None:
            alpha_grid = [0.0, 0.25, 0.5, 0.6, 0.7, 0.75, 1.0]

        train_df, val_df = self.leave_last_one_out(ratings_df)
        heldout_items = dict(zip(val_df["userId"].astype(int), val_df["tmdbId"].astype(int)))

        best_alpha = 0.5
        best_recall = 0.0
        recall_results = []

        for alpha in alpha_grid:
            hits = 0
            valid_users = 0

            for user_id, true_item in heldout_items.items():
                scores = self.hybrid_scores(user_id, alpha=alpha, cf_method=cf_method)
                if scores is None:
                    continue

                recommendations = self.topk_from_scores(scores, k=K)
                if len(recommendations) == 0:
                    continue

                # Convert CF indices to tmdbIds
                recommended_items = {int(self.iid_map[idx]) for idx in recommendations}
                if true_item in recommended_items:
                    hits += 1
                valid_users += 1

            recall = hits / max(1, valid_users)
            recall_results.append(recall)

            if recall > best_recall:
                best_recall = recall
                best_alpha = alpha

        print("Alpha tuning results:")
        for alpha, recall in zip(alpha_grid, recall_results):
            print(f"  α={alpha}: Recall@{K}={recall:.3f}")
        print(f"Best alpha: {best_alpha} (Recall@{K}={best_recall:.3f})")

        return best_alpha

