import numpy as np
import pandas as pd

from collections import defaultdict


from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

from ExploratoryDataAnalysis import EDA
from models.baseline import Baseline
from app.functions import Recommendator, KNN


eda = EDA()
r_full = eda.get_r_full()
baseline = Baseline(r_full)
baseline.final_results()

meta_subset = eda.meta_clean.copy()
recom = Recommendator(meta_subset)
recom.setup()
recom.dimensionality_reduction()

user_id = 123  # example user
recommendations = recom.recommend_content(user_id, r_full, recom.item_vectors, top_n=10)

print(f"\nTop 10 recommendations for user {user_id}:")
for _, row in recommendations.iterrows():
    expl = recom.explain_recommendation(row["tmdbId"], user_id, r_full, meta_subset)
    print(f"{row['title']}  —  {expl}")

knn = KNN(r_full)
global_wr = baseline.get_global_wr()
print("\nTop-10 MF recommendations:")
print(knn.recommend(user_id, "mf", 10, global_wr))

print("\nTop-10 Item–Item recommendations:")
print(knn.recommend(user_id, "itemknn", 10, global_wr))

print("\nTop-10 User–User recommendations:")
print(knn.recommend(user_id, "userknn", 10, global_wr))

"""
# ---------- 0) Expect these globals from Tasks 3–4 ----------
# r_full: ratings with columns [userId, tmdbId, rating, timestamp]
# item_vectors: DataFrame, index=tmdbId (int), values = item features (SVD-reduced)
# R: csr_matrix user×item (Task 4)
# uid_inv: dict raw userId -> user index  (Task 4)
# iid_map: dict item index -> raw tmdbId   (Task 4)
# mf_scores / item_item_knn / user_user_knn (Task 4)
# global_wr: popularity fallback (Task 2)
# TITLE_BY_ID: optional dict tmdbId -> title (Task 3)
# ---------- 1) Utilities ----------

def _normalize_scores(x: np.ndarray) -> np.ndarray:
    ""Min-max normalize per-user over finite scores; keep -inf as -inf.""
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


def _seen_items_for_user(user_idx: int) -> set:
    ""Get set of seen item indices for a user.""
    return set(R[user_idx, :].nonzero()[1])


# ---------- 2) Content-based per-user score vector ----------
def cb_scores_for_user(user_id: int, exclude_seen: bool = True) -> np.ndarray:
    ""
    Returns CB similarity scores for all items for a given user.
    Unseen items get -inf if exclude_seen=True.
    ""
    # Get user ratings
    user_ratings = r_full[r_full["userId"] == user_id]
    if user_ratings.empty:
        return None

    # Calculate mean-centered adjusted ratings
    mean_rating = user_ratings["rating"].mean()
    user_ratings = user_ratings.copy()
    user_ratings["adjusted_rating"] = user_ratings["rating"] - mean_rating

    # Filter to items with available vectors
    available_items = set(item_vectors.index.astype(int))
    user_items = user_ratings["tmdbId"].astype(int)
    valid_mask = user_items.isin(available_items)

    if not valid_mask.any():
        return None

    valid_items = user_items[valid_mask].values
    valid_weights = user_ratings.loc[valid_mask.index[valid_mask], "adjusted_rating"].values

    # Aggregate weights for duplicate items
    item_weights = {}
    for item_id, weight in zip(valid_items, valid_weights):
        item_weights[item_id] = item_weights.get(item_id, 0) + weight

    # Filter out items with negligible weights and check if weights sum to zero
    filtered_items = []
    filtered_weights = []
    total_weight = 0.0

    for item_id, weight in item_weights.items():
        if abs(weight) > 1e-10:
            filtered_items.append(item_id)
            filtered_weights.append(weight)
            total_weight += abs(weight)

    # If all weights are zero or negligible, use uniform weights
    if total_weight < 1e-10:
        filtered_items = list(item_weights.keys())
        filtered_weights = [1.0] * len(filtered_items)  # Uniform weights
        print(f"User {user_id}: Using uniform weights (adjusted ratings sum to zero)")

    if not filtered_items:
        return None

    # Get vectors and compute weighted average profile
    try:
        item_vectors_subset = item_vectors.loc[filtered_items].values

        # Safe weighted average with zero-division protection
        if np.sum(np.abs(filtered_weights)) < 1e-10:
            profile = np.mean(item_vectors_subset, axis=0)
        else:
            profile = np.average(item_vectors_subset, axis=0, weights=filtered_weights)

    except (KeyError, ValueError) as e:
        print(f"Error processing user {user_id}: {e}")
        return None

    # Compute similarities to all items
    all_item_vectors = item_vectors.values
    similarities = cosine_similarity([profile], all_item_vectors)[0]

    # Map similarities to CF item ordering
    n_items = len(iid_map)
    scores = np.full(n_items, -np.inf, dtype=float)

    # Create mapping from tmdbId to item_vectors index
    vector_idx_map = {int(mid): idx for idx, mid in enumerate(item_vectors.index.astype(int))}

    for cf_idx in range(n_items):
        tmdb_id = int(iid_map[cf_idx])
        if tmdb_id in vector_idx_map:
            scores[cf_idx] = similarities[vector_idx_map[tmdb_id]]

    # Exclude already seen items if requested
    if exclude_seen and user_id in uid_inv:
        user_idx = uid_inv[user_id]
        seen_indices = _seen_items_for_user(user_idx)
        scores[list(seen_indices)] = -np.inf

    return scores


# ---------- 3) CF score router ----------
def cf_scores_for_user(user_id: int, method: str = "mf") -> np.ndarray:
    ""Get collaborative filtering scores for a user.""
    if user_id not in uid_inv:
        return None

    user_idx = uid_inv[user_id]

    if method == "itemknn":
        return item_item_knn(user_idx)
    elif method == "userknn":
        return user_user_knn(user_idx)
    else:
        return mf_scores(user_idx)


# ---------- 4) Hybrid scorer ----------
def hybrid_scores(user_id: int, alpha: float = 0.5, cf_method: str = "mf") -> np.ndarray:
    ""Combine CB and CF scores using weighted average.""
    cb_scores = cb_scores_for_user(user_id)
    cf_scores = cf_scores_for_user(user_id, method=cf_method)

    # Fallback strategies
    if cb_scores is None and cf_scores is None:
        return None
    if cb_scores is None:
        return _normalize_scores(cf_scores)
    if cf_scores is None:
        return _normalize_scores(cb_scores)

    # Normalize both score vectors
    cb_normalized = _normalize_scores(cb_scores)
    cf_normalized = _normalize_scores(cf_scores)

    # Blend only where both scores are valid
    valid_mask = np.isfinite(cb_normalized) & np.isfinite(cf_normalized)
    blended_scores = np.full_like(cb_normalized, -np.inf, dtype=float)

    if np.any(valid_mask):
        blended_scores[valid_mask] = (
                alpha * cf_normalized[valid_mask] +
                (1.0 - alpha) * cb_normalized[valid_mask]
        )

    return blended_scores


# ---------- 5) Top-N recommendation ----------
def topk_from_scores(scores: np.ndarray, k: int = 10) -> np.ndarray:
    ""Get top-k item indices from score vector.""
    valid_scores_mask = np.isfinite(scores)
    if not np.any(valid_scores_mask):
        return np.array([], dtype=int)

    valid_indices = np.where(valid_scores_mask)[0]
    valid_scores = scores[valid_scores_mask]

    if len(valid_indices) <= k:
        return valid_indices[np.argsort(-valid_scores)]

    # Use argpartition for efficiency with large arrays
    top_k_indices = np.argpartition(-valid_scores, k)[:k]
    top_k_items = valid_indices[top_k_indices]

    # Return sorted by score (descending)
    return top_k_items[np.argsort(-scores[top_k_items])]


def recommend_hybrid(user_id: int, alpha: float = 0.6, cf_method: str = "mf", k: int = 10) -> pd.DataFrame:
    ""Generate hybrid recommendations for a user.""
    scores = hybrid_scores(user_id, alpha=alpha, cf_method=cf_method)

    # Fallback to global popularity if no valid scores
    if scores is None or not np.any(np.isfinite(scores)):
        if "global_wr" in globals():
            return global_wr.head(k)[["tmdbId", "title", "WR"]].rename(columns={"WR": "score"})
        return pd.DataFrame(columns=["tmdbId", "title", "score"])

    # Get top-k recommendations
    top_indices = topk_from_scores(scores, k=k)
    recommendations = []

    for idx in top_indices:
        tmdb_id = iid_map[idx]
        title = TITLE_BY_ID.get(int(tmdb_id), str(tmdb_id)) if "TITLE_BY_ID" in globals() else str(tmdb_id)
        score = scores[idx]
        recommendations.append({"tmdbId": tmdb_id, "title": title, "score": score})

    return pd.DataFrame(recommendations)


# ---------- 6) Validation and tuning ----------
def leave_last_one_out(ratings_df: pd.DataFrame) -> tuple:
    ""Split data using leave-last-one-out validation.""
    ratings_sorted = ratings_df.sort_values(["userId", "timestamp"])
    last_indices = ratings_sorted.groupby("userId")["timestamp"].idxmax()

    test_set = ratings_sorted.loc[last_indices]
    train_set = ratings_sorted.drop(last_indices, errors="ignore")

    # Keep only users with sufficient interactions
    user_counts = ratings_sorted["userId"].value_counts()
    valid_users = user_counts[user_counts >= 2].index
    test_set = test_set[test_set["userId"].isin(valid_users)]

    return train_set, test_set


def tune_alpha(ratings_df: pd.DataFrame, cf_method: str = "mf", K: int = 10,
               alpha_grid: list = None) -> float:
    ""Tune alpha parameter using Recall@K on validation set.""
    if alpha_grid is None:
        alpha_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    train_df, val_df = leave_last_one_out(ratings_df)
    heldout_items = dict(zip(val_df["userId"].astype(int), val_df["tmdbId"].astype(int)))

    best_alpha = 0.5
    best_recall = 0.0
    recall_results = []

    for alpha in alpha_grid:
        hits = 0
        valid_users = 0

        for user_id, true_item in heldout_items.items():
            scores = hybrid_scores(user_id, alpha=alpha, cf_method=cf_method)
            if scores is None:
                continue

            recommendations = topk_from_scores(scores, k=K)
            if len(recommendations) == 0:
                continue

            # Convert CF indices to tmdbIds
            recommended_items = {int(iid_map[idx]) for idx in recommendations}
            if true_item in recommended_items:
                hits += 1
            valid_users += 1

        recall = hits / max(1, valid_users)
        recall_results.append(recall)

        if recall > best_recall:
            best_recall = recall
            best_alpha = alpha

    print(f"Alpha tuning results:")
    for alpha, recall in zip(alpha_grid, recall_results):
        print(f"  α={alpha}: Recall@{K}={recall:.3f}")
    print(f"Best alpha: {best_alpha} (Recall@{K}={best_recall:.3f})")

    return best_alpha


# ---------- 7) Example usage ----------
# Prepare data
ratings_small = r_full[["userId", "tmdbId", "rating", "timestamp"]].copy()
ratings_small["userId"] = ratings_small["userId"].astype(int)
ratings_small["tmdbId"] = ratings_small["tmdbId"].astype(int)

# Tune alpha parameter
try:
    best_alpha = tune_alpha(
        ratings_small,
        cf_method="mf",
        K=10,
        alpha_grid=[0.0, 0.25, 0.5, 0.6, 0.7, 0.75, 1.0]
    )
except Exception as e:
    print(f"Error during alpha tuning: {e}")
    best_alpha = 0.6  # Fallback value

# Generate recommendations for a user
user_id = 123
print(f"\nTop-10 Hybrid (MF+CB) recommendations for user {user_id}:")
try:
    recommendations = recommend_hybrid(user_id, alpha=best_alpha, cf_method="mf", k=10)
    print(recommendations)
except Exception as e:
    print(f"Error generating recommendations: {e}")"""