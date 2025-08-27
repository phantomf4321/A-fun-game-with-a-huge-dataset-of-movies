import numpy as np
import pandas as pd

from collections import defaultdict

from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

from ExploratoryDataAnalysis import EDA


eda = EDA()
r_full = eda.get_r_full()
# --- 1) Compute movie-level stats from cleaned ratings ---
movie_stats = (
    r_full.groupby(["tmdbId", "title"], as_index=False)
          .agg(
              Ri=("rating", "mean"),   # mean rating
              vi=("rating", "count")   # vote count
          )
)

# attach genres list (first occurrence)
movie_stats["genres"] = movie_stats["tmdbId"].map(
    r_full.drop_duplicates("tmdbId")
          .set_index("tmdbId")["genres"]
)


# --- 2) IMDb WR global baseline ---
# Global C = weighted mean of all ratings
C = (movie_stats["Ri"] * movie_stats["vi"]).sum() / movie_stats["vi"].sum()

# m = 80th percentile of vote counts
m = np.quantile(movie_stats["vi"], 0.80)

# WR formula
v = movie_stats["vi"]
R = movie_stats["Ri"]
movie_stats["WR"] = (v / (v + m)) * R + (m / (v + m)) * C

# Keep movies with v >= m
global_wr = movie_stats[movie_stats["vi"] >= m].copy()
global_wr = global_wr.sort_values("WR", ascending=False)

print("Global WR parameters:")
print({"C": round(C, 3), "m": int(m), "m_quantile": 0.80})
print("\nTop 10 globally popular movies (WR):")
print(global_wr[["title", "vi", "Ri", "WR"]].head(10))


# --- 3) Per-genre WR ---
rows = []
for _, row in movie_stats.iterrows():
    genres = row["genres"] if isinstance(row["genres"], list) and row["genres"] else ["(No Genre)"]
    for g in genres:
        rows.append((g, row["tmdbId"], row["title"], row["vi"], row["Ri"]))
per_genre_df = pd.DataFrame(rows, columns=["genre", "tmdbId", "title", "vi", "Ri"])

# For C_g, use all ratings in r_full for that genre
exploded = r_full.explode("genres").rename(columns={"genres": "genre"})
Cg = exploded.groupby("genre")["rating"].mean()

# Apply WR within each genre
def genre_wr(group):
    m_g = np.quantile(group["vi"], 0.80) if len(group) else 0
    C_g = Cg.get(group.name, group["Ri"].mean())
    v = group["vi"]
    R = group["Ri"]
    group["WR_g"] = (v / (v + m_g)) * R + (m_g / (v + m_g)) * C_g
    group["m_g"] = m_g
    group["C_g"] = C_g
    return group[group["vi"] >= m_g].sort_values("WR_g", ascending=False)

per_genre_wr = per_genre_df.groupby("genre", group_keys=False).apply(genre_wr)

print("\nPer-genre WR example (Action):")
print(per_genre_wr[per_genre_wr["genre"] == "Action"][["title", "vi", "Ri", "WR_g"]].head(10))

# --- 4) Save results ---
global_wr.to_csv("data/baseline/baseline_global_wr.csv", index=False)
per_genre_wr.to_csv("data/baseline/baseline_per_genre_wr.csv", index=False)


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


# =============================
# 1) Prepare item feature matrix
# =============================

# --- a) Text features (overview + tagline) ---
# Make sure these fields exist in metadata
meta_subset = eda.meta_clean.copy()
meta_subset["overview"] = meta_subset["overview"].fillna("")
meta_subset["tagline"] = meta_subset["tagline"].fillna("")
meta_subset["text_all"] = meta_subset["overview"] + " " + meta_subset["tagline"]

tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
tfidf_matrix = tfidf.fit_transform(meta_subset["text_all"])

# --- b) Multi-hot genres ---
mlb_genres = MultiLabelBinarizer()
genres_matrix = mlb_genres.fit_transform(meta_subset["genres"].apply(lambda g: g if isinstance(g, list) else []))

# --- c) Multi-hot keywords (if keywords dataset available) ---
# For now assume we have parsed keywords as a list in meta_clean["keywords_list"]
# If not, set as empty lists
if "keywords_list" not in meta_subset:
    meta_subset["keywords_list"] = [[] for _ in range(len(meta_subset))]
mlb_keywords = MultiLabelBinarizer()
keywords_matrix = mlb_keywords.fit_transform(meta_subset["keywords_list"])

# --- d) Top-k cast/crew multi-hot (if credits dataset available) ---
# Assume meta_subset["top_cast"] and ["top_crew"] are lists of names from preprocessing
if "top_cast" not in meta_subset:
    meta_subset["top_cast"] = [[] for _ in range(len(meta_subset))]
if "top_crew" not in meta_subset:
    meta_subset["top_crew"] = [[] for _ in range(len(meta_subset))]

mlb_cast = MultiLabelBinarizer()
cast_matrix = mlb_cast.fit_transform(meta_subset["top_cast"])

mlb_crew = MultiLabelBinarizer()
crew_matrix = mlb_crew.fit_transform(meta_subset["top_crew"])

# --- e) Concatenate all features ---
from scipy.sparse import hstack
item_features = hstack([tfidf_matrix, genres_matrix, keywords_matrix, cast_matrix, crew_matrix])


# =============================
# 2) dimensionality reduction
# =============================
svd = TruncatedSVD(n_components=300, random_state=42)
item_features_reduced = svd.fit_transform(item_features)

# Map tmdbId → feature vector
item_vectors = pd.DataFrame(item_features_reduced, index=meta_subset["id"])


# =============================
# 3) Build user profiles
# =============================
def build_user_profile(user_id, ratings_df, item_vecs):
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
    profile_vec = np.average(rated_vecs, axis=0, weights=weights.flatten())
    return profile_vec

# =============================
# 4) Recommend for a user
# =============================
def recommend_content(user_id, ratings_df, item_vecs, top_n=10):
    profile = build_user_profile(user_id, ratings_df, item_vecs)
    if profile is None:
        # Cold start — fallback to global popularity
        return global_wr.head(top_n)[["title", "WR"]]

    # Compute cosine similarity to all items
    sims = cosine_similarity([profile], item_vecs.values)[0]
    sim_df = pd.DataFrame({
        "tmdbId": item_vecs.index,
        "similarity": sims
    })

    # Exclude items already rated by the user
    seen = set(ratings_df.loc[ratings_df["userId"] == user_id, "tmdbId"])
    sim_df = sim_df[~sim_df["tmdbId"].isin(seen)]


    # Attach titles
    meta_titles = meta_subset.groupby("id")["title"].first()
    sim_df["title"] = sim_df["tmdbId"].map(meta_titles)
    return sim_df.sort_values("similarity", ascending=False).head(top_n)

# =============================
# 5) Natural-language explanations
# =============================
def explain_recommendation(tmdbId, user_id, ratings_df, meta_df):
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

    explanation = []
    if genre_overlap:
        explanation.append(f"shares genres {', '.join(sorted(genre_overlap))}")
    if cast_overlap:
        explanation.append(f"features cast members {', '.join(sorted(cast_overlap))}")

    return " and ".join(explanation) if explanation else "matches your taste profile"

# =============================
# 6) Example usage
# =============================
user_id = 123  # example user
recommendations = recommend_content(user_id, r_full, item_vectors, top_n=10)

print(f"\nTop 10 recommendations for user {user_id}:")
for _, row in recommendations.iterrows():
    expl = explain_recommendation(row["tmdbId"], user_id, r_full, meta_subset)
    print(f"{row['title']}  —  {expl}")



# --------------------------
# 0) Build user-item matrix
# --------------------------
# map ids to indices
user_ids = r_full["userId"].astype("category")
item_ids = r_full["tmdbId"].astype("category")

uid_map = dict(enumerate(user_ids.cat.categories))
iid_map = dict(enumerate(item_ids.cat.categories))
uid_inv = {v: k for k, v in uid_map.items()}
iid_inv = {v: k for k, v in iid_map.items()}

n_users = len(uid_map)
n_items = len(iid_map)

# sparse user-item rating matrix
R = csr_matrix(
    (r_full["rating"].values,
     (user_ids.cat.codes.values, item_ids.cat.codes.values)),
    shape=(n_users, n_items)
)

# mean ratings per user (for Pearson)
user_means = np.array(R.sum(axis=1)).ravel() / (R != 0).sum(axis=1).A1

# --------------------------
# 1) Neighborhood CF
# --------------------------

def item_item_knn(user_idx, k=50):
    """Score all items for a given user with item–item cosine."""
    user_row = R[user_idx, :]
    seen_items = user_row.nonzero()[1]
    if len(seen_items) == 0:
        return None

    # cosine sim between seen items and all items
    sims = cosine_similarity(R[:, seen_items].T, R.T)  # shape (#seen, n_items)
    ratings_seen = user_row[:, seen_items].toarray().ravel()

    # weighted sum
    scores = ratings_seen @ sims
    norms = np.abs(sims).sum(axis=0)
    scores = scores / np.maximum(norms, 1e-8)

    # zero out seen items
    scores[seen_items] = -np.inf
    return scores

def user_user_knn(user_idx, k=50):
    """Score items for a given user with user–user Pearson similarity."""
    # convert to dense array (safe if dataset <10k users/items; otherwise we can optimize)
    R_dense = R.toarray().astype(float)

    # mean-center (ignoring zeros)
    mask = (R_dense != 0)
    user_means = np.divide(
        R_dense.sum(axis=1), mask.sum(axis=1), out=np.zeros_like(R_dense.sum(axis=1)), where=mask.sum(axis=1)!=0
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
    seen_items = R[user_idx, :].nonzero()[1]
    scores[seen_items] = -np.inf
    return scores


# --------------------------
# 2) Matrix Factorization (Biased SVD via SGD)
# --------------------------
def train_mf(R, n_factors=50, n_epochs=20, lr=0.005, reg=0.02, seed=42):
    rng = np.random.default_rng(seed)
    n_users, n_items = R.shape
    # latent factors
    P = 0.1 * rng.standard_normal((n_users, n_factors))
    Q = 0.1 * rng.standard_normal((n_items, n_factors))
    bu = np.zeros(n_users)
    bi = np.zeros(n_items)
    mu = R[R.nonzero()].mean()  # global mean

    rows, cols = R.nonzero()
    for epoch in range(n_epochs):
        for u, i in zip(rows, cols):
            r_ui = R[u, i]
            pred = mu + bu[u] + bi[i] + P[u, :] @ Q[i, :].T
            err = r_ui - pred
            # update biases
            bu[u] += lr * (err - reg * bu[u])
            bi[i] += lr * (err - reg * bi[i])
            # update latent factors
            P[u, :] += lr * (err * Q[i, :] - reg * P[u, :])
            Q[i, :] += lr * (err * P[u, :] - reg * Q[i, :])
    return mu, bu, bi, P, Q

mu, bu, bi, P, Q = train_mf(R, n_factors=50, n_epochs=15, lr=0.01, reg=0.05)

def mf_scores(user_idx):
    """Predict scores for all items for given user."""
    scores = mu + bu[user_idx] + bi + P[user_idx, :] @ Q.T
    seen = R[user_idx, :].nonzero()[1]
    scores[seen] = -np.inf
    return scores

# --------------------------
# 3) Recommend function
# --------------------------
def recommend(user_id, method="mf", k=10):
    if user_id not in uid_inv:
        # cold-start fallback
        if "global_wr" in globals():
            return global_wr.head(k)[["tmdbId","title","WR"]].rename(columns={"WR":"score"})
        return pd.DataFrame(columns=["tmdbId","title","score"])

    user_idx = uid_inv[user_id]
    if method == "itemknn":
        scores = item_item_knn(user_idx)
    elif method == "userknn":
        scores = user_user_knn(user_idx)
    else:
        scores = mf_scores(user_idx)

    top_idx = np.argpartition(-scores, k)[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    tmdb_ids = [iid_map[i] for i in top_idx]
    titles = [TITLE_BY_ID[int(t)] if "TITLE_BY_ID" in globals() else str(t) for t in tmdb_ids]
    return pd.DataFrame({"tmdbId": tmdb_ids, "title": titles, "score": scores[top_idx]})

# --------------------------
# Example Usage
# --------------------------
user_id = 123
print("\nTop-10 MF recommendations:")
print(recommend(user_id, method="mf", k=10))

print("\nTop-10 Item–Item recommendations:")
print(recommend(user_id, method="itemknn", k=10))

print("\nTop-10 User–User recommendations:")
print(recommend(user_id, method="userknn", k=10))


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
    """Min-max normalize per-user over finite scores; keep -inf as -inf."""
    x = x.astype(float).copy()
    finite = np.isfinite(x)
    if finite.sum() == 0:
        return np.full_like(x, -np.inf, dtype=float)  # no valid scores → all -inf
    lo, hi = np.nanmin(x[finite]), np.nanmax(x[finite])
    if hi - lo < 1e-12:
        x[finite] = 0.5
    else:
        x[finite] = (x[finite] - lo) / (hi - lo)
    # Replace any NaNs (just in case)
    x[~finite] = -np.inf
    return x



def _seen_items_for_user(user_idx):
    return set(R[user_idx, :].nonzero()[1])

# ---------- 2) Content-based per-user score vector (aligned to CF items) ----------
def cb_scores_for_user(user_id: int, k_exclude_seen=True) -> np.ndarray:
    """
    Returns a length-n_items vector of CB similarities for this user,
    aligned to item index ordering in iid_map. Unseen = similarity, seen = -inf (if k_exclude_seen).
    """
    # Build user profile exactly like Task 3 (mean-centered, rating-weighted)
    u_df = r_full[r_full["userId"] == user_id]
    if u_df.empty:
        return None  # cold user for CB

    mean_r = u_df["rating"].mean()
    u_df = u_df.assign(adj=u_df["rating"] - mean_r)

    # Align to items we actually have vectors for
    have_vec = item_vectors.index.astype(int)
    rated_ids = u_df["tmdbId"].astype(int)
    mask = rated_ids.isin(have_vec)
    if not mask.any():
        return None

    rated_ids = rated_ids[mask].values
    weights = u_df.loc[mask.index[mask], "adj"].values

    # Remove duplicates and aggregate weights for the same item
    unique_ids, unique_indices = np.unique(rated_ids, return_inverse=True)
    unique_weights = np.zeros(len(unique_ids))

    for i, idx in enumerate(unique_indices):
        unique_weights[idx] += weights[i]

    # Get vectors for unique items
    vectors = item_vectors.loc[unique_ids].values

    # Filter out zero weights to avoid division issues
    non_zero_mask = np.abs(unique_weights) > 1e-10
    if not non_zero_mask.any():
        return None

    vectors = vectors[non_zero_mask]
    unique_weights = unique_weights[non_zero_mask]

    if np.allclose(np.abs(unique_weights).sum(), 0.0):
        return None

    # Weighted average profile
    profile = np.average(vectors, axis=0, weights=unique_weights)

    # Similarity to all items
    sims = cosine_similarity([profile], item_vectors.values)[0]  # length = #items with vectors

    # Now expand to all items in CF order
    n_items = len(iid_map)
    scores = np.full(n_items, -np.inf, dtype=float)
    # Build map tmdbId->col_index in item_vectors
    tv_idx = {int(mid): j for j, mid in enumerate(item_vectors.index.astype(int))}
    for j in range(n_items):
        tmdb = int(iid_map[j])
        if tmdb in tv_idx:
            scores[j] = sims[tv_idx[tmdb]]

    # Optionally exclude seen
    if k_exclude_seen and user_id in uid_inv:
        uidx = uid_inv[user_id]
        seen = _seen_items_for_user(uidx)
        if seen:
            scores[list(seen)] = -np.inf

    return scores

# ---------- 3) CF score router ----------
def cf_scores_for_user(user_id: int, method="mf") -> np.ndarray:
    if user_id not in uid_inv:
        return None
    uidx = uid_inv[user_id]
    if method == "itemknn":
        return item_item_knn(uidx)
    elif method == "userknn":
        return user_user_knn(uidx)
    else:
        return mf_scores(uidx)

# ---------- 4) Hybrid scorer ----------
def hybrid_scores(user_id: int, alpha=0.5, cf_method="mf") -> np.ndarray:
    s_cb = cb_scores_for_user(user_id)
    s_cf = cf_scores_for_user(user_id, method=cf_method)

    # Fallback logic if one side is missing
    if s_cb is None and s_cf is None:
        return None
    if s_cb is None:
        return _normalize_scores(s_cf)
    if s_cf is None:
        return _normalize_scores(s_cb)

    # Normalize then blend
    s_cb_n = _normalize_scores(s_cb)
    s_cf_n = _normalize_scores(s_cf)
    return alpha * s_cf_n + (1.0 - alpha) * s_cb_n

# ---------- 5) Top-N recommend from any score vector ----------
def topk_from_scores(scores: np.ndarray, k=10) -> np.ndarray:
    finite = np.isfinite(scores)
    if not finite.any():
        return np.array([], dtype=int)
    cand = np.where(finite)[0]
    if len(cand) <= k:
        order = cand[np.argsort(-scores[cand])]
        return order
    # partial argpartition for speed
    idx = np.argpartition(-scores[cand], k)[:k]
    top = cand[idx]
    return top[np.argsort(-scores[top])]

def recommend_hybrid(user_id: int, alpha=0.6, cf_method="mf", k=10) -> pd.DataFrame:
    s = hybrid_scores(user_id, alpha=alpha, cf_method=cf_method)
    if s is None or (~np.isfinite(s)).all():
        # Coldest fallback to popularity
        if "global_wr" in globals():
            out = global_wr.head(k)[["tmdbId","title","WR"]].rename(columns={"WR":"score"}).copy()
            return out
        return pd.DataFrame(columns=["tmdbId","title","score"])
    top_idx = topk_from_scores(s, k=k)
    tmdb_ids = [iid_map[i] for i in top_idx]
    titles = [TITLE_BY_ID.get(int(t), str(t)) if "TITLE_BY_ID" in globals() else str(t) for t in tmdb_ids]
    return pd.DataFrame({"tmdbId": tmdb_ids, "title": titles, "score": s[top_idx]})

# ---------- 6) Validation split + α tuning (Recall@K under LLOO) ----------
def leave_last_one_out(df: pd.DataFrame):
    df = df.sort_values(["userId","timestamp"])
    last = df.groupby("userId")["timestamp"].idxmax()
    test = df.loc[last]
    train = df.drop(index=last, errors="ignore")
    # keep users with >=2 interactions in both sides
    multi = df["userId"].value_counts()
    test = test[test["userId"].isin(multi[multi>=2].index)]
    return train, test

def build_seen_dict(df: pd.DataFrame):
    d = {}
    for uid, g in df.groupby("userId"):
        d[uid] = set(g["tmdbId"].astype(int).tolist())
    return d

def tune_alpha(ratings_df: pd.DataFrame, cf_method="mf", K=10, grid=None) -> float:
    """
    Tune α by maximizing Recall@K on a LLOO validation set.
    """
    if grid is None:
        grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    train_df, val_df = leave_last_one_out(ratings_df)
    # For tuning we only need the held-out item per user
    heldout = dict(zip(val_df["userId"].astype(int), val_df["tmdbId"].astype(int)))

    recalls = []
    for a in grid:
        hits, total = 0, 0
        for uid, iid_true in heldout.items():
            s = hybrid_scores(uid, alpha=a, cf_method=cf_method)
            if s is None:
                continue
            top = topk_from_scores(s, k=K)
            if len(top) == 0:
                continue
            # map back to tmdbId
            top_tmdb = {int(iid_map[i]) for i in top}
            hits += int(iid_true in top_tmdb)
            total += 1
        recalls.append(hits / max(1, total))
    best_idx = int(np.argmax(recalls))
    best_alpha = grid[best_idx]
    print(f"[Hybrid] Tuned alpha={best_alpha} (Recall@{K}={recalls[best_idx]:.3f}) over grid {grid}")
    return best_alpha

# ---------- 7) Example: tune and recommend ----------
# IMPORTANT: r_full userId/tmdbId must be ints here
ratings_small = r_full[["userId","tmdbId","rating","timestamp"]].copy()
ratings_small["userId"] = ratings_small["userId"].astype(int)
ratings_small["tmdbId"] = ratings_small["tmdbId"].astype(int)

best_alpha = tune_alpha(ratings_small, cf_method="mf", K=10, grid=[0.0,0.25,0.5,0.6,0.7,0.75,1.0])

user_id = 123
print("\nTop-10 Hybrid (MF+CB) recommendations:")
print(recommend_hybrid(user_id, alpha=best_alpha, cf_method="mf", k=10))