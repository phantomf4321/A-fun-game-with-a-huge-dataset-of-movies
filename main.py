import numpy as np
import pandas as pd

from collections import defaultdict


from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

from ExploratoryDataAnalysis import EDA
from models.baseline import Baseline
from app.functions import Recommendator, KNN, Hybrid


eda = EDA()
r_full = eda.get_r_full()
baseline = Baseline(r_full)
baseline_res = baseline.final_results()
global_wr = baseline_res["global_wr"]

meta_subset = eda.meta_clean.copy()
recom = Recommendator(meta_subset, global_wr)
recom.setup()
recom.dimensionality_reduction()

user_id = 123  # example user
recommendations = recom.recommend_content(user_id, r_full, recom.item_vectors, top_n=10)

print(f"\nTop 10 recommendations for user {user_id}:")
for _, row in recommendations.iterrows():
    expl = recom.explain_recommendation(row["tmdbId"], user_id, r_full, meta_subset)
    print(f"{row['title']}  —  {expl}")

knn = KNN(r_full)
global_wr = baseline_res["global_wr"]
print("\nTop-10 MF recommendations:")
print(knn.recommend(user_id, "mf", 10, global_wr))

print("\nTop-10 Item–Item recommendations:")
print(knn.recommend(user_id, "itemknn", 10, global_wr))

print("\nTop-10 User–User recommendations:")
print(knn.recommend(user_id, "userknn", 10, global_wr))








# =========================
# Task 6: Evaluation Module
# =========================
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import math
import matplotlib.pyplot as plt

# ---------------------------
# 0) Time-aware split (LLOO)
# ---------------------------
def leave_last_one_out(df: pd.DataFrame):
    """
    Sort by time per user; hold out the last interaction for validation.
    Keep only users with >=2 interactions (so both train & test exist).
    """
    df = df.sort_values(["userId", "timestamp"])
    last_idx = df.groupby("userId")["timestamp"].idxmax()
    val = df.loc[last_idx]
    train = df.drop(index=last_idx, errors="ignore")
    user_counts = df["userId"].value_counts()
    valid_users = user_counts[user_counts >= 2].index
    val = val[val["userId"].isin(valid_users)]
    train = train[train["userId"].isin(valid_users)]
    return train, val

ratings_eval = r_full[["userId","tmdbId","rating","timestamp"]].copy()
ratings_eval["userId"] = ratings_eval["userId"].astype(int)
ratings_eval["tmdbId"] = ratings_eval["tmdbId"].astype(int)
train_df, val_df = leave_last_one_out(ratings_eval)

# Popularity from TRAIN (for novelty & coverage denominators)
item_pop_counts = train_df["tmdbId"].value_counts().to_dict()
total_interactions = float(len(train_df))

# ---------------------------
# 1) Metric helpers
# ---------------------------
def precision_at_k(recommended, relevant_set, k):
    if k == 0: return 0.0
    rec_k = recommended[:k]
    hits = sum(1 for i in rec_k if i in relevant_set)
    return hits / k

def recall_at_k(recommended, relevant_set, k):
    if len(relevant_set) == 0: return 0.0
    rec_k = recommended[:k]
    hits = sum(1 for i in rec_k if i in relevant_set)
    return hits / len(relevant_set)

def hit_rate_at_k(recommended, relevant_set, k):
    rec_k = recommended[:k]
    return 1.0 if any(i in relevant_set for i in rec_k) else 0.0

def ap_at_k(recommended, relevant_set, k):
    """Average Precision@K."""
    if len(relevant_set) == 0: return 0.0
    rec_k = recommended[:k]
    score, hits = 0.0, 0
    for rank, iid in enumerate(rec_k, start=1):
        if iid in relevant_set:
            hits += 1
            score += hits / rank
    return score / min(len(relevant_set), k)

def ndcg_at_k(recommended, relevant_set, k):
    """Binary relevance DCG/IDCG."""
    rec_k = recommended[:k]
    dcg = 0.0
    for rank, iid in enumerate(rec_k, start=1):
        rel = 1.0 if iid in relevant_set else 0.0
        if rel > 0:
            dcg += 1.0 / math.log2(rank + 1)
    # IDCG for |relevant| min k
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0

def catalog_coverage(all_recommended_items, total_catalog_size):
    """Unique recommended items / total catalog size."""
    return len(set(all_recommended_items)) / max(1, total_catalog_size)

def novelty_of_list(recommended, pop_counts, total_interactions):
    """Mean self-information using train popularity: -log2(pop(i))."""
    vals = []
    for i in recommended:
        p = pop_counts.get(int(i), 0.5) / max(1.0, total_interactions)  # smoothing for unseen
        vals.append(-math.log2(max(p, 1e-12)))
    return float(np.mean(vals)) if vals else 0.0

def intra_list_diversity(recommended, item_vectors_lookup):
    """
    Average pairwise dissimilarity among items in the list.
    Use content vectors; ILD = 1 - cosine.
    """
    if len(recommended) <= 1:
        return 0.0
    vecs = []
    for mid in recommended:
        v = item_vectors_lookup.get(int(mid))
        if v is not None:
            vecs.append(v)
    if len(vecs) <= 1:
        return 0.0
    V = np.vstack(vecs)
    S = cosine_similarity(V)
    # exclude diagonal
    n = S.shape[0]
    mask = ~np.eye(n, dtype=bool)
    sim_mean = S[mask].mean()
    return 1.0 - sim_mean

item_vectors = recom.get_item_vectors()
# Build a lightweight lookup for ILD from your Task 3 item_vectors
_item_vec_lookup = {int(mid): item_vectors.loc[int(mid)].values for mid in item_vectors.index.astype(int)}

# ---------------------------
# 2) Model wrappers → score vectors
# ---------------------------
uid_inv = knn.get_uid_inv()
def scores_cf(user_id, kind="mf"):
    if user_id not in uid_inv:  # unseen user in CF
        return None
    uidx = uid_inv[user_id]
    if kind == "mf":
        return knn.mf_scores(uidx)
    elif kind == "itemknn":
        return knn.item_item_knn(uidx)
    elif kind == "userknn":
        return knn.user_user_knn(uidx)
    else:
        raise ValueError("Unknown CF kind")

"""def scores_cb(user_id):
    return cb_scores_for_user(user_id)  # must return length-n_items vector aligned to iid_map

def scores_hybrid(user_id, alpha=0.6, cf_method="mf"):
    return hybrid_scores(user_id, alpha=alpha, cf_method=cf_method)"""

def topk_from_scores(scores, k):
    if scores is None:
        return np.array([], dtype=int)
    finite = np.isfinite(scores)
    if not finite.any():
        return np.array([], dtype=int)
    cand = np.where(finite)[0]
    if len(cand) <= k:
        order = cand[np.argsort(-scores[cand])]
        return order
    idx = np.argpartition(-scores[cand], k)[:k]
    top = cand[idx]
    return top[np.argsort(-scores[top])]

# ---------------------------
# 3) Single-run evaluation on LLOO
# ---------------------------
iid_map = knn.get_iid_map()
def evaluate_models_at_k(models, K_list=(10,20), alpha_for_hybrid=0.6, cf_for_hybrid="mf"):
    """
    models: dict name -> {'type': 'cf'|'cb'|'hybrid', 'cf_kind': 'mf'|'itemknn'|'userknn'}
    Returns: long DataFrame with metrics per model and K, plus extras (coverage, novelty, ILD).
    """
    results = []
    all_rec_items_by_model = defaultdict(list)

    # Held-out (one per user)
    heldout = dict(zip(val_df["userId"].astype(int), val_df["tmdbId"].astype(int)))

    for name, spec in models.items():
        for K in K_list:
            precs, recs, hrs, maps, ndcgs = [], [], [], [], []
            novs, ilds = [], []
            for uid, iid_true in heldout.items():
                if spec["type"] == "cf":
                    scores = scores_cf(uid, kind=spec.get("cf_kind","mf"))
                """elif spec["type"] == "cb":
                    scores = scores_cb(uid)
                else:
                    scores = scores_hybrid(uid, alpha=alpha_for_hybrid, cf_method=cf_for_hybrid)"""

                top_idx = topk_from_scores(scores, k=K)
                tmdb_top = [int(iid_map[i]) for i in top_idx]
                all_rec_items_by_model[name].extend(tmdb_top)

                rel = {iid_true}
                precs.append(precision_at_k(tmdb_top, rel, K))
                recs.append(recall_at_k(tmdb_top, rel, K))
                hrs.append(hit_rate_at_k(tmdb_top, rel, K))
                maps.append(ap_at_k(tmdb_top, rel, K))
                ndcgs.append(ndcg_at_k(tmdb_top, rel, K))

                novs.append(novelty_of_list(tmdb_top, item_pop_counts, total_interactions))
                ilds.append(intra_list_diversity(tmdb_top, _item_vec_lookup))

            # Aggregate
            res = dict(
                model=name, K=K,
                precision=np.mean(precs) if precs else 0.0,
                recall=np.mean(recs) if recs else 0.0,
                hit_rate=np.mean(hrs) if hrs else 0.0,
                map=np.mean(maps) if maps else 0.0,
                ndcg=np.mean(ndcgs) if ndcgs else 0.0,
                novelty=np.mean(novs) if novs else 0.0,
                ild=np.mean(ilds) if ilds else 0.0,
            )
            # catalog coverage (denominator = #items appearing in train)
            catalog_size = train_df["tmdbId"].nunique()
            res["coverage"] = catalog_coverage(all_rec_items_by_model[name], catalog_size)
            results.append(res)

    return pd.DataFrame(results)

# ---------------------------
# 4) Bootstrap 95% CIs (user resampling)
# ---------------------------
def bootstrap_eval(models, K_list=(10,20), B=200, alpha_for_hybrid=0.6, cf_for_hybrid="mf", random_state=42):
    rng = np.random.default_rng(random_state)
    users = val_df["userId"].astype(int).unique()
    metrics = ["precision","recall","hit_rate","map","ndcg","novelty","ild","coverage"]
    # store per-bootstrap scores
    boot_records = []

    for b in range(B):
        sampled_users = rng.choice(users, size=len(users), replace=True)
        sampled_val = val_df[val_df["userId"].isin(sampled_users)].copy()

        # build held-out for this bootstrap
        heldout = dict(zip(sampled_val["userId"].astype(int), sampled_val["tmdbId"].astype(int)))
        # recompute coverage denominator
        catalog_size = train_df["tmdbId"].nunique()

        for name, spec in models.items():
            # accumulate recommended items for coverage inside bootstrap
            all_rec = defaultdict(list)
            for K in K_list:
                precs, recs, hrs, maps, ndcgs = [], [], [], [], []
                novs, ilds = [], []
                for uid in sampled_users:
                    iid_true = heldout.get(uid, None)
                    if iid_true is None:
                        continue
                    if spec["type"] == "cf":
                        scores = scores_cf(uid, kind=spec.get("cf_kind","mf"))
                    """elif spec["type"] == "cb":
                        scores = scores_cb(uid)
                    else:
                        scores = scores_hybrid(uid, alpha=alpha_for_hybrid, cf_method=cf_for_hybrid)"""

                    top_idx = topk_from_scores(scores, k=K)
                    tmdb_top = [int(iid_map[i]) for i in top_idx]
                    all_rec[K].extend(tmdb_top)

                    rel = {iid_true}
                    precs.append(precision_at_k(tmdb_top, rel, K))
                    recs.append(recall_at_k(tmdb_top, rel, K))
                    hrs.append(hit_rate_at_k(tmdb_top, rel, K))
                    maps.append(ap_at_k(tmdb_top, rel, K))
                    ndcgs.append(ndcg_at_k(tmdb_top, rel, K))
                    novs.append(novelty_of_list(tmdb_top, item_pop_counts, total_interactions))
                    ilds.append(intra_list_diversity(tmdb_top, _item_vec_lookup))

                # record bootstrap means
                rec_cov = catalog_coverage(all_rec[K], catalog_size)
                boot_records.append({
                    "b": b, "model": name, "K": K,
                    "precision": np.mean(precs) if precs else 0.0,
                    "recall": np.mean(recs) if recs else 0.0,
                    "hit_rate": np.mean(hrs) if hrs else 0.0,
                    "map": np.mean(maps) if maps else 0.0,
                    "ndcg": np.mean(ndcgs) if ndcgs else 0.0,
                    "novelty": np.mean(novs) if novs else 0.0,
                    "ild": np.mean(ilds) if ilds else 0.0,
                    "coverage": rec_cov
                })

    boot_df = pd.DataFrame(boot_records)

    # aggregate: mean and 95% CI
    rows = []
    for (m, K), g in boot_df.groupby(["model","K"]):
        row = {"model": m, "K": int(K)}
        for metric in metrics:
            vals = g[metric].values
            mu = vals.mean()
            lo, hi = np.percentile(vals, [2.5, 97.5])
            row[f"{metric}_mean"] = mu
            row[f"{metric}_lo"] = lo
            row[f"{metric}_hi"] = hi
        rows.append(row)
    ci_df = pd.DataFrame(rows)
    return boot_df, ci_df

# ---------------------------
# 5) Define which models to compare
# ---------------------------
models = {
    #"CB": {"type": "cb"},
    "CF-MF": {"type": "cf", "cf_kind": "mf"},
    "CF-ItemKNN": {"type": "cf", "cf_kind": "itemknn"},
    "CF-UserKNN": {"type": "cf", "cf_kind": "userknn"},
    #"Hybrid(MF+CB)": {"type": "hybrid"},  # uses alpha below
}

# Tune alpha beforehand (Task 5) or pick a value; we’ll use 0.6 as example
alpha_star = 0.6

# ---------------------------
# 6) Run evaluation + bootstrap CIs
# ---------------------------
point_df = evaluate_models_at_k(models, K_list=(10,20), alpha_for_hybrid=alpha_star, cf_for_hybrid="mf")
boot_df, ci_df = bootstrap_eval(models, K_list=(10,20), B=200, alpha_for_hybrid=alpha_star, cf_for_hybrid="mf")

"""print("\nPoint estimates (no CI):")
print(point_df.sort_values(["K","model"]))

print("\nBootstrap 95% CI summary:")
print(ci_df.sort_values(["K","model"]))"""

# ---------------------------
# 7) Visualization (tables & plots)
# ---------------------------
def plot_metric_with_ci(ci_df, metric, Ks=(10,20)):
    plt.figure(figsize=(8,5))
    # one subplot per K as separate series overlayed by model
    x_positions = {}
    unique_models = list(ci_df["model"].unique())
    for i, K in enumerate(Ks):
        sub = ci_df[ci_df["K"] == K]
        xs = np.arange(len(unique_models)) + (i * 0.12)  # small offset per K
        x_positions[K] = xs
        means = sub[f"{metric}_mean"].values
        los = sub[f"{metric}_lo"].values
        his = sub[f"{metric}_hi"].values
        errs = np.vstack([means - los, his - means])
        plt.errorbar(xs, means, yerr=errs, fmt='o', capsize=3, label=f"K={K}")
    plt.xticks(np.arange(len(unique_models)) + 0.06, unique_models, rotation=0)
    plt.ylabel(metric.upper())
    plt.title(f"{metric.upper()} with 95% Bootstrap CI")
    plt.legend()
    plt.tight_layout()
    filename = "src/" + metric + ".png"
    plt.savefig(filename)

"""# Example plots:
plot_metric_with_ci(ci_df, "precision", Ks=(10,20))
plot_metric_with_ci(ci_df, "recall", Ks=(10,20))
plot_metric_with_ci(ci_df, "ndcg", Ks=(10,20))"""



# app.py
import os
import math
import time
import joblib
import numpy as np
import pandas as pd
import requests
from functools import lru_cache
from typing import List, Dict, Optional, Tuple

import gradio as gr
from PIL import Image
from io import BytesIO

# -------------------------
# Configuration: local artifact paths
# -------------------------
ARTIFACT_DIR = "artifacts"
ITEM_VECTORS_PATH = os.path.join(ARTIFACT_DIR, "item_vectors.npz")   # .npz or joblib
ITEM_INDEX_PATH   = os.path.join(ARTIFACT_DIR, "item_index.pkl")     # list of tmdbId (ints) matching rows of item_vectors
TITLE_BY_ID_PATH  = os.path.join(ARTIFACT_DIR, "title_by_id.pkl")
GENRES_BY_ID_PATH = os.path.join(ARTIFACT_DIR, "genres_by_id.pkl")
CAST_BY_ID_PATH   = os.path.join(ARTIFACT_DIR, "cast_by_id.pkl")
IID_MAP_PATH      = os.path.join(ARTIFACT_DIR, "iid_map.pkl")        # optional: CF item index -> tmdbId
UID_MAP_PATH      = os.path.join(ARTIFACT_DIR, "uid_map.pkl")        # optional: user id -> idx
MF_PARAMS_PATH    = os.path.join(ARTIFACT_DIR, "mf_params.npz")      # contains mu, bu, bi, P, Q
POSTERS_PATH      = "meta_posters.csv"                              # optional small csv

# Placeholder poster (small local file or remote)
PLACEHOLDER_URL = "https://via.placeholder.com/180x270?text=No+Image"

# -------------------------
# Utilities: load artifacts lazily
# -------------------------
@lru_cache(maxsize=1)
def load_item_vectors():
    """Load item vectors as numpy array and index list. Accept .npz or joblib."""
    if os.path.exists(ITEM_VECTORS_PATH):
        ext = os.path.splitext(ITEM_VECTORS_PATH)[1].lower()
        if ext == ".npz":
            arr = np.load(ITEM_VECTORS_PATH)
            # Expect saved as arr["X"]
            if "X" in arr:
                X = arr["X"]
            else:
                # fallback: first array
                X = arr[list(arr.keys())[0]]
            # load index list separately
        else:
            data = joblib.load(ITEM_VECTORS_PATH)
            # accept dict or array
            if isinstance(data, dict) and "X" in data:
                X = data["X"]
            else:
                X = np.array(data)
    else:
        raise FileNotFoundError(f"{ITEM_VECTORS_PATH} not found in repo. Precompute and push it.")

    if os.path.exists(ITEM_INDEX_PATH):
        idx = joblib.load(ITEM_INDEX_PATH)
        idx = [int(x) for x in idx]
    else:
        raise FileNotFoundError("item_index.pkl not found. It should be list of tmdbId in same row order.")

    return X, idx

@lru_cache(maxsize=1)
def load_lookups():
    title_by_id = joblib.load(TITLE_BY_ID_PATH) if os.path.exists(TITLE_BY_ID_PATH) else {}
    genres_by_id = joblib.load(GENRES_BY_ID_PATH) if os.path.exists(GENRES_BY_ID_PATH) else {}
    cast_by_id = joblib.load(CAST_BY_ID_PATH) if os.path.exists(CAST_BY_ID_PATH) else {}
    iid_map = joblib.load(IID_MAP_PATH) if os.path.exists(IID_MAP_PATH) else None
    uid_map = joblib.load(UID_MAP_PATH) if os.path.exists(UID_MAP_PATH) else None
    return title_by_id, genres_by_id, cast_by_id, iid_map, uid_map

@lru_cache(maxsize=1)
def load_mf_params():
    if not os.path.exists(MF_PARAMS_PATH):
        return None
    arr = np.load(MF_PARAMS_PATH, allow_pickle=True)
    # Expect keys: mu, bu, bi, P, Q (P: users x f, Q: items x f)
    mu = float(arr["mu"].item()) if "mu" in arr else float(arr["mu"])
    bu = arr["bu"]
    bi = arr["bi"]
    P = arr["P"]
    Q = arr["Q"]
    return mu, bu, bi, P, Q

@lru_cache(maxsize=1)
def load_posters():
    if os.path.exists(POSTERS_PATH):
        df = pd.read_csv(POSTERS_PATH)
        # Expect columns tmdbId, poster_url
        return dict(zip(df["tmdbId"].astype(int), df["poster_url"].astype(str)))
    return {}

# -------------------------
# Helper: poster fetcher (simple, caches images)
# -------------------------
@lru_cache(maxsize=512)
def fetch_poster(url: str, resize=(180, 270)) -> Image.Image:
    if not url:
        url = PLACEHOLDER_URL
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")
        if resize:
            img = img.resize(resize)
        return img
    except Exception:
        # return placeholder image
        try:
            r = requests.get(PLACEHOLDER_URL, timeout=3)
            return Image.open(BytesIO(r.content)).convert("RGB").resize(resize)
        except Exception:
            # blank PIL image fallback
            return Image.new("RGB", resize, color=(200,200,200))

# -------------------------
# Recommendation algorithms (inference-only)
# -------------------------
from sklearn.metrics.pairwise import cosine_similarity

# Load global artifacts
ITEM_VECTORS, ITEM_INDEX = load_item_vectors()    # ITEM_VECTORS: (n_items, d), ITEM_INDEX: [tmdbId]
TITLE_BY_ID, GENRES_BY_ID, CAST_BY_ID, IID_MAP, UID_MAP = load_lookups()
POSTERS_DICT = load_posters()
MF_PARAMS = load_mf_params()

# Build index maps for quick lookup
tmdb_to_row = {int(tmdb): i for i, tmdb in enumerate(ITEM_INDEX)}
row_to_tmdb = {i: int(tmdb) for i, tmdb in enumerate(ITEM_INDEX)}
n_items = ITEM_VECTORS.shape[0]

# Basic CB scoring from Task 3: build user profile (rating-weighted average)
def build_profile_from_history(rated_tmdb: List[int], ratings: List[float]) -> Optional[np.ndarray]:
    """
    rated_tmdb: list of tmdbIds the user has rated
    ratings: list of corresponding numeric ratings
    Returns: profile vector (d,)
    """
    # align to item vectors we have
    rows = []
    ws = []
    for tmdb, r in zip(rated_tmdb, ratings):
        if tmdb in tmdb_to_row:
            rows.append(tmdb_to_row[tmdb])
            ws.append(r)
    if len(rows) == 0:
        return None
    X = ITEM_VECTORS[np.array(rows)]
    # mean-center the weights (user mean)
    ws = np.array(ws, dtype=float)
    ws = ws - ws.mean()
    if np.allclose(np.abs(ws).sum(), 0.0):
        return None
    profile = np.average(X, axis=0, weights=ws)
    return profile

def cb_score_from_profile(profile: np.ndarray) -> np.ndarray:
    """Return vector length n_items of cosine similarities (not masked)."""
    if profile is None:
        return np.full(n_items, -np.inf)
    sims = cosine_similarity(profile.reshape(1, -1), ITEM_VECTORS).ravel()
    return sims

def mf_score_for_user_raw(user_idx: int) -> Optional[np.ndarray]:
    """
    If you saved MF parameters, compute predicted score for all items: mu + bu[u] + bi + p_u @ Q.T
    user_idx should index into P (0..n_users-1)
    """
    if MF_PARAMS is None:
        return None
    mu, bu, bi, P, Q = MF_PARAMS
    if user_idx < 0 or user_idx >= P.shape[0]:
        return None
    s = mu + bu[user_idx] + bi + P[user_idx].dot(Q.T)
    return s

# -------------------------
# Merge scores and produce top-k with explanation
# -------------------------
def make_explanation(tmdb_id:int, user_history_tmdb:List[int], user_history_ratings:List[float]) -> str:
    # genres overlap
    rec_genres = set(GENRES_BY_ID.get(int(tmdb_id), []))
    liked_genres = set()
    liked_cast = set()
    for mid in user_history_tmdb:
        liked_genres |= set(GENRES_BY_ID.get(int(mid), []))
        liked_cast |= set(CAST_BY_ID.get(int(mid), []))
    g_overlap = sorted(rec_genres & liked_genres)
    # cast overlap
    rec_cast = set(CAST_BY_ID.get(int(tmdb_id), []))
    c_overlap = sorted(rec_cast & liked_cast)
    parts = []
    if g_overlap:
        parts.append("shares genres " + ", ".join(g_overlap[:3]))
    if c_overlap:
        parts.append("features cast " + ", ".join(c_overlap[:3]))
    return " and ".join(parts) if parts else "matches your taste profile"

def blend_scores(s_cf:np.ndarray, s_cb:np.ndarray, alpha:float):
    """Safely normalize (min-max) each score vector, handle -inf, then blend."""
    def normalize(v):
        v = v.copy().astype(float)
        finite = np.isfinite(v)
        if finite.sum()==0:
            return np.full_like(v, -np.inf)
        lo, hi = v[finite].min(), v[finite].max()
        if hi - lo < 1e-12:
            v[finite] = 0.5
        else:
            v[finite] = (v[finite] - lo) / (hi - lo)
        v[~finite] = -np.inf
        return v
    a = normalize(s_cf) if s_cf is not None else None
    b = normalize(s_cb) if s_cb is not None else None
    if a is None and b is None:
        return np.full(n_items, -np.inf)
    if a is None:
        return b
    if b is None:
        return a
    # replace any nonfinite with -inf before multiply
    a[~np.isfinite(a)] = -np.inf
    b[~np.isfinite(b)] = -np.inf
    # blend, guard NaNs
    out = alpha * a + (1.0 - alpha) * b
    out[~np.isfinite(out)] = -np.inf
    return out

# -------------------------
# Main recommend function used by Gradio
# -------------------------
def recommend(
    user_id_input: str,
    picked_title: str,
    mode: str = "Hybrid (MF+CB)",
    alpha: float = 0.6,
    top_k: int = 10
):
    """
    user_id_input: string of form "user:123" or can be left blank
    picked_title: title string (optional) — if provided will be used for seed (cold-start)
    mode: "CF (MF)", "CB", or "Hybrid (MF+CB)"
    """
    # Build minimal user profile from input:
    # For deployment we assume you will precompute and store per-user histories,
    # but here we accept a short manual seed:
    # user_id_input expected as "123" (raw user id) or blank.
    user_history_tmdb = []
    user_history_ratings = []
    # If user_id_input maps to a stored UID_MAP (optional):
    uid = None
    if user_id_input:
        try:
            uid = int(user_id_input)
        except Exception:
            uid = None

    # If user provided a picked_title, use that as single positive signal (cold-start shortcut)
    if picked_title:
        # try to map title -> tmdbId via TITLE_BY_ID reverse lookup
        picked_tmdb = None
        for k,v in TITLE_BY_ID.items():
            if isinstance(v, str) and v.lower() == picked_title.lower():
                picked_tmdb = int(k)
                break
        if picked_tmdb is not None:
            user_history_tmdb = [picked_tmdb]
            user_history_ratings = [5.0]   # assume enthusiastic rating
    # Otherwise if UID_MAP present and we store per-user histories in artifacts, load them (optional)
    # (You can enhance this part to load a per-user history file on disk.)
    # For now, if user provided numeric uid and UID_MAP exists with histories, try load:
    if UID_MAP and uid is not None:
        # If UID_MAP maps to index, we might have user histories file; skip by default
        pass

    # Compute CB profile and scores
    if user_history_tmdb:
        profile = build_profile_from_history(user_history_tmdb, user_history_ratings)
        s_cb = cb_score_from_profile(profile)  # length n_items
    else:
        s_cb = None

    # Compute CF score (MF) if we have MF params and UID_MAP maps user to index
    s_cf = None
    if MF_PARAMS and uid is not None and UID_MAP:
        # attempt to get user index in trained P
        try:
            user_idx = UID_MAP.get(uid, None)
            if user_idx is not None:
                s_cf = mf_score_for_user_raw(user_idx)
        except Exception:
            s_cf = None

    # Choose mode
    if mode.lower().startswith("cb"):
        final_scores = s_cb if s_cb is not None else (s_cf if s_cf is not None else np.full(n_items, -np.inf))
    elif mode.lower().startswith("cf"):
        final_scores = s_cf if s_cf is not None else (s_cb if s_cb is not None else np.full(n_items, -np.inf))
    else:
        final_scores = blend_scores(s_cf, s_cb, alpha)

    # Mask out -inf or missing
    finite_mask = np.isfinite(final_scores)
    if not finite_mask.any():
        # fallback to popularity top-k (global_wr must exist in artifacts)
        # Try to load global_wr from artifacts if present
        try:
            global_wr = joblib.load(os.path.join(ARTIFACT_DIR, "global_wr.pkl"))
            top = list(global_wr.head(top_k)["tmdbId"].astype(int))
            rows = []
            for tmdb in top:
                rows.append({
                    "title": TITLE_BY_ID.get(tmdb, str(tmdb)),
                    "tmdbId": int(tmdb),
                    "score": float(global_wr.loc[global_wr["tmdbId"]==tmdb, "WR"].iloc[0]) if "WR" in global_wr.columns else 0.0,
                    "explanation": "popularity fallback",
                    "poster": POSTERS_DICT.get(int(tmdb), PLACEHOLDER_URL)
                })
            return pd.DataFrame(rows)
        except Exception:
            return pd.DataFrame([])

    # Get top-k indices
    cand_idx = np.where(finite_mask)[0]
    topk_idx = cand_idx[np.argsort(-final_scores[cand_idx])][:top_k]

    rows = []
    for idx in topk_idx:
        tmdb = row_to_tmdb[idx]
        title = TITLE_BY_ID.get(tmdb, str(tmdb))
        explanation = make_explanation(tmdb, user_history_tmdb, user_history_ratings)
        poster_url = POSTERS_DICT.get(int(tmdb), PLACEHOLDER_URL)
        rows.append({
            "title": title,
            "tmdbId": int(tmdb),
            "score": float(final_scores[idx]),
            "explanation": explanation,
            "poster": poster_url
        })
    return pd.DataFrame(rows)

# -------------------------
# Gradio UI
# -------------------------
def build_interface():
    with gr.Blocks(css=".result-card {display:flex; gap:8px;}") as demo:
        gr.Markdown("## Movie Recommender — Demo Space\nChoose a model, provide a title or a user id, tune α, and get top-k recommendations with explanations and posters.")
        with gr.Row():
            with gr.Column(scale=2):
                user_id = gr.Textbox(label="User ID (optional)", placeholder="Type numeric user id (e.g. 123) or leave blank for cold-start")
                title_picker = gr.Dropdown(choices=sorted(list(set(TITLE_BY_ID.values()))), label="Pick a seed title (optional)", searchable=True)
                model_radio = gr.Radio(choices=["Hybrid (MF+CB)", "CF (MF)", "CB"], value="Hybrid (MF+CB)", label="Model")
                alpha_slider = gr.Slider(0.0, 1.0, value=0.6, step=0.05, label="α (CF weight in hybrid)")
                topk_slider = gr.Slider(1, 50, value=10, step=1, label="Top-K")
                run = gr.Button("Recommend")
                info = gr.Markdown("Tips: If you provide a title we treat it as a strong positive signal. For a real-deployment you'd wire a user's history file.")

            with gr.Column(scale=3):
                gallery = gr.Dataframe(headers=["title","tmdbId","score","explanation","poster"], datatype=["str","number","number","str","str"], interactive=False, label="Recommendations")
                # Alternative nicer display: use a custom HTML component per item (omitted for brevity)

        def on_run(uid, picked, model, alpha, topk):
            df = recommend(uid, picked, mode=model, alpha=alpha, top_k=int(topk))
            # Format: show poster URLs in table (Gradio will render string)
            if df.empty:
                return pd.DataFrame([])
            return df[["title","tmdbId","score","explanation","poster"]]

        run.click(on_run, inputs=[user_id, title_picker, model_radio, alpha_slider, topk_slider], outputs=[gallery])

        gr.Markdown("### About\nThis demo loads precomputed item vectors + models on startup. Keep large files in `artifacts/` and use compressed `.npz` or `joblib` formats.")
    return demo

if __name__ == "__main__":
    demo = build_interface()
    demo.launch(server_name="0.0.0.0", share=False)



"""







ratings_small = r_full[["userId", "tmdbId", "rating", "timestamp"]].copy()
ratings_small["userId"] = ratings_small["userId"].astype(int)
ratings_small["tmdbId"] = ratings_small["tmdbId"].astype(int)
item_vectors = recom.get_item_vectors()
iid_map = knn.get_iid_map()
uid_inv = knn.get_uid_inv()
R = knn.get_R()
title_by_id = knn.get_titles()

hybrid = Hybrid(R, r_full, item_vectors, iid_map, uid_inv, global_wr, title_by_id)

# Tune alpha parameter
try:
    best_alpha = hybrid.tune_alpha(
        ratings_small,
        cf_method="mf",
        K=10,
        alpha_grid=[0.0, 0.25, 0.5, 0.6, 0.7, 0.75, 1.0]
    )
except Exception as e:
    print(f"Error during alpha tuning: {e}")
    best_alpha = 0.6  # Fallback value

# Generate recommendations for a user
print(f"\nTop-10 Hybrid (MF+CB) recommendations for user {user_id}:")
try:
    recommendations = hybrid.recommend_hybrid(user_id, alpha=best_alpha, cf_method="mf", k=10)
    print(recommendations)
except Exception as e:
    print(f"Error generating recommendations: {e}")
    
    
    
    
    
    
    
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