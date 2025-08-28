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
global_wr = baseline_res["global_wr"]
print("\nTop-10 MF recommendations:")
print(knn.recommend(user_id, "mf", 10, global_wr))

print("\nTop-10 Item–Item recommendations:")
print(knn.recommend(user_id, "itemknn", 10, global_wr))

print("\nTop-10 User–User recommendations:")
print(knn.recommend(user_id, "userknn", 10, global_wr))


ratings_small = r_full[["userId", "tmdbId", "rating", "timestamp"]].copy()
ratings_small["userId"] = ratings_small["userId"].astype(int)
ratings_small["tmdbId"] = ratings_small["tmdbId"].astype(int)
item_vectors = recom.get_item_vectors()
iid_map = knn.get_iid_map()
uid_inv = knn.get_uid_inv()
R = knn.get_R()

hybrid = Hybrid(R, r_full, item_vectors, iid_map, uid_inv, global_wr)

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
    print(f"Error generating recommendations: {e}")"

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