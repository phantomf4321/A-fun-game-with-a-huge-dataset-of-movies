import numpy as np
import pandas as pd

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
    sim_df["title"] = sim_df["tmdbId"].map(meta_subset.set_index("id")["title"])
    return sim_df.sort_values("similarity", ascending=False).head(top_n)

# =============================
# 5) Natural-language explanations
# =============================
def explain_recommendation(tmdbId, user_id, ratings_df, meta_df):
    # Find top overlapping genres and cast with items user liked
    rec_genres = set(meta_df.loc[meta_df["id"] == tmdbId, "genres"].values[0] or [])
    rec_cast = set(meta_df.loc[meta_df["id"] == tmdbId, "top_cast"].values[0] or [])

    # Get user's highly rated movies
    liked = ratings_df[(ratings_df["userId"] == user_id) & (ratings_df["rating"] >= 4.0)]
    liked_ids = liked["tmdbId"].tolist()

    liked_genres = set()
    liked_cast = set()
    for mid in liked_ids:
        liked_genres.update(meta_df.loc[meta_df["id"] == mid, "genres"].values[0] or [])
        liked_cast.update(meta_df.loc[meta_df["id"] == mid, "top_cast"].values[0] or [])

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