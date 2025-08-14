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
