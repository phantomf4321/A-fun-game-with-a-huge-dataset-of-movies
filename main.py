import numpy as np
import pandas as pd

from ExploratoryDataAnalysis import EDA


eda = EDA()

# --- 1) Compute movie-level stats from cleaned ratings ---
movie_stats = (
    eda.r_full.groupby(["tmdbId", "title"], as_index=False)
          .agg(
              Ri=("rating", "mean"),   # mean rating
              vi=("rating", "count")   # vote count
          )
)

# attach genres list (first occurrence)
movie_stats["genres"] = movie_stats["tmdbId"].map(
    eda.r_full.drop_duplicates("tmdbId")
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
