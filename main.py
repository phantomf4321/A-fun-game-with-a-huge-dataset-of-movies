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
          .set_index("tmdbId")["genres_names"]
)

print(movie_stats)
