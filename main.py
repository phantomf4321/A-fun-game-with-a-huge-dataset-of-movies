import numpy as np
import pandas as pd

from ExploratoryDataAnalysis import EDA


# --- 1) Compute movie-level stats from cleaned ratings ---
movie_stats = (
    r_full.groupby(["tmdbId", "title"], as_index=False)
          .agg(
              Ri=("rating", "mean"),   # mean rating
              vi=("rating", "count")   # vote count
          )
)
