import numpy as np
import pandas as pd
from functools import partial
class Baseline():
    def __init__(self, r_full):
        self.r_full = r_full
        print("Bsaseline is constructed...")

        # --- Compute movie-level stats from cleaned ratings ---
        self.movie_stats = (
            self.r_full.groupby(["tmdbId", "title"], as_index=False)
            .agg(
                Ri=("rating", "mean"),  # mean rating
                vi=("rating", "count")  # vote count
            )
        )

        # attach genres list (first occurrence)
        self.movie_stats["genres"] = self.movie_stats["tmdbId"].map(
            self.r_full.drop_duplicates("tmdbId")
            .set_index("tmdbId")["genres"]
        )

    def top_10_globally_popular_movies(self):
        # --- 2) IMDb WR global baseline ---
        # Global C = weighted mean of all ratings
        C = (self.movie_stats["Ri"] * self.movie_stats["vi"]).sum() / self.movie_stats["vi"].sum()

        # m = 80th percentile of vote counts
        m = np.quantile(self.movie_stats["vi"], 0.80)

        # WR formula
        v = self.movie_stats["vi"]
        R = self.movie_stats["Ri"]
        self.movie_stats["WR"] = (v / (v + m)) * R + (m / (v + m)) * C

        # Keep movies with v >= m
        global_wr = self.movie_stats[self.movie_stats["vi"] >= m].copy()
        global_wr = global_wr.sort_values("WR", ascending=False)

        resaults = {
            "C": C,
            "m": m,
            "global_wr": global_wr
        }

        return resaults

    # --- Per-genre WR ---
    def Per_genre(self):
        # --- Per-genre WR ---
        rows = []
        for _, row in self.movie_stats.iterrows():
            genres = row["genres"] if isinstance(row["genres"], list) and row["genres"] else ["(No Genre)"]
            for g in genres:
                rows.append((g, row["tmdbId"], row["title"], row["vi"], row["Ri"]))
        per_genre_df = pd.DataFrame(rows, columns=["genre", "tmdbId", "title", "vi", "Ri"])

        # For C_g, use all ratings in r_full for that genre
        exploded = self.r_full.explode("genres").rename(columns={"genres": "genre"})
        Cg = exploded.groupby("genre")["rating"].mean()

        # Apply WR within each genre - FIXED: use partial to bind Cg argument
        genre_wr_with_cg = partial(self.genre_wr, Cg=Cg)
        per_genre_wr = per_genre_df.groupby("genre", group_keys=False).apply(genre_wr_with_cg)

        return per_genre_wr