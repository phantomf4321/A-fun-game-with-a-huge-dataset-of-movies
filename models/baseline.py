import numpy as np
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