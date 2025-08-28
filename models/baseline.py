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

    def get_global_wr(self):
        return self.global_wr

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

        results = {
            "C": C,
            "m": m,
            "global_wr": global_wr
        }

        return results

    # --- Per-genre WR ---
    def genre_wr(self, group, Cg):
        """Calculate weighted rating for a genre group."""
        # Extract the genre name from the group (it's the index/group name)
        genre_name = group.name if hasattr(group, 'name') else None

        m_g = np.quantile(group["vi"], 0.80) if len(group) else 0
        C_g = Cg.get(genre_name, group["Ri"].mean())  # Use the actual genre name
        v = group["vi"]
        R = group["Ri"]
        group["WR_g"] = (v / (v + m_g)) * R + (m_g / (v + m_g)) * C_g
        group["m_g"] = m_g
        group["C_g"] = C_g
        return group[group["vi"] >= m_g].sort_values("WR_g", ascending=False)

    def Per_genre(self):
        """Calculate per-genre weighted ratings."""
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

        # Apply WR within each genre
        per_genre_wr = per_genre_df.groupby("genre", group_keys=False).apply(
            lambda group: self.genre_wr(group, Cg)
        )

        return per_genre_wr

    def final_results(self):

        res = self.top_10_globally_popular_movies()
        C = res["C"]
        m = res["m"]
        global_wr = res["global_wr"]

        print("Global WR parameters:")
        print({"C": round(C, 3), "m": int(m), "m_quantile": 0.80})
        print("\nTop 10 globally popular movies (WR):")
        print(global_wr[["title", "vi", "Ri", "WR"]].head(10))

        per_genre_wr = self.Per_genre()

        # Saving
        global_wr.to_csv("data/baseline/baseline_global_wr.csv", index=False)
        per_genre_wr.to_csv("data/baseline/baseline_per_genre_wr.csv", index=False)

        print("Baseline's results are saved successfully!")