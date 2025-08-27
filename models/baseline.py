class Baseline():
    def __init__(self, r_full):
        self.r_full = r_full
        print("Bsaseline is constructed...")

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