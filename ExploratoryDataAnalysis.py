from app.functions import *

print("========== EDA.py is running ==========")

plot = Plot()
GO = General_Operations()



class EDA:
    def __init__(self):
        print("========== EDA is running ==========")
        self.r_full = None
        self.cleaner()
        self.save_plots()
        print("========== EDA.py is ended ==========")

    def load_dataset(self):
        meta = Datasets("data/movies_metadata.csv")
        self.metadata_df = meta.get_dataframe()

        ratings = Datasets("data/ratings_small.csv")
        self.ratings_df = ratings.get_dataframe()

        links = Datasets("data/links.csv")
        self.links_df = links.get_dataframe()

        # Load with low_memory=False to avoid dtype guessing issues
        GO.log_step("load_raw",
                    ratings_rows=len(self.ratings_df),
                    links_rows=len(self.links_df),
                    meta_rows=len(self.metadata_df))

    def cleaner(self):
        print("===== clearing is started! =====")
        self.load_dataset()
        # Ensure numeric IDs
        self.links_df["tmdbId"] = pd.to_numeric(self.links_df["tmdbId"], errors="coerce")
        self.links_df["movieId"] = pd.to_numeric(self.links_df["movieId"], errors="coerce")

        self.metadata_df["id"] = pd.to_numeric(self.metadata_df["id"], errors="coerce")

        # Keep valid rows only
        self.links_clean = self.links_df.dropna(subset=["movieId", "tmdbId"]).astype({"movieId": int, "tmdbId": int})
        self.meta_clean = self.metadata_df.dropna(subset=["id"])

        GO.log_step("clean_ids",
                    links_clean_rows=len(self.links_clean),
                    meta_clean_rows=len(self.meta_clean))

        # Join ratings → links → metadata (document row counts)
        # ratings ⨝ links (movieId)
        self.r_links = self.ratings_df.merge(self.links_clean[["movieId", "tmdbId"]], on="movieId", how="inner")
        GO.log_step("ratings_join_links",
                    ratings_before=len(self.ratings_df),
                    r_links_rows=len(self.r_links))

        # (ratings⨝links) ⨝ metadata (tmdbId == id)
        self.r_full = self.r_links.merge(
            self.meta_clean[["id", "title", "original_language", "genres", "production_countries", "release_date"]],
            left_on="tmdbId", right_on="id", how="inner")

        GO.log_step("join_with_metadata",
                    r_links_before=len(self.r_links),
                    r_full_rows=len(self.r_full))

        # Minimal filtering for valid analysis + row counts
        # Convert timestamp and release_date safely
        self.r_full["timestamp"] = pd.to_numeric(self.r_full["timestamp"], errors="coerce")
        self.r_full = self.r_full.dropna(subset=["timestamp"])

        # release_date can be NaT; keep it but parse
        self.r_full["release_date"] = pd.to_datetime(self.r_full["release_date"], errors="coerce")

        # Keep ratings within common bounds if needed (0.5..5.0); dataset sometimes has 0..5
        self.r_full = self.r_full[(self.r_full["rating"] >= 0.5) & (self.r_full["rating"] <= 5.0)]

        GO.log_step("filter_valid_rows",
                    r_full_rows=len(self.r_full),
                    unique_users=self.r_full["userId"].nunique(),
                    unique_movies=self.r_full["movieId"].nunique())

        # Summary stats
        self.rating_desc = self.r_full["rating"].describe()
        print("\nRating summary:\n", self.rating_desc)

        # User activity (ratings per user)
        self.user_cnt = self.r_full.groupby("userId")["movieId"].count().rename("n_ratings").reset_index()
        self.movie_cnt = self.r_full.groupby("movieId")["userId"].count().rename("n_ratings").reset_index()

        GO.log_step("long_tail_sizes",
                    users=len(self.user_cnt), movies=len(self.movie_cnt),
                    median_user_ratings=int(self.user_cnt["n_ratings"].median()),
                    median_movie_ratings=int(self.movie_cnt["n_ratings"].median()))

        print("Share of all ratings by top 10% users:",
              round(GO.cumulative_coverage(self.user_cnt, 0.10), 3))
        print("Share of all ratings for top 10% movies:",
              round(GO.cumulative_coverage(self.movie_cnt, 0.10), 3))

        n_users = self.r_full["userId"].nunique()
        n_movies = self.r_full["movieId"].nunique()
        nnz = len(self.r_full)  # number of observed ratings
        sparsity = 1 - nnz / (n_users * n_movies)
        print(f"Sparsity (overall): {sparsity:.6f} (1 − nnz/(U×M))")

        # Heatmap of a manageable submatrix:
        # take top-N users & movies by activity to keep the plot readable
        self.TOP_USERS = 200
        self.TOP_MOVIES = 200

        self.top_users = self.user_cnt.sort_values("n_ratings", ascending=False).head(self.TOP_USERS)["userId"]
        self.top_movies = self.movie_cnt.sort_values("n_ratings", ascending=False).head(self.TOP_MOVIES)["movieId"]

        sub = self.r_full[self.r_full["userId"].isin(self.top_users) & self.r_full["movieId"].isin(self.top_movies)]
        pivot = sub.pivot_table(index="userId", columns="movieId", values="rating", aggfunc="mean")

        # Show as a presence/absence heatmap (binary mask) to emphasize sparsity pattern
        self.present = (~pivot.isna()).astype(int)

        # Convert unix seconds to datetime
        self.r_full["date"] = pd.to_datetime(self.r_full["timestamp"], unit="s", errors="coerce")

        # Ratings per month (volume) and monthly average rating
        self.monthly = (self.r_full
                   .set_index("date")
                   .sort_index()
                   .resample("MS")
                   .agg(n_ratings=("rating", "count"),
                        avg_rating=("rating", "mean"))
                   .dropna(subset=["n_ratings"]))
        print("\nMonthly coverage rows:", len(self.monthly))

        # Parse nested fields
        self.meta_use = self.meta_clean[
            ["id", "title", "original_language", "genres", "production_countries", "spoken_languages"]].copy()

        for col in ["genres", "production_countries", "spoken_languages"]:
            self.meta_use[col] = self.meta_use[col].apply(GO.tidy_json_list)

        # Explode genres
        self.genres_exploded = (self.meta_use
                           .explode("genres", ignore_index=True))
        self.genres_exploded["genre_name"] = self.genres_exploded["genres"].apply(
            lambda d: d.get("name") if isinstance(d, dict) else (d if isinstance(d, str) else np.nan)
        )
        self.genre_counts = (self.genres_exploded
                        .dropna(subset=["genre_name"])
                        .groupby("genre_name")["id"].nunique()
                        .sort_values(ascending=False))

        print("\nTop 15 genres by #movies:\n", self.genre_counts.head(15))

        # Languages (original_language)
        self.lang_counts = (self.meta_use
                       .dropna(subset=["original_language"])
                       .groupby("original_language")["id"].nunique()
                       .sort_values(ascending=False))

        print("\nTop 15 original languages by #movies:\n", self.lang_counts.head(15))

        # Countries (production_countries)
        self.countries_exploded = self.meta_use.explode("production_countries", ignore_index=True)
        self.countries_exploded["country_name"] = self.countries_exploded["production_countries"].apply(
            lambda d: d.get("name") if isinstance(d, dict) else (d if isinstance(d, str) else np.nan)
        )
        self.country_counts = (self.countries_exploded
                          .dropna(subset=["country_name"])
                          .groupby("country_name")["id"].nunique()
                          .sort_values(ascending=False))

        print("\nTop 15 production countries by #movies:\n", self.country_counts.head(15))

        self.audit = pd.DataFrame(GO.get_logs())
        print("\n=== Row-count audit trail ===")
        print(self.audit.to_string(index=False))

        # Save audit, summaries
        self.audit.to_csv("data/clean/eda_rowcount_audit.csv", index=False)
        self.user_cnt.to_csv("data/clean/eda_user_activity_counts.csv", index=False)
        self.movie_cnt.to_csv("data/clean/eda_movie_popularity_counts.csv", index=False)
        self.genre_counts.to_csv("data/clean/eda_genre_counts.csv")
        self.lang_counts.to_csv("data/clean/eda_language_counts.csv")
        self.country_counts.to_csv("data/clean/eda_country_counts.csv")
        self.monthly.to_csv("data/clean/eda_temporal_monthly.csv")

        print("===== clearing is ended! =====")

    def save_plots(self):
        print("===== Save plots is running! =====")
        plot.save_histogram(self.r_full, "rating", "Rating Distribution", "Small_rate_histogram")
        plot.save_log_log_histogram(self.user_cnt, "n_ratings", "Ratings per user (log)", "Long Tail: User Activity","Long Tail: User Activity", "Long_Tail_User_Activity")
        plot.save_log_log_histogram(self.movie_cnt, "n_ratings", "Ratings per movie (log)", "","Long Tail: Movie Popularity", "Long_Tail_Movie_Popularity")
        plot.save_heatmap(self.present, "user", "movie", self.TOP_USERS, self.TOP_MOVIES, "user_movie_heatmap")
        plot.save_simple_plot(self.monthly, "n_ratings", "Month", "Number of ratings", "Ratings Volume Over Time (Monthly)","Ratings_Volume_Over_Time(Monthly)")
        plot.save_simple_plot(self.monthly, "avg_rating", "Month", "Average of ratings","Ratings Average Over Time (Monthly)", "Ratings_Average_Over_Time(Monthly)")
        plot.save_bar(self.genre_counts, "# Movies", "Genre", "Top Genres by # Movies", "Top_Genres_by_Movies")
        plot.save_bar(self.genre_counts, "# Movies", "Language", "Top Languages by # Movies", "Top_Languages_by_Movies")
        print("===== Save plots is ended! =====")

    def get_r_full(self):
        return self.r_full
