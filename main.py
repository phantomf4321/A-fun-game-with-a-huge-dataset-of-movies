from app.functions import *

plot = Plot()
GO = General_Operations()

meta = Datasets("data/movies_metadata.csv")
metadata_df = meta.get_dataframe()

ratings = Datasets("data/ratings_small.csv")
ratings_df = ratings.get_dataframe()

links = Datasets("data/links.csv")
links_df = links.get_dataframe()

# Load with low_memory=False to avoid dtype guessing issues
GO.log_step("load_raw",
         ratings_rows=len(ratings_df),
         links_rows=len(links_df),
         meta_rows=len(metadata_df))


# Ensure numeric IDs
links_df["tmdbId"] = pd.to_numeric(links_df["tmdbId"], errors="coerce")
links_df["movieId"] = pd.to_numeric(links_df["movieId"], errors="coerce")

metadata_df["id"] = pd.to_numeric(metadata_df["id"], errors="coerce")

# Keep valid rows only
links_clean = links_df.dropna(subset=["movieId", "tmdbId"]).astype({"movieId": int, "tmdbId": int})
meta_clean  = metadata_df.dropna(subset=["id"])

GO.log_step("clean_ids",
         links_clean_rows=len(links_clean),
         meta_clean_rows=len(meta_clean))

# Join ratings → links → metadata (document row counts)
# ratings ⨝ links (movieId)
r_links = ratings_df.merge(links_clean[["movieId","tmdbId"]], on="movieId", how="inner")
GO.log_step("ratings_join_links",
         ratings_before=len(ratings_df),
         r_links_rows=len(r_links))

# (ratings⨝links) ⨝ metadata (tmdbId == id)
r_full = r_links.merge(meta_clean[["id","title","original_language","genres","production_countries","release_date"]],
                       left_on="tmdbId", right_on="id", how="inner")

GO.log_step("join_with_metadata",
         r_links_before=len(r_links),
         r_full_rows=len(r_full))


# Minimal filtering for valid analysis + row counts
# Convert timestamp and release_date safely
r_full["timestamp"] = pd.to_numeric(r_full["timestamp"], errors="coerce")
r_full = r_full.dropna(subset=["timestamp"])

# release_date can be NaT; keep it but parse
r_full["release_date"] = pd.to_datetime(r_full["release_date"], errors="coerce")

# Keep ratings within common bounds if needed (0.5..5.0); dataset sometimes has 0..5
r_full = r_full[(r_full["rating"] >= 0.5) & (r_full["rating"] <= 5.0)]

GO.log_step("filter_valid_rows",
         r_full_rows=len(r_full),
         unique_users=r_full["userId"].nunique(),
         unique_movies=r_full["movieId"].nunique())


# Summary stats
rating_desc = r_full["rating"].describe()
print("\nRating summary:\n", rating_desc)

plot.save_histogram(r_full, "rating", "Rating Distribution", "Small_rate_histogram")


# User activity (ratings per user)
user_cnt = r_full.groupby("userId")["movieId"].count().rename("n_ratings").reset_index()
movie_cnt = r_full.groupby("movieId")["userId"].count().rename("n_ratings").reset_index()

GO.log_step("long_tail_sizes",
         users=len(user_cnt), movies=len(movie_cnt),
         median_user_ratings=int(user_cnt["n_ratings"].median()),
         median_movie_ratings=int(movie_cnt["n_ratings"].median()))


plot.save_log_log_histogram(user_cnt, "n_ratings", "Ratings per user (log)", "Long Tail: User Activity", "Long Tail: User Activity", "Long_Tail_User_Activity")
plot.save_log_log_histogram(movie_cnt, "n_ratings", "Ratings per movie (log)", "", "Long Tail: Movie Popularity", "Long_Tail_Movie_Popularity")

print("Share of all ratings by top 10% users:",
      round(GO.cumulative_coverage(user_cnt, 0.10), 3))
print("Share of all ratings for top 10% movies:",
      round(GO.cumulative_coverage(movie_cnt, 0.10), 3))

n_users = r_full["userId"].nunique()
n_movies = r_full["movieId"].nunique()
nnz = len(r_full)  # number of observed ratings
sparsity = 1 - nnz / (n_users * n_movies)
print(f"Sparsity (overall): {sparsity:.6f} (1 − nnz/(U×M))")

# Heatmap of a manageable submatrix:
# take top-N users & movies by activity to keep the plot readable
TOP_USERS = 200
TOP_MOVIES = 200

top_users = user_cnt.sort_values("n_ratings", ascending=False).head(TOP_USERS)["userId"]
top_movies = movie_cnt.sort_values("n_ratings", ascending=False).head(TOP_MOVIES)["movieId"]

sub = r_full[r_full["userId"].isin(top_users) & r_full["movieId"].isin(top_movies)]
pivot = sub.pivot_table(index="userId", columns="movieId", values="rating", aggfunc="mean")

# Show as a presence/absence heatmap (binary mask) to emphasize sparsity pattern
present = (~pivot.isna()).astype(int)
plot.save_heatmap(present, "user", "movie", TOP_USERS, TOP_MOVIES, "user_movie_heatmap")

# Convert unix seconds to datetime
r_full["date"] = pd.to_datetime(r_full["timestamp"], unit="s", errors="coerce")

# Ratings per month (volume) and monthly average rating
monthly = (r_full
           .set_index("date")
           .sort_index()
           .resample("MS")
           .agg(n_ratings=("rating","count"),
                avg_rating=("rating","mean"))
           .dropna(subset=["n_ratings"]))
print("\nMonthly coverage rows:", len(monthly))

plot.save_simple_plot(monthly, "n_ratings", "Month", "Number of ratings", "Ratings Volume Over Time (Monthly)", "Ratings_Volume_Over_Time(Monthly)")
plot.save_simple_plot(monthly, "avg_rating", "Month", "Average of ratings", "Ratings Average Over Time (Monthly)", "Ratings_Average_Over_Time(Monthly)")


# Parse nested fields
meta_use = meta_clean[["id","title","original_language","genres","production_countries","spoken_languages"]].copy()

for col in ["genres","production_countries","spoken_languages"]:
    meta_use[col] = meta_use[col].apply(GO.tidy_json_list)

# Explode genres
genres_exploded = (meta_use
                   .explode("genres", ignore_index=True))
genres_exploded["genre_name"] = genres_exploded["genres"].apply(
    lambda d: d.get("name") if isinstance(d, dict) else (d if isinstance(d, str) else np.nan)
)
genre_counts = (genres_exploded
                .dropna(subset=["genre_name"])
                .groupby("genre_name")["id"].nunique()
                .sort_values(ascending=False))

print("\nTop 15 genres by #movies:\n", genre_counts.head(15))

# Languages (original_language)
lang_counts = (meta_use
               .dropna(subset=["original_language"])
               .groupby("original_language")["id"].nunique()
               .sort_values(ascending=False))

print("\nTop 15 original languages by #movies:\n", lang_counts.head(15))

# Countries (production_countries)
countries_exploded = meta_use.explode("production_countries", ignore_index=True)
countries_exploded["country_name"] = countries_exploded["production_countries"].apply(
    lambda d: d.get("name") if isinstance(d, dict) else (d if isinstance(d, str) else np.nan)
)
country_counts = (countries_exploded
                  .dropna(subset=["country_name"])
                  .groupby("country_name")["id"].nunique()
                  .sort_values(ascending=False))

print("\nTop 15 production countries by #movies:\n", country_counts.head(15))

plot.save_bar(genre_counts, "# Movies", "Genre", "Top Genres by # Movies", "Top_Genres_by_Movies")
plot.save_bar(genre_counts, "# Movies", "Language", "Top Languages by # Movies", "Top_Languages_by_Movies")


audit = pd.DataFrame(GO.get_logs())
print("\n=== Row-count audit trail ===")
print(audit.to_string(index=False))


# Save audit, summaries
audit.to_csv("data/clean/eda_rowcount_audit.csv", index=False)
user_cnt.to_csv("data/clean/eda_user_activity_counts.csv", index=False)
movie_cnt.to_csv("data/clean/eda_movie_popularity_counts.csv", index=False)
genre_counts.to_csv("data/clean/eda_genre_counts.csv")
lang_counts.to_csv("data/clean/eda_language_counts.csv")
country_counts.to_csv("data/clean/eda_country_counts.csv")
monthly.to_csv("data/clean/eda_temporal_monthly.csv")
