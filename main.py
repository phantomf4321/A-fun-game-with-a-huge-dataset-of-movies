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





"""# Apply to genres column
df_metadata['genres'] = df_metadata['genres'].apply(parse_json_column)

# Now you can extract genre names
df_metadata['genre_names'] = df_metadata['genres'].apply(lambda x: [g['name'] for g in x] if isinstance(x, list) else [])
print(df_metadata['genre_names'].head())"""