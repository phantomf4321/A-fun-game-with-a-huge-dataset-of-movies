from app.functions import *

plot = Plot()
GO = General_Operations()

meta = Datasets("data/movies_metadata.csv")
metadata_df = meta.get_dataframe()

ratings = Datasets("data/ratings_small.csv")
ratings_df = ratings.get_dataframe()

links = Datasets("data/ratings_small.csv")
links_df = links.get_dataframe()

# Load with low_memory=False to avoid dtype guessing issues
GO.log_step("load_raw",
         ratings_rows=len(ratings_df),
         links_rows=len(links_df),
         meta_rows=len(metadata_df))


# Ensure numeric IDs
links_df["tmdbId"] = pd.to_numeric(links["tmdbId"], errors="coerce")
links_df["movieId"] = pd.to_numeric(links["movieId"], errors="coerce")

metadata_df["id"] = pd.to_numeric(meta["id"], errors="coerce")

# Keep valid rows only
links_clean = links_df.dropna(subset=["movieId", "tmdbId"]).astype({"movieId": int, "tmdbId": int})
meta_clean  = metadata_df.dropna(subset=["id"])

GO.log_step("clean_ids",
         links_clean_rows=len(links_clean),
         meta_clean_rows=len(meta_clean))



#rates = ratings.get_dataframe_col("rating")
#plot.save_histogram(rates, "rate", "Small_rate_histogram")



"""# Apply to genres column
df_metadata['genres'] = df_metadata['genres'].apply(parse_json_column)

# Now you can extract genre names
df_metadata['genre_names'] = df_metadata['genres'].apply(lambda x: [g['name'] for g in x] if isinstance(x, list) else [])
print(df_metadata['genre_names'].head())"""