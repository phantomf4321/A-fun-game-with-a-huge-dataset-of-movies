from app.functions import Datasets, Plot

plot = Plot()

meta = Datasets("data/movies_metadata.csv")
metadata_df = meta.get_dataframe()

ratings = Datasets("data/ratings_small.csv")
ratings_df = ratings.get_dataframe()

links = Datasets("data/ratings_small.csv")
links_df = links.get_dataframe()

# Load with low_memory=False to avoid dtype guessing issues
meta.log_step("load_raw",
         ratings_rows=len(ratings_df),
         links_rows=len(links_df),
         meta_rows=len(metadata_df))


#rates = ratings.get_dataframe_col("rating")
#plot.save_histogram(rates, "rate", "Small_rate_histogram")



"""# Apply to genres column
df_metadata['genres'] = df_metadata['genres'].apply(parse_json_column)

# Now you can extract genre names
df_metadata['genre_names'] = df_metadata['genres'].apply(lambda x: [g['name'] for g in x] if isinstance(x, list) else [])
print(df_metadata['genre_names'].head())"""