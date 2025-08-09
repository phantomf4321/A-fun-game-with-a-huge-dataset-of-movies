from app.functions import Datasets

metadata = Datasets("data/movies_metadata.csv")
metadata_df = metadata.get_dataframe()

print(metadata_df.head())

"""# Apply to genres column
df_metadata['genres'] = df_metadata['genres'].apply(parse_json_column)

# Now you can extract genre names
df_metadata['genre_names'] = df_metadata['genres'].apply(lambda x: [g['name'] for g in x] if isinstance(x, list) else [])
print(df_metadata['genre_names'].head())"""