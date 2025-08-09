from app.functions import Datasets
import matplotlib.pyplot as plt


metadata = Datasets("data/movies_metadata.csv")
metadata_df = metadata.get_dataframe()

ratings = Datasets("data/ratings_small.csv")
ratings_df = ratings.get_dataframe()
rates = ratings.get_dataframe_col("rating")

plt.hist(rates)
plt.savefig('output.png')


"""# Apply to genres column
df_metadata['genres'] = df_metadata['genres'].apply(parse_json_column)

# Now you can extract genre names
df_metadata['genre_names'] = df_metadata['genres'].apply(lambda x: [g['name'] for g in x] if isinstance(x, list) else [])
print(df_metadata['genre_names'].head())"""