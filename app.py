import pandas as pd
import ast

# Load the data
df_metadata = pd.read_csv("data/movies_metadata.csv")
df_rating = pd.read_csv("data/ratings_small.csv")

# Function to safely parse JSON/string-represented lists

# Apply to genres column
df_metadata['genres'] = df_metadata['genres'].apply(parse_json_column)

# Now you can extract genre names
df_metadata['genre_names'] = df_metadata['genres'].apply(lambda x: [g['name'] for g in x] if isinstance(x, list) else [])
print(df_metadata['genre_names'].head())