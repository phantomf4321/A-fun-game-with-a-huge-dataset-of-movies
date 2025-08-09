import pandas as pd
import ast

# Load the data
df = pd.read_csv("data/movies_metadata.csv")

# Function to safely parse JSON/string-represented lists
def parse_json_column(col):
    try:
        # First try proper JSON parsing
        return pd.io.json.loads(col)
    except:
        try:
            # If that fails, try literal_eval which handles Python-style strings
            return ast.literal_eval(col)
        except:
            # If all fails, return empty list
            return []

# Apply to genres column
df['genres'] = df['genres'].apply(parse_json_column)

# Now you can extract genre names
df['genre_names'] = df['genres'].apply(lambda x: [g['name'] for g in x] if isinstance(x, list) else [])
print(df['genre_names'].head())