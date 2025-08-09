import pandas as pd
import json

df = pd.read_csv("data/movies_metadata.csv")
df['genres_id'] = df['genres'].str['id']
df['genres_name'] = df['genres'].str['name']
for raw in df['genres']:
    genres = [g["name"] for g in json.loads(raw)]
    print(genres)

