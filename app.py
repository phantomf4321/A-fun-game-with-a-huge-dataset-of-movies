import pandas as pd
import json

df = pd.read_csv("data/movies_metadata.csv")
df['genres_id'] = df['genres'].str['id']
df['genres_name'] = df['genres'].str['name']
for g in df['genres']:
    print(g)

