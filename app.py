import kagglehub

# Download latest version
path = kagglehub.dataset_download("https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset")

print("Path to dataset files:", path)