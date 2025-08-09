import pandas as pd
import ast

class Datasets:
    def __init__(self, directory):
        self.df = pd.read_csv(directory)
        print("Dataset constructor is called for {}".format(directory))
        return self.df

    def get_dataframe(self):
        return self.df

    def get_dataframe_col(self, col):
        return self.df[col]
    def parse_json_column(self, col):
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