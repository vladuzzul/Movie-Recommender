import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

MOVIES_DATA_PATH = DATA_DIR / "movies.csv"
RATINGS_DATA_PATH = DATA_DIR / "ratings.csv"

def load_dataframes():
    movies_df = pd.read_csv(MOVIES_DATA_PATH)
    ratings_df = pd.read_csv(RATINGS_DATA_PATH)
    return movies_df, ratings_df

def save_dataframe(df, file_name):
    df.to_csv(file_name, index=False)

def main():
    movies_df, ratings_df = load_dataframes()
    print("Movies Dataframe:")
    print("Shape before filtering NA data:", movies_df.shape)
    movies_df = movies_df.dropna()
    print("Shape after filtering NA data:", movies_df.shape)
    print("Primele randuri din dataframe:", movies_df.head(5), sep='\n')

    print("\nRatings Dataframe:")
    print("Shape before filtering NA data:", ratings_df.shape)
    ratings_df = ratings_df.dropna()
    print("Shape after filtering NA data:", ratings_df.shape)
    print("Primele randuri din dataframe:", ratings_df.head(5), sep='\n')
    num_users = ratings_df["userId"].nunique()
    print("Numar de useri:", num_users)

    save_dataframe(movies_df, MOVIES_DATA_PATH)
    save_dataframe(ratings_df, RATINGS_DATA_PATH)
    

if __name__ == "__main__":
    main()