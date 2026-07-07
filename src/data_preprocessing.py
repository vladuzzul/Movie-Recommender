import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MOVIES_DATA_PATH = DATA_DIR / "movies.csv"
RATINGS_DATA_PATH = DATA_DIR / "ratings.csv"
PROCESSED_MOVIES_DATA_PATH = PROCESSED_DATA_DIR / "movies.csv"
PROCESSED_RATINGS_DATA_PATH = PROCESSED_DATA_DIR / "ratings.csv"

TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train_data.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test_data.csv"
VAL_DATA_PATH = PROCESSED_DATA_DIR / "val_data.csv"

def load_dataframes():
    movies_df = pd.read_csv(MOVIES_DATA_PATH)
    ratings_df = pd.read_csv(RATINGS_DATA_PATH)
    return movies_df, ratings_df

def save_dataframe(df, file_name):
    file_name.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_name, index=False)

def data_analysis():
    movies_df, ratings_df = load_dataframes()
    print("Movies Dataframe:")
    print("Shape:", movies_df.shape)
    print("First rows of the dataframe:", movies_df.head(5), sep='\n')
    num_movies = movies_df["movieId"].nunique()
    print("Number of movies:", num_movies)

    print("\nRatings Dataframe:")
    print("Shape:", ratings_df.shape)
    print("First rows of the dataframe:", ratings_df.head(5), sep='\n')
    num_users = ratings_df["userId"].nunique()
    print("Number of users:", num_users)

def split_dataset(ratings_df):
    """Split the data 70% training, 15% test, 15% validation"""
    train_df, temp_df = train_test_split(
        ratings_df,
        test_size=0.3,
        random_state=42
    )

    test_df, val_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42
    )
    return train_df, test_df, val_df
    

    
def data_cleaning():
    movies_df, ratings_df = load_dataframes()

    min_user_ratings = 5
    min_movie_interactions = 5

    print("Initial shapes:")
    print("Movies:", movies_df.shape)
    print("Ratings:", ratings_df.shape)

    ratings_df = ratings_df.drop_duplicates()
    print("\nAfter removing duplicate ratings:", ratings_df.shape)

    changed = True
    while changed:
        previous_ratings_shape = ratings_df.shape
        previous_movies_shape = movies_df.shape

        user_counts = ratings_df["userId"].value_counts()
        active_users = user_counts[user_counts >= min_user_ratings].index
        ratings_df = ratings_df[ratings_df["userId"].isin(active_users)].copy()

        movie_counts = ratings_df["movieId"].value_counts()
        active_movies = movie_counts[movie_counts >= min_movie_interactions].index
        ratings_df = ratings_df[ratings_df["movieId"].isin(active_movies)].copy()
        movies_df = movies_df[movies_df["movieId"].isin(active_movies)].copy()

        if ratings_df["rating"].max() > 1:
            ratings_df["rating"] /= 5

        changed = ratings_df.shape != previous_ratings_shape or movies_df.shape != previous_movies_shape

    ratings_df = ratings_df.drop(columns=["timestamp", "rating_centered"])

    print("\nAfter filtering sparse users/movies:")
    print("Movies:", movies_df.shape)
    print("Ratings:", ratings_df.shape)
    print("Remaining users:", ratings_df["userId"].nunique())
    print("Remaining movies:", ratings_df["movieId"].nunique())

    print("\nRating distribution:")
    rating_distribution = ratings_df["rating"].value_counts().sort_index()
    rating_distribution_percent = ratings_df["rating"].value_counts(normalize=True).sort_index() * 100
    for rating_value in rating_distribution.index:
        count = int(rating_distribution.loc[rating_value])
        percent = rating_distribution_percent.loc[rating_value]
        print(f"Rating {rating_value * 5}: {count} ({percent:.2f}%)")

    save_dataframe(movies_df, PROCESSED_MOVIES_DATA_PATH)
    save_dataframe(ratings_df, PROCESSED_RATINGS_DATA_PATH)
    print("\nCleaned datasets saved to:")
    print(PROCESSED_MOVIES_DATA_PATH)
    print(PROCESSED_RATINGS_DATA_PATH)

    train_df, test_df, val_df = split_dataset(ratings_df)
    
    save_dataframe(train_df, TRAIN_DATA_PATH)
    save_dataframe(test_df, TEST_DATA_PATH)
    save_dataframe(val_df, VAL_DATA_PATH)
    print("\nSplit datasets saved to:")
    print(TRAIN_DATA_PATH)
    print(TEST_DATA_PATH)
    print(VAL_DATA_PATH)

if __name__ == "__main__":
    data_cleaning()
    data_analysis()