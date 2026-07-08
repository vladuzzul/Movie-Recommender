import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MOVIES_DATA_PATH = DATA_DIR / "movies.csv"
RATINGS_DATA_PATH = DATA_DIR / "ratings.csv"
PROCESSED_MOVIES_DATA_PATH = PROCESSED_DATA_DIR / "movies.csv"
PROCESSED_RATINGS_DATA_PATH = PROCESSED_DATA_DIR / "ratings.csv"

def load_dataframes():
    movies_df = pd.read_csv(MOVIES_DATA_PATH)
    ratings_df = pd.read_csv(RATINGS_DATA_PATH)
    return movies_df, ratings_df

def load_processed_dataframes():
    movies_df = pd.read_csv(PROCESSED_MOVIES_DATA_PATH)
    ratings_df = pd.read_csv(PROCESSED_RATINGS_DATA_PATH)
    return movies_df, ratings_df


def save_dataframe(df, file_name):
    file_name.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_name, index=False)

def data_analysis():
    movies_df, ratings_df = load_processed_dataframes()
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
    
    print("\nRating distribution:")
    rating_distribution = ratings_df["rating"].value_counts().sort_index()
    rating_distribution_percent = ratings_df["rating"].value_counts(normalize=True).sort_index() * 100
    for rating_value in rating_distribution.index:
        count = int(rating_distribution.loc[rating_value])
        percent = rating_distribution_percent.loc[rating_value]
        print(f"Rating {rating_value * 5}: {count} ({percent:.2f}%)")

def data_cleaning():
    movies_df, ratings_df = load_dataframes()

    print("Dataframe shapes:")
    print("Movies:", movies_df.shape)
    print("Ratings:", ratings_df.shape)    
    print("Total users:", ratings_df["userId"].nunique())
    print("Total movies:", ratings_df["movieId"].nunique())

    ratings_df = ratings_df.drop_duplicates()
    print("\nAfter removing duplicate ratings:", ratings_df.shape)

    movies_df = movies_df.drop_duplicates()
    print("After removing duplicate movies:", movies_df.shape)

    if ratings_df["rating"].max() > 1:
        ratings_df["rating"] /= 5

    year_col = []
    for i, row in movies_df.iterrows():
        year = row["title"].split()[-1][1:-1]
        year_col.append(year)
    movies_df["year"] = year_col

    genre_col = []
    for i, row in movies_df.iterrows():
        genre = row["genres"].replace("|", " ").strip('"')
        genre_col.append(genre)
    movies_df["genres"] = genre_col

    movie_title_col = []
    for i, row in movies_df.iterrows():
        title_list = row["title"].split()[0:-1]
        title = " ".join(title_list)
        movie_title_col.append(title)
    movies_df["title"] = movie_title_col
    
    if "timestamp" in ratings_df.columns:
        ratings_df = ratings_df.drop(columns=["timestamp"])

    save_dataframe(movies_df, PROCESSED_MOVIES_DATA_PATH)
    save_dataframe(ratings_df, PROCESSED_RATINGS_DATA_PATH)
    print("\nCleaned datasets saved to:")
    print(PROCESSED_MOVIES_DATA_PATH)
    print(PROCESSED_RATINGS_DATA_PATH)

if __name__ == "__main__":
    data_cleaning()
    print("\n")
    data_analysis()