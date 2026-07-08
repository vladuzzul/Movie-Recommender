from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MOVIES_DATA_PATH = PROCESSED_DATA_DIR / "movies.csv"
RATINGS_DATA_PATH = PROCESSED_DATA_DIR / "ratings.csv"

def load_dataframes():
    movies_df = pd.read_csv(MOVIES_DATA_PATH)
    ratings_df = pd.read_csv(RATINGS_DATA_PATH)
    return movies_df, ratings_df

def build_recommender(movies_df):
    cv = TfidfVectorizer()
    tfidf_matrix = cv.fit_transform(movies_df["genres"])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    indices = pd.Series(movies_df.index, index=movies_df["title"])
    titles = movies_df["title"]

    return titles, indices, cosine_sim

def resolve_movie_index(movies_df, indices, title=None, movie_id=None):
    if movie_id is not None:
        matches = movies_df.index[movies_df["movieId"] == movie_id].tolist()
        if not matches:
            raise KeyError(movie_id)
        return int(matches[0])

    idx = indices[title]
    if hasattr(idx, "iloc"):
        idx = idx.iloc[0]
    return int(idx)

def recommendations(movies_df, ratings_df, titles, indices, cosine_sim, title=None, movie_id=None):
    idx = resolve_movie_index(movies_df, indices, title=title, movie_id=movie_id)
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [score for score in sim_scores if score[0] != idx][:10]
    movie_indices = [i[0] for i in sim_scores]
    recommended = movies_df.iloc[movie_indices].copy()
    avg_rating = ratings_df.groupby("movieId")["rating"].mean()

    recommended["rating"] = (recommended["movieId"].map(avg_rating) * 5).round(2)

    recommended = recommended.sort_values(
        by="rating",
        ascending=False
    )

    recommended = recommended[0:5]


    return recommended[["movieId", "title", "rating"]]

def main(movie_title=None, movie_id=None):
    movies_df, ratings_df = load_dataframes()
    titles, indices, cosine_sim = build_recommender(movies_df)
    return recommendations(
            movies_df,
            ratings_df,
            titles,
            indices,
            cosine_sim, 
            movie_title,
            movie_id
        )

if __name__ == "__main__":
    print(main("Spectre"))
