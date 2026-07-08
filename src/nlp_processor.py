from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train_data.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test_data.csv"
VAL_DATA_PATH = PROCESSED_DATA_DIR / "val_data.csv"

MOVIES_DATA_PATH = DATA_DIR / "movies.csv"


def load_datasets():
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)
    val_df = pd.read_csv(VAL_DATA_PATH)
    movies_df = pd.read_csv(MOVIES_DATA_PATH)

    return movies_df, train_df, test_df, val_df

def build_recommender(movies_df, train_df):
    cv=TfidfVectorizer()
    tfidf_matrix=cv.fit_transform(movies_df['genres'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    indices=pd.Series(movies_df.index,index=movies_df['title'])
    titles=movies_df['title']

    return titles, indices, cosine_sim

def recommendations(titles, indices, cosine_sim, title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:21]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices]

def main():
    movies_df, train_df, _, _ = load_datasets()
    titles, indices, cosine_sim = build_recommender(movies_df, train_df)
    print(
        recommendations(
            titles,
            indices,
            cosine_sim, 
            "Toy Story (1995)"
        )
    )

if __name__ == "__main__":
    main()