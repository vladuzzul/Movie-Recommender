from functools import lru_cache
from pathlib import Path
from typing import Annotated

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.nlp_processor import load_dataframes, main as recommend_movie


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = PROJECT_ROOT / "frontend"


class MovieRating(BaseModel):
    title: str
    rating: Annotated[int, Field(ge=1, le=5)]
    movieId: int | None = None


class RecommendationRequest(BaseModel):
    ratings: list[MovieRating] = Field(min_length=1)


class MovieResponse(BaseModel):
    movieId: int
    title: str
    genres: str
    year: int | None = None
    rating: float | None = None


class RecommendationResponse(BaseModel):
    movieId: int
    title: str
    rating: float | None = None
    genres: str
    year: int | None = None
    score: float
    based_on: list[str]


app = FastAPI(
    title="Movie Recommender API",
    description="FastAPI wrapper for the NLP movie recommender.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_catalog() -> tuple[pd.DataFrame, pd.Series]:
    movies_df, ratings_df = load_dataframes()
    movies_df = movies_df.copy()
    movies_df["title_key"] = movies_df["title"].str.casefold()
    movies_df["year"] = pd.to_numeric(movies_df["year"], errors="coerce")

    average_rating = ratings_df.groupby("movieId")["rating"].mean().mul(5).round(2)
    movies_df["rating"] = movies_df["movieId"].map(average_rating)
    return movies_df, average_rating


def clean_optional_int(value) -> int | None:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return None
    return int(numeric)


def serialize_movie(row: pd.Series) -> dict:
    year = row.get("year")
    rating = row.get("rating")
    return {
        "movieId": int(row["movieId"]),
        "title": str(row["title"]),
        "genres": str(row.get("genres", "")),
        "year": clean_optional_int(year),
        "rating": None if pd.isna(rating) else float(rating),
    }


def find_movie(title: str, movie_id: int | None = None) -> pd.Series | None:
    movies_df, _ = get_catalog()
    if movie_id is not None:
        exact_id = movies_df[movies_df["movieId"] == movie_id]
        if not exact_id.empty:
            return exact_id.iloc[0]

    exact = movies_df[movies_df["title_key"] == title.casefold()]
    if exact.empty:
        return None
    exact = exact.sort_values(["rating", "year"], ascending=[False, False])
    return exact.iloc[0]


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/rate", include_in_schema=False)
def rate_page():
    return FileResponse(STATIC_DIR / "rate.html")


@app.get("/suggestions", include_in_schema=False)
def suggestions_page():
    return FileResponse(STATIC_DIR / "suggestions.html")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/movies", response_model=list[MovieResponse])
def list_movies(
    q: Annotated[str | None, Query(max_length=80)] = None,
    limit: Annotated[int, Query(ge=1, le=60)] = 24,
):
    movies_df, _ = get_catalog()

    if q:
        query = q.casefold().strip()
        filtered = movies_df[
            movies_df["title_key"].str.contains(query, regex=False, na=False)
        ]
        filtered = filtered.sort_values(["rating", "year"], ascending=[False, False])
    else:
        filtered = movies_df.sort_values(["rating", "year"], ascending=[False, False])

    return [serialize_movie(row) for _, row in filtered.head(limit).iterrows()]


@app.post("/api/recommendations", response_model=list[RecommendationResponse])
def recommendations(payload: RecommendationRequest):
    movies_df, _ = get_catalog()
    catalog_by_movie_id = movies_df.set_index("movieId")
    catalog_by_title = movies_df.set_index("title")
    rated_movie_ids = {item.movieId for item in payload.ratings if item.movieId is not None}
    has_rated_movie_ids = bool(rated_movie_ids)
    rated_titles = {item.title for item in payload.ratings}

    seeds = sorted(payload.ratings, key=lambda item: item.rating, reverse=True)
    positive_seeds = [item for item in seeds if item.rating >= 3] or seeds[:1]

    combined: dict[str, dict] = {}
    unknown_titles: list[str] = []

    for seed in positive_seeds[:5]:
        movie = find_movie(seed.title, seed.movieId)
        if movie is None:
            unknown_titles.append(seed.title)
            continue

        seed_movie_id = int(movie["movieId"])
        seed_title = str(movie["title"])
        seed_weight = seed.rating / 5
        try:
            frame = recommend_movie(seed_title, seed_movie_id)
        except KeyError:
            unknown_titles.append(seed.title)
            continue

        for rank, row in frame.reset_index(drop=True).iterrows():
            movie_id = int(row["movieId"])
            title = str(row["title"])
            if movie_id in rated_movie_ids or (not has_rated_movie_ids and title in rated_titles):
                continue

            recommender_rating = row.get("rating")
            normalized_rating = (
                0.0 if pd.isna(recommender_rating) else float(recommender_rating) / 5
            )
            rank_boost = (5 - rank) / 5
            score = round((seed_weight * 0.65 + normalized_rating * 0.25 + rank_boost * 0.10) * 100, 2)

            current = combined.setdefault(
                movie_id,
                {
                    "movieId": movie_id,
                    "title": title,
                    "score": 0.0,
                    "based_on": set(),
                    "rating": None,
                },
            )
            current["score"] += score
            current["based_on"].add(seed_title)
            current["rating"] = None if pd.isna(recommender_rating) else float(recommender_rating)

    if not combined:
        detail = "No recommendations found for your rated movies."
        if unknown_titles:
            detail += f" Unknown titles: {', '.join(unknown_titles)}."
        raise HTTPException(status_code=404, detail=detail)

    response = []
    for movie_id, item in combined.items():
        title = item["title"]
        if movie_id in catalog_by_movie_id.index:
            catalog_row = catalog_by_movie_id.loc[movie_id]
        else:
            catalog_row = catalog_by_title.loc[title] if title in catalog_by_title.index else None
        if isinstance(catalog_row, pd.DataFrame):
            catalog_row = catalog_row.iloc[0]
        genres = "" if catalog_row is None else str(catalog_row.get("genres", ""))
        year = None if catalog_row is None else clean_optional_int(catalog_row.get("year"))

        response.append(
            {
                "movieId": item["movieId"],
                "title": title,
                "rating": item["rating"],
                "genres": genres,
                "year": year,
                "score": round(item["score"], 2),
                "based_on": sorted(item["based_on"]),
            }
        )

    return sorted(response, key=lambda item: item["score"], reverse=True)[:10]


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
