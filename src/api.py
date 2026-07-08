from functools import lru_cache
from pathlib import Path
from typing import Annotated

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.nlp_processor import load_dataframes, main as recommend_for_title


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = PROJECT_ROOT / "frontend"


class MovieRating(BaseModel):
    title: str
    rating: Annotated[int, Field(ge=1, le=5)]


class RecommendationRequest(BaseModel):
    ratings: list[MovieRating] = Field(min_length=1)


class MovieResponse(BaseModel):
    movieId: int
    title: str
    genres: str
    year: int | None = None
    rating: float | None = None


class RecommendationResponse(BaseModel):
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

    average_rating = ratings_df.groupby("movieId")["rating"].mean().mul(5).round(2)
    movies_df["rating"] = movies_df["movieId"].map(average_rating)
    return movies_df, average_rating


def serialize_movie(row: pd.Series) -> dict:
    year = row.get("year")
    rating = row.get("rating")
    return {
        "movieId": int(row["movieId"]),
        "title": str(row["title"]),
        "genres": str(row.get("genres", "")),
        "year": None if pd.isna(year) else int(year),
        "rating": None if pd.isna(rating) else float(rating),
    }


def find_movie(title: str) -> pd.Series | None:
    movies_df, _ = get_catalog()
    exact = movies_df[movies_df["title_key"] == title.casefold()]
    if exact.empty:
        return None
    return exact.iloc[0]


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


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
    catalog_by_title = movies_df.set_index("title")
    rated_titles = {item.title for item in payload.ratings}

    seeds = sorted(payload.ratings, key=lambda item: item.rating, reverse=True)
    positive_seeds = [item for item in seeds if item.rating >= 3] or seeds[:1]

    combined: dict[str, dict] = {}
    unknown_titles: list[str] = []

    for seed in positive_seeds[:5]:
        movie = find_movie(seed.title)
        if movie is None:
            unknown_titles.append(seed.title)
            continue

        seed_title = str(movie["title"])
        seed_weight = seed.rating / 5
        try:
            frame = recommend_for_title(seed_title)
        except KeyError:
            unknown_titles.append(seed.title)
            continue

        for rank, row in frame.reset_index(drop=True).iterrows():
            title = str(row["title"])
            if title in rated_titles:
                continue

            recommender_rating = row.get("rating")
            normalized_rating = (
                0.0 if pd.isna(recommender_rating) else float(recommender_rating) / 5
            )
            rank_boost = (5 - rank) / 5
            score = round((seed_weight * 0.65 + normalized_rating * 0.25 + rank_boost * 0.10) * 100, 2)

            current = combined.setdefault(
                title,
                {"title": title, "score": 0.0, "based_on": set(), "rating": None},
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
    for title, item in combined.items():
        catalog_row = catalog_by_title.loc[title] if title in catalog_by_title.index else None
        if isinstance(catalog_row, pd.DataFrame):
            catalog_row = catalog_row.iloc[0]
        genres = "" if catalog_row is None else str(catalog_row.get("genres", ""))
        year = (
            None
            if catalog_row is None or pd.isna(catalog_row.get("year"))
            else int(catalog_row.get("year"))
        )

        response.append(
            {
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
