# Movie Recommender

An NLP-focused movie recommendation app that compares movies by the language of
their genre descriptions. The project cleans MovieLens-style metadata, turns
each movie's genres into TF-IDF text vectors, measures similarity between those
vectors, and serves the ranked recommendations through a FastAPI backend and a
small browser interface.

## What the NLP Recommender Does

The main recommendation engine lives in `src/nlp_processor.py`. It is a
content-based recommender: instead of learning a user's taste from a large model,
it represents every movie as text and recommends movies whose text features are
close to the movies a user liked.

The current text signal is the processed `genres` column. During preprocessing,
raw genre strings such as `Adventure|Animation|Children|Comedy|Fantasy` are
normalized into space-separated text such as:

```text
Adventure Animation Children Comedy Fantasy
```

That cleaned text is then passed through scikit-learn's `TfidfVectorizer`.
TF-IDF gives each genre token a weight based on how important it is for one
movie relative to the whole catalog. Common genres still matter, but more
distinctive combinations help separate movies from each other.

Similarity is calculated with `linear_kernel` over the TF-IDF matrix. Because
TF-IDF vectors are normalized by default, this acts like cosine similarity:
movies with more similar weighted genre language receive higher similarity
scores.

## Recommendation Flow

1. `src/data_preprocessing.py` loads the raw movie and rating CSV files.
2. Movie titles are split into `title` and `year`.
3. Genre pipes are converted into plain text tokens for NLP processing.
4. Ratings are normalized to a 0-1 scale and saved to `data/processed`.
5. `src/nlp_processor.py` builds a TF-IDF matrix from processed movie genres.
6. For a seed movie, the recommender finds the most similar movies by text
   similarity.
7. The similar movies are re-ranked using average audience rating.
8. `src/api.py` combines recommendations from the user's highest-rated movies
   and returns a final scored list.

The API scoring layer uses three signals:

- the user's rating for the seed movie
- the recommended movie's average rating
- the NLP similarity rank returned by `src/nlp_processor.py`

This keeps the NLP similarity at the center while still nudging the final list
toward movies that other users rated highly.

## Project Structure

```text
src/
  api.py                 FastAPI app, static routes, and recommendation API
  data_preprocessing.py  CSV cleaning, title/year extraction, genre text cleanup
  nlp_processor.py       TF-IDF vectorization and similarity-based ranking

frontend/
  index.html             Start screen
  rate.html              Movie search and rating screen
  suggestions.html       Recommendation results screen
  app.js                 Browser-side state and API calls
  styles.css             Web interface styling

data/
  movies.csv             Raw movie metadata
  ratings.csv            Raw user ratings
  processed/             Cleaned CSV files used by the recommender
```

## Setup

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

If you need to regenerate the processed CSV files, run:

```bash
.venv/bin/python src/data_preprocessing.py
```

## Run the Web App

Start the FastAPI server from the project root:

```bash
.venv/bin/uvicorn src.api:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

The browser experience is split into three screens:

- `/` starts the Cinema Quest interface.
- `/rate` lets a user search for movies and rate them from 1 to 5 stars.
- `/suggestions` sends those ratings to the NLP recommender and displays the
  ranked results.

## API

Health check:

```http
GET /api/health
```

Search or list movies:

```http
GET /api/movies?q=toy&limit=24
```

Generate recommendations:

```http
POST /api/recommendations
Content-Type: application/json
```

Example payload:

```json
{
  "ratings": [
    { "title": "Spectre", "rating": 5 },
    { "title": "Toy Story", "rating": 4 }
  ]
}
```

Example response fields:

- `title`: recommended movie title
- `rating`: average audience rating on a 0-5 scale
- `genres`: processed genre text used by the NLP pipeline
- `year`: release year extracted during preprocessing
- `score`: combined recommendation score
- `based_on`: rated movies that generated the recommendation

## Run the NLP Recommender Directly

You can call the NLP recommender without the web app:

```bash
.venv/bin/python src/nlp_processor.py
```

By default, the script prints recommendations for `Spectre`. To experiment with
other seed movies, call `main("Movie Title")` from `src.nlp_processor`.

## Current NLP Limitations

The recommender currently uses genre text only. That makes it fast,
interpretable, and easy to debug, but it also means the model cannot yet
understand plot summaries, cast, directors, user review text, or semantic themes
that are not present in the genre labels.