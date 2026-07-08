# Movie-Recommender
Neural Network based movie recommender

## Web app

Run the FastAPI app from the project root:

```bash
.venv/bin/uvicorn src.api:app --reload
```

Then open `http://127.0.0.1:8000`.

The game-like web interface is split into multiple pages:

- `/` starts the Cinema Quest experience.
- `/rate` lets a user search for movies and rate them from 1 to 5 stars.
- `/suggestions` shows a separate recommendations screen generated through
  `src/nlp_processor.py`.

Main API endpoints:

- `GET /api/movies?q=toy&limit=24`
- `POST /api/recommendations`

Example recommendation payload:

```json
{
  "ratings": [
    { "title": "Spectre", "rating": 5 }
  ]
}
```

## Evaluation

Run the evaluation script to log a new MLflow run with test metrics and generate personalized recommendations for a user:

```bash
python src/evaluation.py --user-id 123 --top-n 10
```

The script writes the test-set metrics to MLflow, saves the recommendation list as an artifact, and prints the top movies for the requested user id.
