# Movie-Recommender
Neural Network based movie recommender

## Evaluation

Run the evaluation script to log a new MLflow run with test metrics and generate personalized recommendations for a user:

```bash
python src/evaluation.py --user-id 123 --top-n 10
```

The script writes the test-set metrics to MLflow, saves the recommendation list as an artifact, and prints the top movies for the requested user id.
