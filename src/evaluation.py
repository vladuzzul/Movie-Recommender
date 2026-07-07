import argparse
import sys
import tempfile
from pathlib import Path

import mlflow
import pandas as pd
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent

for path in (PROJECT_ROOT, SRC_DIR):
	path_str = str(path)
	if path_str not in sys.path:
		sys.path.insert(0, path_str)

from model_training import (  # noqa: E402
	MODEL_PATH,
	PROCESSED_DATA_DIR,
	build_model,
	load_datasets,
)


EVALUATION_EXPERIMENT = "movie-recommender-evaluation"
EVALUATION_BATCH_SIZE = 4096
RECOMMENDATION_BATCH_SIZE = 4096


def load_trained_model(model_path=MODEL_PATH):
	print("Loading training and test data...", flush=True)
	train_df, test_df, val_df = load_datasets()

	print("Building model and restoring weights...", flush=True)
	model = build_model(train_df)
	warmup_row = train_df.iloc[0]
	warmup_features = {
		"userId": tf.convert_to_tensor([warmup_row["userId"]]),
		"movieId": tf.convert_to_tensor([warmup_row["movieId"]]),
		"genres": tf.convert_to_tensor([str(warmup_row["genres"])]),
		"year": tf.convert_to_tensor([warmup_row["year"]]),
	}
	model(warmup_features, training=False)
	model.load_weights(model_path)
	print("Model ready.", flush=True)
	return model, train_df, test_df, val_df


def iter_dataframe_batches(df, batch_size, feature_columns, target_column=None):
	for start_index in range(0, len(df), batch_size):
		batch_df = df.iloc[start_index:start_index + batch_size]
		features = {
			column: batch_df[column].to_numpy()
			for column in feature_columns
		}
		if target_column is None:
			yield features
		else:
			yield features, batch_df[target_column].to_numpy()


def evaluate_on_test_set(model, test_df):
	print(f"Evaluating test set with batch size {EVALUATION_BATCH_SIZE}...", flush=True)
	huber_loss = tf.keras.losses.Huber()
	loss_metric = tf.keras.metrics.Mean(name="loss")
	mae_metric = tf.keras.metrics.MeanAbsoluteError(name="mae")
	rmse_metric = tf.keras.metrics.RootMeanSquaredError(name="rmse")

	for batch_index, (features, targets) in enumerate(
		iter_dataframe_batches(
			test_df,
			EVALUATION_BATCH_SIZE,
			feature_columns=["userId", "movieId", "genres", "year"],
			target_column="rating",
		),
		start=1,
	):
		predictions = model(features, training=False)
		batch_loss = huber_loss(targets, predictions)
		batch_size = tf.cast(tf.shape(targets)[0], tf.float32)
		loss_metric.update_state(batch_loss, sample_weight=batch_size)
		mae_metric.update_state(targets, predictions)
		rmse_metric.update_state(targets, predictions)

		if batch_index == 1 or batch_index % 5 == 0:
			print(f"  processed {batch_index} evaluation batches", flush=True)

	return {
		"loss": float(loss_metric.result().numpy()),
		"mae": float(mae_metric.result().numpy()),
		"rmse": float(rmse_metric.result().numpy()),
	}


def build_recommendations(model, user_id, top_n=10):
	print(f"Building recommendations for user {user_id}...", flush=True)
	movies_df = pd.read_csv(PROCESSED_DATA_DIR / "movies.csv")
	ratings_df = pd.read_csv(PROCESSED_DATA_DIR / "ratings.csv")

	seen_movie_ids = set(ratings_df.loc[ratings_df["userId"] == user_id, "movieId"])
	candidate_df = movies_df.loc[~movies_df["movieId"].isin(seen_movie_ids)].copy()

	if candidate_df.empty:
		candidate_df = movies_df.copy()

	candidate_df["userId"] = user_id
	candidate_df["genres"] = candidate_df["genres"].fillna("").astype(str)
	candidate_df["year"] = pd.to_numeric(candidate_df["year"], errors="coerce").fillna(0).astype(int)

	print(
		f"Scoring {len(candidate_df)} candidate movies with batch size {RECOMMENDATION_BATCH_SIZE}...",
		flush=True,
	)
	all_scores = []
	for batch_index, batch_features in enumerate(
		iter_dataframe_batches(
			candidate_df,
			RECOMMENDATION_BATCH_SIZE,
			feature_columns=["userId", "movieId", "genres", "year"],
		),
		start=1,
	):
		batch_scores = model(batch_features, training=False)
		all_scores.append(tf.squeeze(batch_scores, axis=-1).numpy())
		print(f"  scored {batch_index} recommendation batches", flush=True)
	scores = pd.Series(all_scores).explode().astype(float).to_numpy()
	print("Scoring finished.", flush=True)
	recommendations = (
		candidate_df.assign(predicted_rating=scores)
		.sort_values("predicted_rating", ascending=False)
		.head(top_n)
		.reset_index(drop=True)
	)
	recommendations.insert(0, "rank", recommendations.index + 1)
	return recommendations[["rank", "movieId", "title", "genres", "year", "predicted_rating"]]


def log_recommendations_artifact(recommendations):
	with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as temp_file:
		recommendations.to_csv(temp_file.name, index=False)
		temp_path = Path(temp_file.name)

	try:
		mlflow.log_artifact(str(temp_path), artifact_path="recommendations")
	finally:
		temp_path.unlink(missing_ok=True)


def run_evaluation(user_id, top_n=10, model_path=MODEL_PATH):
	mlflow.set_experiment(EVALUATION_EXPERIMENT)

	print("Starting evaluation run...", flush=True)
	model, train_df, test_df, val_df = load_trained_model(model_path=model_path)
	print("Evaluating on test set...", flush=True)
	test_metrics = evaluate_on_test_set(model, test_df)
	print("Generating recommendations...", flush=True)
	recommendations = build_recommendations(model, user_id=user_id, top_n=top_n)
	print("Logging MLflow run...", flush=True)

	with mlflow.start_run(run_name="evaluation"):
		mlflow.log_params(
			{
				"user_id": user_id,
				"top_n": top_n,
				"model_path": str(model_path),
				"train_samples": len(train_df),
				"val_samples": len(val_df),
				"test_samples": len(test_df),
			}
		)
		mlflow.log_metrics({f"test_{name}": value for name, value in test_metrics.items()})
		log_recommendations_artifact(recommendations)

	print("Evaluation run complete.", flush=True)

	return test_metrics, recommendations


def print_report(test_metrics, recommendations, user_id):
	print(f"\nTest metrics for user {user_id}:")
	for name, value in test_metrics.items():
		print(f"  {name}: {value:.6f}")

	print(f"\nTop {len(recommendations)} recommendations:")
	print(recommendations.to_string(index=False))


def parse_args():
	parser = argparse.ArgumentParser(description="Evaluate the recommender and generate top-N movies for a user.")
	parser.add_argument("--user-id", type=int, required=True, help="User id to score.")
	parser.add_argument("--top-n", type=int, default=10, help="Number of recommended movies to return.")
	parser.add_argument(
		"--model-path",
		type=Path,
		default=MODEL_PATH,
		help="Path to the saved Keras model archive.",
	)
	return parser.parse_args()


def main():
	args = parse_args()
	test_metrics, recommendations = run_evaluation(
		user_id=args.user_id,
		top_n=args.top_n,
		model_path=args.model_path,
	)
	print_report(test_metrics, recommendations, args.user_id)


if __name__ == "__main__":
	main()
