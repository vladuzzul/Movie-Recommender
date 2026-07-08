import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
)
import pandas as pd
from pathlib import Path
import mlflow

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_PATH = PROJECT_ROOT / "models" / "model.keras"
HISTORY_PATH = DATA_DIR / "model_history.csv"

TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train_data.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test_data.csv"
VAL_DATA_PATH = PROCESSED_DATA_DIR / "val_data.csv"
MOVIES_DATA_PATH = PROCESSED_DATA_DIR / "movies.csv"

BATCH_SIZE = 128
EMBEDDING_SIZE = 50
DROPOUT_RATE = 0.6
EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

def load_datasets():
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)
    val_df = pd.read_csv(VAL_DATA_PATH)

    return train_df, test_df, val_df

def create_data_generators(train_df, test_df, val_df):
    train_datagen = tf.data.Dataset.from_tensor_slices((
        {
            "userId": train_df["userId"].values,
            "movieId": train_df["movieId"].values,
            "genres": train_df["genres"].astype(str).values,
            "year": train_df["year"].values,
        },
        train_df["rating"].values
    ))

    val_datagen = tf.data.Dataset.from_tensor_slices((
        {
            "userId": val_df["userId"].values,
            "movieId": val_df["movieId"].values,
            "genres": val_df["genres"].astype(str).values,
            "year": val_df["year"].values,
        },
        val_df["rating"].values
    ))

    test_datagen = tf.data.Dataset.from_tensor_slices((
        {
            "userId": test_df["userId"].values,
            "movieId": test_df["movieId"].values,
            "genres": test_df["genres"].astype(str).values,
            "year": test_df["year"].values,
        },
        test_df["rating"].values
    ))

    train_datagen = (
        train_datagen
        .shuffle(len(train_df))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_datagen = (
        val_datagen
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    test_datagen = (
        test_datagen
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_datagen, test_datagen, val_datagen

class RecommenderModel(tf.keras.Model):
    def __init__(self, user_vocab, movie_vocab, genre_vocab, embedding_size=EMBEDDING_SIZE, dropout_rate=DROPOUT_RATE):
        super().__init__()
        self.user_lookup = tf.keras.layers.IntegerLookup(vocabulary=user_vocab)
        self.movie_lookup = tf.keras.layers.IntegerLookup(vocabulary=movie_vocab)
        self.genre_lookup = tf.keras.layers.StringLookup(vocabulary=genre_vocab, mask_token=None)

        self.user_embedding = tf.keras.layers.Embedding(
            input_dim=self.user_lookup.vocabulary_size(),
            output_dim=embedding_size,
        )
        self.movie_embedding = tf.keras.layers.Embedding(
            input_dim=self.movie_lookup.vocabulary_size(),
            output_dim=embedding_size,
        )
        self.genre_embedding = tf.keras.layers.Embedding(
            input_dim=self.genre_lookup.vocabulary_size(),
            output_dim=embedding_size,
        )

        self.year_normalization = tf.keras.layers.Normalization(axis=None)

        self.concatenate = tf.keras.layers.Concatenate()
        self.dense_1 = tf.keras.layers.Dense(128, activation="relu")
        self.dropout_1 = tf.keras.layers.Dropout(dropout_rate)
        self.dense_2 = tf.keras.layers.Dense(64, activation="relu")
        self.dropout_2 = tf.keras.layers.Dropout(dropout_rate)
        self.output_layer = tf.keras.layers.Dense(1)

    def get_config(self):
        config = super().get_config()
        config.update({
        })
        return config

    def call(self, inputs, training=False):
        user_id = tf.cast(inputs["userId"], tf.int64)
        movie_id = tf.cast(inputs["movieId"], tf.int64)
        year = tf.cast(inputs["year"], tf.float32)

        user_vector = self.user_embedding(self.user_lookup(user_id))
        movie_vector = self.movie_embedding(self.movie_lookup(movie_id))
        genre_tokens = tf.strings.split(inputs["genres"], ",")
        genre_indices = self.genre_lookup(genre_tokens)
        genre_vector = self.genre_embedding(genre_indices)
        genre_vector = tf.reduce_mean(genre_vector, axis=1)
        year_vector = tf.expand_dims(self.year_normalization(year), axis=-1)

        x = self.concatenate([user_vector, movie_vector, genre_vector, year_vector])
        x = self.dense_1(x)
        x = self.dropout_1(x, training=training)
        x = self.dense_2(x)
        x = self.dropout_2(x, training=training)
        return self.output_layer(x)


def build_model(train_df=None):
    if train_df is None:
        train_df, _, _ = load_datasets()

    user_vocab = train_df["userId"].unique().tolist()
    movie_vocab = train_df["movieId"].unique().tolist()
    genre_vocab = sorted({genre for genres in train_df["genres"].fillna("").astype(str) for genre in genres.split(",") if genre})

    model = RecommenderModel(user_vocab=user_vocab, movie_vocab=movie_vocab, genre_vocab=genre_vocab)
    model.year_normalization.adapt(train_df["year"].values)
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        ),
        loss=tf.keras.losses.Huber(),
        metrics=[
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
        ],
    )
    return model

def create_callbacks():
    """Create training callbacks"""
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )
    return [early_stop, reduce_lr]


def log_run_params(train_df, test_df, val_df):
    mlflow.log_params({
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "embedding size": EMBEDDING_SIZE,
        "dropout": DROPOUT_RATE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "loss": "Huber",
        "optimizer": "AdamW",
        "callbacks": "EarlyStopping,ReduceLROnPlateau",
        "features": "userId,movieId,genres,year",
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df)
    })

def log_epochs_metrics(history):
    def metric_at(name, epoch_index):
        values = history.history.get(name)
        if values is None:
            return None
        return values[epoch_index]

    for epoch in range(len(history.history["loss"])):
        metrics = {
            "train_loss": history.history["loss"][epoch],
            "val_loss": history.history["val_loss"][epoch],
        }

        for history_name, mlflow_name in [
            ("rmse", "train_rmse"),
            ("val_rmse", "val_rmse"),
            ("mae", "train_mae"),
            ("val_mae", "val_mae"),
            ("learning_rate", "learning_rate"),
            ("lr", "learning_rate"),
        ]:
            value = metric_at(history_name, epoch)
            if value is not None:
                metrics[mlflow_name] = value

        mlflow.log_metrics(metrics, step=epoch + 1)

def save_training_artifacts(model, history):
    """Attach the model and history to the active MLflow run."""
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(HISTORY_PATH)
    mlflow.log_artifact(str(HISTORY_PATH))

    model.save(MODEL_PATH)
    mlflow.log_artifact(str(MODEL_PATH))

def train_model(epochs=EPOCHS):
    train_df, test_df, val_df = load_datasets()
    train_ds, test_ds, val_ds = create_data_generators(train_df, test_df, val_df)

    with mlflow.start_run(run_name="training"):
        log_run_params(train_df, test_df, val_df)

        model = build_model(train_df)
        model.summary()

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=create_callbacks()
        )

        log_epochs_metrics(history)
        save_training_artifacts(model, history)

    return model, history, test_ds


def evaluate_model(model=None, test_ds=None):
    if model is None:
        train_df, test_df, val_df = load_datasets()
        _, test_ds, _ = create_data_generators(train_df, test_df, val_df)
        model = build_model(train_df)

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Saved model not found at {MODEL_PATH}. Train first before calling evaluate_model() without a model argument."
            )

        sample_features, _ = next(iter(test_ds.take(1)))
        model(sample_features, training=False)
        model.load_weights(MODEL_PATH)

    elif test_ds is None:
        train_df, test_df, val_df = load_datasets()
        _, test_ds, _ = create_data_generators(train_df, test_df, val_df)

    return model.evaluate(test_ds)

if __name__ == "__main__":
    model, history, test_ds = train_model()
    test_results = evaluate_model(model=model, test_ds=test_ds)
    print("Test results:", test_results)
    