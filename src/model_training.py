import tensorflow as tf
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train_data.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test_data.csv"
VAL_DATA_PATH = PROCESSED_DATA_DIR / "val_data.csv"

BATCH_SIZE = 128

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
        },
        train_df["rating"].values
    ))

    val_datagen = tf.data.Dataset.from_tensor_slices((
        {
            "userId": val_df["userId"].values,
            "movieId": val_df["movieId"].values,
        },
        val_df["rating"].values
    ))

    test_datagen = tf.data.Dataset.from_tensor_slices((
        {
            "userId": test_df["userId"].values,
            "movieId": test_df["movieId"].values,
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