import os
from pathlib import Path
from typing import List

from loguru import logger
import pandas as pd
from pandas_path import path
from PIL import Image
import torch
import typer
import random

import pytorch_lightning as pl
from benchmark_src.cloud_dataset import CloudDataset
from benchmark_src.cloud_model import CloudModel

import warnings

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

warnings.filterwarnings("ignore")

DATA_DIR = Path.cwd() / "data"
TRAIN_FEATURES = DATA_DIR / "train_features"
TRAIN_LABELS = DATA_DIR / "train_labels"
PREDICTIONS_DIRECTORY = DATA_DIR / "predictions"
ASSETS_DIRECTORY = DATA_DIR / "assets"

BANDS = ["B02", "B03", "B04", "B08"]

assert TRAIN_FEATURES.exists()

# Set the pytorch cache directory and include cached models in your submission.zip
os.environ["TORCH_HOME"] = str(ASSETS_DIRECTORY / "torch")

def add_paths(df, feature_dir, label_dir=None, bands=BANDS):
    """
    Given dataframe with a column for chip_id, returns a dataframe with a column
    added indicating the path to each band's TIF image as "{band}_path", eg "B02_path".
    A column is also added to the dataframe with paths to the label TIF, if the
    path to the labels directory is provided.
    """
    for band in bands:
        df[f"{band}_path"] = feature_dir / df["chip_id"] / f"{band}.tif"
        
        assert df[f"{band}_path"].path.exists().all()
        
    if label_dir is not None:
        df["label_path"] = label_dir / (df["chip_id"] + ".tif")
        
        assert df["label_path"].path.exists().all()

    return df

def main(model_weights_path: Path = ASSETS_DIRECTORY / "cloud_model_V1.pt",
         test_features_dir: Path = DATA_DIR / "test_features",
         predictions_dir: Path = PREDICTIONS_DIRECTORY,
         bands: List[str] = ["B02", "B03", "B04", "B08"],
         fast_dev_run: bool = False,
         ):

    # Create the predictions directory
    predictions_dir.mkdir(exist_ok=True, parents=True)

    # Load Data
    train_meta = pd.read_csv(DATA_DIR / "train_metadata.csv")
    train_meta = add_paths(train_meta, TRAIN_FEATURES, TRAIN_LABELS)

    # Split the data, set a seed for reproducibility
    random.seed(9)

    # put 1/3 of chips into the validation set
    chip_ids = train_meta.chip_id.unique().tolist()
    val_chip_ids = random.sample(chip_ids, round(len(chip_ids) * 0.33))

    val_mask = train_meta.chip_id.isin(val_chip_ids)
    val = train_meta[val_mask].copy().reset_index(drop=True)
    train = train_meta[~val_mask].copy().reset_index(drop=True)

    # separate features from labels
    feature_cols = ["chip_id"] + [f"{band}_path" for band in BANDS]

    val_x = val[feature_cols].copy()
    val_y = val[["chip_id", "label_path"]].copy()

    train_x = train[feature_cols].copy()
    train_y = train[["chip_id", "label_path"]].copy()

    # Set up pytorch_lightning.Trainer object
    cloud_model = CloudModel(
        bands=BANDS,
        x_train=train_x,
        y_train=train_y,
        x_val=val_x,
        y_val=val_y,
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="iou_epoch", mode="max", verbose=True
    )

    early_stopping_callback = pl.callbacks.early_stopping.EarlyStopping(
        monitor="iou_epoch",
        patience=(cloud_model.patience * 3),
        mode="max",
        verbose=True,
    )

    trainer = pl.Trainer(
        gpus=None,
        fast_dev_run=False,
        callbacks=[checkpoint_callback, early_stopping_callback],
    )

    trainer.fit(model=cloud_model)
    writer.flush()


if __name__ == "__main__":
    typer.run(main)