"""Unearthed Sound The Alarm Prediction Template"""
import logging
import argparse
from os import getenv
from os.path import join
import numpy as np
import pandas as pd

from preprocess import preprocess
from train import model_fn

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    """Prediction.

    The main function is only used by the Unearthed CLI.

    When a submission is made online AWS SageMaker Processing Jobs are used to perform
    preprocessing and Batch Transform Jobs are used to pass the result of preprocessing
    to the trained model.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=getenv("SM_MODEL_DIR", "/opt/ml/models"))
    parser.add_argument("--data_dir", type=str, default=getenv("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"))
    args, _ = parser.parse_known_args()
    
    # call preprocessing on the data
    x_inputs, _ = preprocess(join(args.data_dir, "public.csv.gz"))

    # pass the model the preprocessed data
    logger.info("creating predictions")

    model = model_fn(args.model_dir)
    predictions = model.predict(pd.concat(x_inputs, axis=1).reset_index())

    logger.info(f"predictions have shape of {predictions.shape}")

    # save the predictions
    predictions.to_csv(
        "/opt/ml/output/predictions.csv.out"
    )
