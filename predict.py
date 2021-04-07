"""Unearthed Sound The Alarm Prediction Template"""
import argparse
import logging
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
    parser.add_argument(
        "--model_dir", type=str, default=getenv("SM_MODEL_DIR", "/opt/ml/models")
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=getenv("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"),
    )
    args, _ = parser.parse_known_args()

    # call preprocessing on the data
    x_inputs, _ = preprocess(join(args.data_dir, "public.csv.gz"))

    # load the model
    models = model_fn(args.model_dir)

    # pass the model the preprocessed data
    logger.info("creating predictions")

    # do predictions
    lags = ["y_1", "y_4", "y_12", "y_24"]
    y_predict = [dict() for _ in x_inputs]
    for model, lag in zip(models, lags):
        for y_machine, x_machine in zip(y_predict, x_inputs):
            y_machine[lag] = model.predict(x_machine.dropna().values).astype(bool)
    predictions = [
        pd.DataFrame(y, index=x.dropna().index, columns=lags)
        for x, y in zip(x_inputs, y_predict)
    ]

    # combine into one dataframe, machines are sorted in alphabetical order
    order = np.argsort([x.index.name for x in x_inputs])
    predictions = [predictions[i] for i in order]
    predictions = pd.concat(predictions, axis=1)

    logger.info(f"predictions have shape of {predictions.shape}")

    # save the predictions
    predictions.to_csv("/opt/ml/output/predictions.csv.out")
