"""Unearthed Sound The Alarm Training Template"""
import argparse
import logging
import os
import pickle
import sys
from io import StringIO
from os import getenv
from os.path import abspath, join

import pandas as pd

from ensemble_model import EnsembleModel
from preprocess import preprocess

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

# Work around for a SageMaker path issue
# (see https://github.com/aws/sagemaker-python-sdk/issues/648)
# WARNING - removing this may cause the submission process to fail
if abspath("/opt/ml/code") not in sys.path:
    sys.path.append(abspath("/opt/ml/code"))


def train(args):
    """Train

    Your model code goes here.
    """
    logger.info("calling training function")

    # preprocess
    # if you require any particular preprocessing to create features then this
    # *must* be contained in the preprocessing function for the Unearthed pipeline
    # apply it to the private data

    # In local read in 10sec instead of 10mins
    if os.path.exists(join(args.data_dir, "public.parquet")):
        fname = "public.parquet"
    else:
        fname = "public.csv.gz"

    df_learning, y_learning = preprocess(data_file=join(args.data_dir, fname))
    # 4min20sec in local, we should avoid this when dev

    my_model = EnsembleModel()
    predictions = my_model.train(df_learning, y_learning)
    # importance = my_model.get_feature_importance()
    # importance["isdigit"] = importance["feature"].str.split("_").str.len()
    # isdigit = importance["feature"].str.split("_").str[-1].str.isdigit()
    # importance = importance[isdigit]
    # importance["feature_root"] = (
    #     importance["feature"].str.split("_").str[:-3].str.join("_")
    # )
    # N_model = my_model.N_FOLD * len(my_model.targets)
    # importance_agg = (
    #     importance.groupby("feature_root")["importance"].sum() / N_model
    # ).reset_index()
    # importance_agg = importance_agg.sort_values("importance")

    # importance = importance[["feature", "feature_root"]].merge(importance_agg)
    # list(importance[importance["importance"] < 0.5]["feature_root"].unique())
    # noisy_features = importance[importance["importance"] < 1]["feature"].unique()
    # predictions = my_model.train(df_learning.drop(columns=noisy_features), y_learning)

    print(predictions.sample(5))
    # save the model to disk
    save_model(my_model, args.model_dir)

    my_model.predict(df_learning)


def save_model(model, model_dir):
    """Save model to a binary file.

    This function must write the model to disk in a format that can
    be loaded from the model_fn.

    WARNING - modifying this function may cause the submission process to fail.
    """
    logger.info(f"saving model to {model_dir}")
    with open(join(model_dir, "model.pkl"), "wb") as model_file:
        pickle.dump(model, model_file)


def model_fn(model_dir):
    """Load model from binary file.

    This function loads the model from disk. It is called by SageMaker.

    WARNING - modifying this function may case the submission process to fail.
    """
    logger.info("loading model")
    with open(__file__) as f:
        file_contents = f.read()
        logger.info(file_contents)
    logger.info(str(EnsembleModel))
    with open(join(model_dir, "model.pkl"), "rb") as file:
        return pickle.load(file)


def input_fn(input_data, content_type):
    """Take request data and de-serialize the data into an object for prediction.

    In the Unearthed submission pipeline the data is passed as "text/csv". This
    function reads the CSV into a Pandas dataframe ready to be passed to the model.

    WARNING - modifying this function may cause the submission process to fail.
    """
    return pd.read_csv(StringIO(input_data))


if __name__ == "__main__":
    """Training Main

    The main function is called by both Unearthed's SageMaker pipeline and the
    Unearthed CLI's "unearthed train" command.

    WARNING - modifying this function may cause the submission process to fail.

    The main function must call preprocess, arrange th
    """

    if os.path.exists("./data/public"):
        default_model_dir = "./data/public"
        default_data_dir = "./data/public"
        default_input = "/opt/ml/processing/input/public/public.csv.gz"
    else:
        default_model_dir = getenv("SM_MODEL_DIR", "/opt/ml/models")
        default_data_dir = getenv("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")
        default_input = "./data/public/public.parquet"

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=default_input)
    parser.add_argument("--model_dir", type=str, default=default_model_dir)
    parser.add_argument("--data_dir", type=str, default=default_data_dir)
    args, _ = parser.parse_known_args()
    train(args)
