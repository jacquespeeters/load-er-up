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
from score import scoring_fn_func

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

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

    # %%time
    df_learning, y_learning = preprocess(data_file=join(args.data_dir, fname))

    # 4min20sec in local, we should avoid this when dev

    my_model = EnsembleModel()
    my_model.train(df_learning, y_learning)
    # save the model to disk
    save_model(my_model, args.model_dir)

    # TODO - Move this part to EnsembleModel() and add fscore to MLflow
    def format_targets(y_learning):
        machines_names = y_learning["machine"].unique().tolist()
        machines_names.sort()

        targets = pd.pivot(
            y_learning,
            columns="machine",
            index=["window"],
            values=my_model.targets,
        )

        targets.columns = [f"{col[1]}.{col[0]}" for col in targets.columns]
        cols = []
        for machine_name in machines_names:
            for target in my_model.targets:
                cols.append(f"{machine_name}.{target}")

        targets = targets.reindex(cols, axis=1)
        targets = targets.reset_index()
        return targets

    predictions = my_model.predict(df_learning)
    targets = format_targets(y_learning)

    fscore = scoring_fn_func(targets, predictions)
    print(f"Training fscore is {round(fscore, 3)}")


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
    else:
        default_model_dir = getenv("SM_MODEL_DIR", "/opt/ml/models")

    if os.path.exists("./data/public"):
        default_data_dir = "./data/public"
    else:
        default_data_dir = getenv("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=default_model_dir)
    parser.add_argument("--data_dir", type=str, default=default_data_dir)
    args, _ = parser.parse_known_args()
    train(args)
