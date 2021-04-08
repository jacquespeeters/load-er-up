"""Unearthed Sound The Alarm Training Template"""
import argparse
import logging
import pickle
import sys
from io import StringIO
from os import getenv
from os.path import abspath, join

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from preprocess import preprocess
from ensemble_model import EnsembleModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Work around for a SageMaker path issue
# (see https://github.com/aws/sagemaker-python-sdk/issues/648)
# WARNING - removing this may cause the submission process to fail
if abspath('/opt/ml/code') not in sys.path:
    sys.path.append(abspath('/opt/ml/code'))


def train(args):
    """Train

    Your model code goes here.
    """
    logger.info('calling training function')
    with open(__file__, 'r') as f:
        file_content = f.read()
        logger.info(file_content)

    # preprocess
    # if you require any particular preprocessing to create features then this 
    # *must* be contained in the preprocessing function for the Unearthed pipeline
    # apply it to the private data
    x_inputs, y_inputs = preprocess(join(args.data_dir, 'public.csv.gz'))
    logger.info(f"training input shape for each machine is {x_inputs[0].shape}")
    logger.info(f"training target shape for each machine is {y_inputs[0].shape}")

    # an example model
    lags = ['y_1', 'y_4', 'y_12', 'y_24']
    models = [RandomForestClassifier() for _ in lags]
    for model, lag in zip(models, lags):
        x = []
        y = []
        for x_machine, y_machine in zip(x_inputs, y_inputs):

            mask = (~y_machine[lag].isna()) & y_machine.operating
            y_machine = y_machine[lag]
            x_machine = x_machine[mask]
            y_machine = y_machine[mask]
            x.append(x_machine.values)
            y.append(y_machine.values)
        y = np.concatenate(y).astype(float)
        x = np.concatenate(x)
        model.fit(x, y)

    # save the model to disk
    save_model(EnsembleModel(models), args.model_dir)


def save_model(model, model_dir):
    """Save model to a binary file.

    This function must write the model to disk in a format that can
    be loaded from the model_fn.

    WARNING - modifying this function may cause the submission process to fail.
    """
    logger.info(f"saving model to {model_dir}")
    with open(join(model_dir, 'model.pkl'), 'wb') as model_file:
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
    with open(join(model_dir, 'model.pkl'), 'rb') as file:
        return pickle.load(file)


def input_fn(input_data, content_type):
    """Take request data and de-serialize the data into an object for prediction.

    In the Unearthed submission pipeline the data is passed as "text/csv". This
    function reads the CSV into a Pandas dataframe ready to be passed to the model.

    WARNING - modifying this function may cause the submission process to fail.
    """
    return pd.read_csv(StringIO(input_data))


if __name__ == '__main__':
    """Training Main

    The main function is called by both Unearthed's SageMaker pipeline and the
    Unearthed CLI's "unearthed train" command.
    
    WARNING - modifying this function may cause the submission process to fail.

    The main function must call preprocess, arrange th
    """    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=getenv('SM_MODEL_DIR', '/opt/ml/models'))
    parser.add_argument('--data_dir', type=str, default=getenv('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training'))
    train(parser.parse_args())
