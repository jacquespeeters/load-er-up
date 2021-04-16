import argparse
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

weights = {"y_1": 1, "y_4": 2, "y_12": 4, "y_24": 3}


def scoring_fn(y, y_pred, weights, error):
    """Scoring Function

    The score for each machine is determined by calculating the "weighted" f1 score
    for each of the lagged ?, and then taking an average of these weighted f1
    scores.
    """
    logger.info("scoring_fn")

    f1_scores = []

    for i in range(0, len(y)):

        y_i = y[i]
        y_pred_i = y_pred[i]

        # Ensures the two indices are equal, otherwise throws an exception
        if not y_i.index.equals(y_pred_i.index):
            raise Exception("Indices do not agree!")
        print(y_i.shape)
        print(y_pred_i.shape)
        total_weight = sum(weights.values())
        ret = 0.0
        lags = ["y_1", "y_4", "y_12", "y_24"]

        # Simply get the error for each column and calculate the weighted sum.
        for j in range(0, len(lags)):
            ret += weights[lags[j]] * error(y_i.iloc[:, j], y_pred_i.iloc[:, j])

        f1_scores.append(ret / total_weight)

    return np.average(f1_scores)


def f1_error(y, y_pred):
    """
    Returns the f1 score of the prediction.
    y: pd.Series with actuals
    y_pred: pd.Series with predictions
    """
    # Ignore predictions where y is NaN
    predictions_to_keep = ~y.isna()
    y = y[predictions_to_keep]
    y_pred = y_pred[predictions_to_keep]
    true_positives = (y & y_pred).sum()

    # Precision: True positives over all positive actuals
    precision = true_positives / y.sum()
    # Recall: True positives over all positive predictions
    if y_pred.sum() == 0:
        recall = 0
    else:
        recall = true_positives / y_pred.sum()

    if (precision + recall) == 0:
        error = 0
    else:
        error = 2 * precision * recall / (precision + recall)

    return error


def scoring_fn_func(df_targets, df_predictions):
    df_targets = df_targets.set_index("window")
    targets = []
    for i in range(0, len(df_targets.columns), 4):
        targets.append(df_targets.iloc[:, i : (i + 4)])

    df_predictions = df_predictions.set_index(df_predictions.columns[0])
    # predictions.columns = predictions.columns.map(lambda _: _.split('.')[0])

    predictions = []
    for i in range(0, len(df_predictions.columns), 4):
        predictions.append(df_predictions.iloc[:, i : (i + 4)])

    return scoring_fn(targets, predictions, weights, f1_error)


if __name__ == "__main__":
    """Scoring Function

    This function is called by Unearthed's SageMaker pipeline. It must be left intact.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--actual", type=str, default="/opt/ml/processing/input/public/public.csv.gz"
    )
    parser.add_argument(
        "--predicted",
        type=str,
        default="/opt/ml/processing/input/predictions/public.csv.out",
    )
    parser.add_argument(
        "--output", type=str, default="/opt/ml/processing/output/scores/public.txt"
    )
    args = parser.parse_args()

    # read the targets, targets are ordered by machine name
    df_actual = pd.read_csv(args.actual, parse_dates=[0])
    print("ACTUAL SHAPE", len(df_actual.columns), len(df_actual))
    df_actual = df_actual.set_index(df_actual.columns[0])
    df_actual.columns = df_actual.columns.map(lambda _: _.split(".")[0])
    targets = []
    for i in range(0, len(df_actual.columns), 4):
        targets.append(df_actual.iloc[:, i : (i + 4)])
    logger.info(f"targets have shape of {targets[0].shape}")

    # read the predictions, predictions are ordered by machine name
    df_pred = pd.read_csv(args.predicted, parse_dates=[0], header=None)
    print("PRED SHAPE", len(df_pred.columns), len(df_pred))
    df_pred = df_pred.set_index(df_pred.columns[0])
    # df_pred.columns = df_pred.columns.map(lambda _: _.split('.')[0])

    predictions = []
    for i in range(0, len(df_pred.columns), 4):
        predictions.append(df_pred.iloc[:, i : (i + 4)])

    score = scoring_fn(targets, predictions, weights, f1_error)

    # write to the output location
    with open(args.output, "w") as f:
        f.write(str(score))
