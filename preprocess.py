"""Preprocess data for the Sound The Alarm challenge.

This script will be invoked in two ways during the Unearthed scoring pipeline:
 - first during model training on the 'public' dataset
 - secondly during generation of predictions on the 'private' dataset
"""
import argparse
import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def preprocess(data_file):
    """Apply preprocessing and featurization steps to each file in the data directory.

    Your preprocessing and feature generation goes here.
    """
    logger.info(f"running preprocess on {data_file}")

    # read the data file
    _, file_extension = os.path.splitext(data_file)
    if file_extension == ".parquet":
        # Read data in 10sec instead of 11mins
        df = pd.read_parquet("./data/public/public.parquet")
    else:
        df = pd.read_csv(
            data_file, parse_dates=["timestamp"], na_values=["Shutdown", "Pt Created"]
        )

    if (os.path.exists("./data/public")) and (
        not os.path.exists("./data/public/public.parquet")
    ):
        # If local and file don't exists yet
        df.to_parquet("./data/public/public.parquet")

    logger.info(f"running preprocess on {df.shape}")
    # (4030366, 1009)

    # We seem to drop binary columns here
    cols_to_keep = [
        col
        for col in df.columns
        if ("BrakeSwitch" not in col)
        and ("NeutralizerSwitch" not in col)
        and ("ParkingBrakeSwitch" not in col)
    ]

    df = df[cols_to_keep]

    logger.info(f"data read from {data_file} has shape of {df.shape}")

    # split data into a list of dataframes, one for each machine
    df = df.sort_values("timestamp").reset_index(drop=True)
    df_machines = split_by_machine(df)

    # generate features
    x_inputs = [_build_x_input(_) for _ in df_machines]
    df_learning = pd.concat(x_inputs)

    # TODO do here rolling FE
    cols = list(df_learning)

    # list_window = [60 * 2 ** i for i in range(6)]
    # Iterate faster atm
    list_window = [60 * 2 ** 3]
    for window in list_window:
        cols_fe = [f"{col}_mean_{window}" for col in cols]
        df_learning[cols_fe] = df_learning.groupby(["machine"])[cols].transform(
            lambda x: x.rolling(window, min_periods=window).mean()
        )

    # Drop useless columns
    df_learning = df_learning.drop(columns=cols)

    # generate targets
    actuals = [generate_actuals(df_machine) for df_machine in df_machines]
    y_learning = pd.concat(actuals)
    y_learning = y_learning.reset_index()

    # Yes concat work it is correctly aligned
    # nrow_before = df_learning.shape[0]
    # df_learning = pd.concat([df_learning, df_y], axis=1)
    # assert df_learning.shape[0] == nrow_before

    df_learning = df_learning.reset_index()

    return df_learning, y_learning


def split_by_machine(df):
    """Creates one dataframe by machine with machine as column (wide to long)

    Args:
        df ([type]): [description]

    Returns:
        [type]: [description]
    """

    df = (
        # Set the timestamp as index so we don't need to worry about it as a column
        df.set_index("timestamp")
        # Only keep columns containing a period
        .pipe(lambda _: _[[col for col in _.columns if len(col.split(".")) == 2]])
    )

    # Turn the machine name into a multiindex
    # The columns will then be:
    #
    # LD115                   LD118                   LD119
    # column1, column2, ...   column1, column2, ...   column1, column2, ...
    df.columns = pd.MultiIndex.from_tuples(
        [(col.split(".")[0], col.split(".")[1]) for col in df.columns]
    )

    # Extract the machine names
    machines = sorted({machine for machine in df.columns.get_level_values(level=0)})

    # For each machine, "cut" out the relevant columns
    ret = []
    for machine in machines:
        df_machine = df[[(machine)]].copy()
        # Drop the "machine" multiindex level, so df_machine has columns:
        #
        # column1, column2, ...
        df_machine.columns = df_machine.columns.droplevel()
        df_machine["machine"] = machine
        ret.append(df_machine)

    return ret


def generate_actuals(df_machine):
    """df_machine is the dataframe for one machine"""
    return (
        df_machine.assign(window=lambda _: _.index.floor("min"))
        .assign(
            operating=lambda _: (_["AccelPedalPos1"] > 98)
            & (_["EngSpeed"] > 1800)
            & (_["ActualEngPercentTorque"] > 98)
        )
        .assign(performing=lambda _: _["EngTurboBoostPress"] > 140)
        # y_0 is whether the machine is operating and performing
        # right now. Important to do this *before* aggregating,
        # as otherwise it might be operating and performing within
        # the same minute, but not necessarily at the same time
        .assign(y_0=lambda _: _["operating"] & _["performing"])
        .groupby(["machine", "window"])[["operating", "performing", "y_0"]]
        .max()
        .assign(y_1=lambda _: _["y_0"].shift(periods=-1 * 60))
        .assign(y_4=lambda _: _["y_0"].shift(periods=-4 * 60))
        .assign(y_12=lambda _: _["y_0"].shift(periods=-12 * 60))
        .assign(y_24=lambda _: _["y_0"].shift(periods=-24 * 60))
    )


def _build_x_input(df):
    df = df.copy()
    input_columns = df.columns.values.tolist()
    # Aggregate data at minute granularity
    df["window"] = df.index.floor("min")
    df = df.groupby(["machine", "window"])[input_columns].agg(["mean", "max"])
    df = df.fillna(0.0)  # TODO leave NA
    df.columns = ["_".join(_) for _ in df.columns]
    return df


if __name__ == "__main__":
    """Preprocess Main

    The main function is called by both Unearthed's SageMaker pipeline and the
    Unearthed CLI's "unearthed preprocess" command.

    WARNING - modifying this file may cause the submission process to fail.
    """

    if os.path.exists("/opt/ml/processing/input/public/public.csv.gz"):
        default_input = "/opt/ml/processing/input/public/public.csv.gz"
    else:
        default_input = "./data/public/public.parquet"

    if os.path.exists("/opt/ml/processing/input/public/public.csv.gz"):
        default_output = "/opt/ml/processing/output/preprocess/public.csv"
    else:
        default_output = "./data/public/public.csv"

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=default_input)
    parser.add_argument("--output", type=str, default=default_output)
    args, _ = parser.parse_known_args()

    # call preprocessing on private data
    df_learning, _ = preprocess(data_file=args.input)

    # write to the output location
    # to_csv is super slow
    df_learning.to_csv(args.output)
