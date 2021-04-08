"""Preprocess data for the Sound The Alarm challenge.

This script will be invoked in two ways during the Unearthed scoring pipeline:
 - first during model training on the 'public' dataset
 - secondly during generation of predictions on the 'private' dataset
"""
import argparse
import logging

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def preprocess(data_file):
    """Apply preprocessing and featurization steps to each file in the data directory.
    
    Your preprocessing and feature generation goes here.
    """
    logger.info(f'running preprocess on {data_file}')

    # read the data file
    df = pd.read_csv(data_file, parse_dates=['timestamp'], na_values=['Shutdown', 'Pt Created'])

    cols_to_keep = [col for col in df.columns if
                    ('BrakeSwitch' not in col) and ('NeutralizerSwitch' not in col) and ('ParkingBrakeSwitch' not in col)]
    df = df[cols_to_keep]

    logger.info(f"data read from {data_file} has shape of {df.shape}")
    
    # split data into a list of dataframes, one for each machine
    df = df.sort_values('timestamp').reset_index(drop=True)
    df_machines = split_by_machine(df)

    # generate features
    x_inputs = [_build_x_input(_) for _ in df_machines]

    # generate targets
    actuals = [generate_actuals(df_machine) for df_machine in df_machines]

    logger.info(f"features after preprocessing has {len(x_inputs)} machines and shape of {x_inputs[0].shape}")
    logger.info(f"target after preprocessing has {len(actuals)} machines and shape of {actuals[0].shape}")
    return x_inputs, actuals


def split_by_machine(df):
    df = (
        # Set the timestamp as index so we don't need to worry about it as a column
        df.set_index('timestamp')
        # Only keep columns containing a period
        .pipe(lambda _: _[[col for col in _.columns if len(col.split('.')) == 2]])
    )

    # Turn the machine name into a multiindex
    # The columns will then be:
    #
    # LD115                   LD118                   LD119
    # column1, column2, ...   column1, column2, ...   column1, column2, ...
    df.columns = pd.MultiIndex.from_tuples([(col.split('.')[0], col.split('.')[1]) for col in df.columns])

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
        # Set the name of the column index to the machine name just to keep track
        df_machine.columns.name = machine
        ret.append(df_machine)

    return ret


def generate_actuals(df_machine):
    """df_machine is the dataframe for one machine"""
    return (
        df_machine
        .assign(window=lambda _: _.index.floor('min'))
        .assign(operating=lambda _: (_['AccelPedalPos1'] > 98) & (_['EngSpeed'] > 1800) & (_['ActualEngPercentTorque'] > 98))
        .assign(performing=lambda _: _['EngTurboBoostPress'] > 140)
        # y_0 is whether the machine is operating and performing
        # right now. Important to do this *before* aggregating,
        # as otherwise it might be operating and performing within
        # the same minute, but not necessarily at the same time
        .assign(y_0=lambda _: _['operating'] & _['performing'])
        .groupby('window')[['operating', 'performing', 'y_0']].max()
        .assign(y_1=lambda _: _['y_0'].shift(periods=-1*60))
        .assign(y_4=lambda _: _['y_0'].shift(periods=-4*60))
        .assign(y_12=lambda _: _['y_0'].shift(periods=-12*60))
        .assign(y_24=lambda _: _['y_0'].shift(periods=-24*60))
    )


def _build_x_input(df):
    df = df.copy()
    input_columns = df.columns.values.tolist()
    df['window'] = df.index.floor('min')
    df = df.groupby('window')[input_columns].agg(['mean', 'max'])
    df = df.fillna(0.0)
    df.columns = ['_'.join(_) for _ in df.columns]
    return df


if __name__ == '__main__':
    """Preprocess Main

    The main function is called by both Unearthed's SageMaker pipeline and the
    Unearthed CLI's "unearthed preprocess" command.
    
    WARNING - modifying this file may cause the submission process to fail.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/opt/ml/processing/input/public/public.csv.gz')
    parser.add_argument('--output', type=str, default='/opt/ml/processing/output/preprocess/public.csv')
    args, _ = parser.parse_known_args()

    # call preprocessing on private data
    df, _ = preprocess(args.input)

    # write to the output location
    pd.concat(df, axis=1).to_csv(args.output)

