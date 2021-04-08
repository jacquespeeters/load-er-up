import pandas as pd
import numpy as np

class EnsembleModel:
    def __init__(self, models):
        self.models = models

    def predict(self, inputs):
        inputs.iloc[:,0] = pd.to_datetime(inputs.iloc[:,0])
        inputs = inputs.set_index(inputs.columns[0])
        inputs.columns = inputs.columns.map(lambda _: _.split('.')[0])
        x_inputs = []
        for i in range(0, len(inputs.columns), 106):
            x_inputs.append(inputs.iloc[:, i:(i+106)])

        # do predictions
        lags=['y_1', 'y_4', 'y_12', 'y_24']
        y_predict = [dict() for _ in x_inputs]
        for model, lag in zip(self.models, lags):
            for y_machine, x_machine in zip(y_predict, x_inputs):
                y_machine[lag] = model.predict(x_machine.dropna().values).astype(bool)
        predictions = [pd.DataFrame(y, index=x.dropna().index, columns=lags) for x, y in zip(x_inputs, y_predict)]

        # combine into one dataframe, machines are sorted in alphabetical order
        order = np.argsort([x.index.name for x in x_inputs])
        predictions = [predictions[i][lags] for i in order]
        predictions = pd.concat(predictions, axis=1)
        predictions = predictions.reset_index()

        return predictions
