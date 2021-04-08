import logging

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# from preprocess import preprocess

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class EnsembleModel:
    def __init__(self):
        self.targets = ["y_1", "y_4", "y_12", "y_24"]
        self.models = [RandomForestClassifier(10) for _ in self.targets]

    def train(self, X_learning, y_learning):
        for model, target in zip(self.models, self.targets):
            logger.info(f"Train {target}")
            mask = (~y_learning[target].isna()) & y_learning.operating
            X_learning_tmp = X_learning[mask]
            y_learning_tmp = y_learning[mask][target]
            model.fit(X_learning_tmp.fillna(0), y_learning_tmp)

    def predict(self, inputs):
        print("INPUTS SHAPE", len(inputs.columns), len(inputs))
        inputs = inputs.rename(columns={inputs.columns[0]: "timestamp"})
        # Cast to date
        inputs.iloc[:, 0] = pd.to_datetime(inputs.iloc[:, 0])
        inputs = inputs.set_index(inputs.columns[0])
        inputs.columns = inputs.columns.map(lambda _: _.split(".")[0])
        x_inputs = []
        for i in range(0, len(inputs.columns), 106):
            x_inputs.append(inputs.iloc[:, i : (i + 106)])

        # do predictions
        y_predict = [dict() for _ in x_inputs]
        for model, target in zip(self.models, self.targets):
            for y_machine, x_machine in zip(y_predict, x_inputs):
                # Drop NA useless no? we already did it previously
                # Should we cast to bool?
                y_machine[target] = model.predict(x_machine.fillna(0).values).astype(
                    bool
                )
        predictions = [
            pd.DataFrame(y, index=x.dropna().index, columns=self.targets)
            for x, y in zip(x_inputs, y_predict)
        ]
        import numpy as np

        # combine into one dataframe, machines are sorted in alphabetical order
        order = np.argsort([x.index.name for x in x_inputs])
        predictions = [predictions[i][self.targets] for i in order]
        predictions = pd.concat(predictions, axis=1)
        predictions = predictions.reset_index()
        print("PREDICTIONS SHAPE", len(predictions.columns), len(predictions))

        return predictions
