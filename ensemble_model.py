import logging

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# from preprocess import preprocess

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class EnsembleModel:
    def __init__(self):
        self.targets = ["y_1", "y_4", "y_12", "y_24"]
        self.models = [RandomForestClassifier(20) for _ in self.targets]

    def get_X(self, df):
        X = df.drop(columns=["machine", "window"])
        return X

    def train(self, df_learning, y_learning):
        for model, target in zip(self.models, self.targets):
            logger.info(f"Train {target}")
            mask = (~y_learning[target].isna()) & y_learning.operating
            X_learning_tmp = self.get_X(df_learning[mask])
            self.X_cols = list(X_learning_tmp)
            y_learning_tmp = y_learning[mask][target]
            model.fit(X_learning_tmp.fillna(0), y_learning_tmp.astype(int))

    def predict(self, df_prod):
        print("INPUTS SHAPE", len(df_prod.columns), len(df_prod))
        print("df_prod.head()", df_prod.head())
        predictions = df_prod[["machine", "window"]].copy()

        for model, target in zip(self.models, self.targets):
            logger.info(f"Predict {target}")
            X_prod_tmp = self.get_X(df_prod)
            X_prod_tmp = X_prod_tmp[self.X_cols]
            predictions[target] = model.predict(X_prod_tmp.fillna(0)).astype(bool)

        machines_names = predictions["machine"].unique().tolist()
        machines_names.sort()

        predictions = pd.pivot(
            predictions,
            columns="machine",
            index=["window"],
            values=self.targets,
        )

        predictions.columns = [f"{col[1]}.{col[0]}" for col in predictions.columns]
        cols = []
        for machine_name in machines_names:
            for target in self.targets:
                cols.append(f"{machine_name}.{target}")

        predictions = predictions.reindex(cols, axis=1)
        predictions = predictions.reset_index()

        print("PREDICTIONS SHAPE", len(predictions.columns), len(predictions))
        print("predictions.head(5)", predictions.head(5))

        return predictions
