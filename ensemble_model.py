import logging

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# How to run Mlflow:
# > mlflow ui
# How to make port redirection from DS1 to local to access MLflow
# > ssh -N -f -L localhost:5000:localhost:5000 jpeeters@ds1
# (replace jpeeters with your username)

# from preprocess import preprocess

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


class EnsembleModel:
    def __init__(self):
        self.targets = ["y_1", "y_4", "y_12", "y_24"]
        self.models = [lgb.LGBMClassifier(importance_type="gain") for _ in self.targets]

    def get_X(self, df):
        X = df.drop(columns=["machine", "window", "FOLD"], errors="ignore")
        return X

    def expected_optim_f1(self, preds):
        """Optimize expected threshold

        Args:
            preds ([type]): [description]

        Returns:
            [type]: boolean preds given best expected threshold
        """
        expected_sum = preds.sum()
        expected_sum

        threshold = preds.quantile([i / 100 for i in range(0, 100)]).unique()
        fscore = []
        for thresh in threshold:
            thresh
            precision = preds[preds > thresh].mean()
            recall = preds[preds > thresh].sum() / expected_sum
            tmp_fscore = 2 * (precision * recall) / (precision + recall)
            fscore.append(tmp_fscore)

        best_tresh = threshold[np.argmax(fscore)]
        print(f"Expected best threshold is {best_tresh.round(3)}")
        print(f"Expected best f-score {round(max(fscore),3)}")
        return preds > best_tresh

    def format_predictions(self, predictions):
        """Format predictions to be compliant with competition one

        Args:
            predictions ([type]): [description]

        Returns:
            [type]: [description]
        """
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
        return predictions

    def train(self, df_learning, y_learning):

        df_learning["FOLD"] = pd.qcut(df_learning["window"], 3, labels=False)

        N_FOLD = 5
        df_learning["FOLD"] = (
            df_learning["machine"].astype("category").cat.codes
            + pd.qcut(df_learning["window"], N_FOLD, labels=False)
        ).mod(N_FOLD)

        # df_learning[mask].groupby("FOLD").size()

        mlflow.set_experiment("training")
        mlflow.start_run()
        for model, target in zip(self.models, self.targets):
            logger.info(f"Train {target}")
            # & y_learning.operating => aparently we train on this given forum input
            mask = ~y_learning[target].isna()
            X_learning_tmp = self.get_X(df_learning[mask])
            self.X_cols = list(X_learning_tmp)
            y_learning_tmp = y_learning[mask][target].astype(int)
            # Split should be by machine to mimic private leaderboard
            is_valid = df_learning[mask]["FOLD"] == 0
            X_train = X_learning_tmp[~is_valid]
            y_train = y_learning_tmp[~is_valid]
            X_valid = X_learning_tmp[is_valid]
            y_valid = y_learning_tmp[is_valid]

            # X_train, X_valid, y_train, y_valid = train_test_split(
            #     X_learning_tmp,
            #     y_learning_tmp,
            #     test_size=0.20,
            #     random_state=42,
            # )

            model.set_params(**{"min_data_in_leaf": 10000})

            model.fit(
                X_train,
                y_train,
                eval_set=[(X_train, y_train), (X_valid, y_valid)],
                early_stopping_rounds=20,
                # eval_metric="auc",
                verbose=10,
            )

            print(
                lgb.plot_importance(model, importance_type="gain", max_num_features=20)
            )
            mlflow.log_metric(
                f"train_loss_{target}", model.best_score_["training"]["binary_logloss"]
            )
            mlflow.log_metric(
                f"valid_loss_{target}", model.best_score_["valid_1"]["binary_logloss"]
            )
            mlflow.log_metric(f"best_iteration_{target}", model._best_iteration)
            # mlflow.log_artifact(path.path_feature_importance())

        mlflow.end_run()

    def predict(self, df_prod):
        predictions = df_prod[["machine", "window"]].copy()

        for model, target in zip(self.models, self.targets):
            logger.info(f"Predict {target}")
            X_prod_tmp = self.get_X(df_prod)
            X_prod_tmp = X_prod_tmp[self.X_cols]
            predictions[target] = model.predict_proba(X_prod_tmp)[:, 1]

        def predict_optim_f1(single_machine_pred):
            for target in self.targets:
                single_machine_pred[target] = self.expected_optim_f1(
                    single_machine_pred[target]
                )
            return single_machine_pred

        predictions = predictions.groupby("machine").apply(predict_optim_f1)
        predictions = self.format_predictions(predictions)

        print("predictions.tail(5)")
        print(predictions.tail(5))

        return predictions
