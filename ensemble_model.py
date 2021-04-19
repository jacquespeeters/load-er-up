import logging

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd

from score import scoring_fn_func

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
        self.N_BAGGING = 4
        self.N_FOLD = 3

    def get_X(self, df):
        X = df.drop(columns=["machine", "window", "FOLD"], errors="ignore")
        return X

    def get_importance_lgb(self, model_gbm, X_cols=None):
        importance = pd.DataFrame()
        if X_cols is None:
            importance["feature"] = model_gbm.feature_name_
        else:
            importance["feature"] = X_cols
        importance["importance"] = model_gbm.feature_importances_
        importance["importance"] = (
            importance["importance"] / importance["importance"].replace(np.inf, 0).sum()
        )
        importance["importance"] = importance["importance"] * 100
        importance["importance_rank"] = importance["importance"].rank(
            ascending=False
        )  # .astype(int)
        importance = importance.sort_values("importance_rank").round(2)
        return importance

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
        print(
            f"Expected best threshold is {best_tresh.round(3)}, Expected best f-score {round(max(fscore),3)}"  # noqa
        )
        return preds > best_tresh

    def predict_optim_f1(self, single_machine_pred):
        for target in self.targets:
            single_machine_pred[target] = self.expected_optim_f1(
                single_machine_pred[target]
            )
        return single_machine_pred

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
        predictions = df_learning[["window", "machine"]].copy()
        predictions[self.targets] = 0

        # Create folder by combining time and machine split
        N_MACHINES = df_learning["machine"].nunique()
        mlflow.set_experiment("training")
        mlflow.start_run()
        X_learning = self.get_X(df_learning)
        self.X_cols = list(X_learning)

        models = {}
        train_loss = []
        valid_loss = []
        for target in self.targets:
            model_best_score_training = []
            model_best_score_valid = []
            models_BAGGING = {}
            for BAGGING in range(self.N_BAGGING):
                models_FOLD = {}
                np.random.seed(BAGGING)
                code_machine = np.random.choice(
                    range(N_MACHINES), N_MACHINES, replace=False
                )
                code_machine = dict(zip(df_learning["machine"].unique(), code_machine))

                df_learning["FOLD"] = (
                    df_learning["machine"].map(code_machine)
                    + pd.qcut(df_learning["window"], self.N_FOLD, labels=False)
                ).mod(self.N_FOLD)

                for FOLD in range(self.N_FOLD):
                    logger.info(f"Train {target}, Bagging {BAGGING}, Folder {FOLD}")
                    model = lgb.LGBMClassifier(importance_type="gain")
                    y = y_learning[target]
                    # Split should be by machine to mimic private leaderboard
                    is_valid = df_learning["FOLD"] == FOLD
                    X_train, y_train = X_learning[~is_valid], y[~is_valid]
                    X_valid, y_valid = X_learning[is_valid], y[is_valid]

                    # & y_learning.operating => aparently we train on this given forum input # noqa
                    # mask = ~y_learning[target].isna()

                    X_train, y_train = (
                        X_train[y_train.notnull()],
                        y_train[y_train.notnull()].astype(int),
                    )
                    X_valid, y_valid = (
                        X_valid[y_valid.notnull()],
                        y_valid[y_valid.notnull()].astype(int),
                    )

                    model.set_params(
                        **{
                            "min_data_in_leaf": int(X_train.shape[0] / 100),
                            "n_estimators": 2000,
                        }
                    )

                    model.fit(
                        X_train,
                        y_train,
                        eval_set=[(X_train, y_train), (X_valid, y_valid)],
                        early_stopping_rounds=20,
                        verbose=10,
                    )

                    predictions.loc[is_valid, target] += (
                        model.predict_proba(X_learning[is_valid])[:, 1] / self.N_BAGGING
                    )
                    model_best_score_training.append(
                        model.best_score_["training"]["binary_logloss"]
                    )
                    model_best_score_valid.append(
                        model.best_score_["valid_1"]["binary_logloss"]
                    )
                    models_FOLD[FOLD] = model

                models_BAGGING[BAGGING] = models_FOLD

            models[target] = models_BAGGING

            print(
                lgb.plot_importance(model, importance_type="gain", max_num_features=20)
            )
            train_loss.append(pd.Series(model_best_score_training).mean())
            mlflow.log_metric(
                f"train_loss_{target}", pd.Series(model_best_score_training).mean()
            )
            valid_loss.append(pd.Series(model_best_score_valid).mean())
            mlflow.log_metric(
                f"valid_loss_{target}", pd.Series(model_best_score_valid).mean()
            )
            mlflow.log_metric(f"best_iteration_{target}", model._best_iteration)
            # mlflow.log_artifact(path.path_feature_importance())

        self.models = models

        train_loss = pd.Series(train_loss).mean()
        valid_loss = pd.Series(valid_loss).mean()

        predictions = predictions.groupby("machine").apply(self.predict_optim_f1)
        predictions = self.format_predictions(predictions)
        targets = self.format_predictions(y_learning)
        fscore = scoring_fn_func(targets, predictions)
        mlflow.log_metric("train_loss", train_loss)
        mlflow.log_metric("valid_loss", valid_loss)
        mlflow.log_metric("fscore", fscore)
        logger.info(f"Out-of-fold fscore is {round(fscore, 3)}")
        mlflow.end_run()

        return predictions

    def predict(self, df_prod):
        predictions = df_prod[["machine", "window"]].copy()
        predictions[self.targets] = 0

        X_prod_tmp = self.get_X(df_prod)[self.X_cols]
        N_MODEL = self.N_FOLD * self.N_BAGGING
        for target in self.targets:
            logger.info(f"Predict {target}")
            for BAGGING in range(self.N_BAGGING):
                for FOLD in range(self.N_FOLD):
                    model = self.models[target][BAGGING][FOLD]
                    predictions[target] += (
                        model.predict_proba(X_prod_tmp)[:, 1] / N_MODEL
                    )

        predictions = predictions.groupby("machine").apply(self.predict_optim_f1)
        predictions = self.format_predictions(predictions)
        return predictions

    def get_feature_importance(self):
        importance = []
        for target in self.targets:
            for BAGGING in range(self.N_BAGGING):
                for FOLD in range(self.N_FOLD):
                    model = self.models[target][FOLD]
                    importance_tmp = self.get_importance_lgb(model)
                    importance_tmp["FOLD"] = FOLD
                    importance_tmp["target"] = target
                    importance.append(importance_tmp)
        return pd.concat(importance)
