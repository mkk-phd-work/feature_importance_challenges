# from dataset import ForecasterMsDataset
from tsxp.dataset import ForecasterMsDataset
from lightgbm import LGBMRegressor
from skforecast.model_selection import bayesian_search_forecaster_multiseries
from skforecast.recursive import ForecasterRecursiveMultiSeries
from sklearn.preprocessing import MinMaxScaler
from skforecast.model_selection import TimeSeriesFold
import matplotlib.pyplot as plt
import pandas as pd


def get_default_regressor(self):
    return LGBMRegressor(
        n_estimators=300,
        device="gpu",
        gpu_platform_id=1,
        gpu_device_id=0,
        random_state=42,
        # seed=42,
    )


def get_default_search_space(trial):
    return {
        "lags": trial.suggest_categorical("lags", [4, [1, 2, 4, 5]]),
        "n_estimators": trial.suggest_int("n_estimators", 526, 526),
        "max_depth": trial.suggest_int("max_depth", 8, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.01),
        # "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 1),
    }


def get_default_fold_generator(self, train_size):
    return TimeSeriesFold(
        steps=356,
        initial_train_size=train_size,
        #  refit                 = True,
        #  fixed_train_size      = True,
        #  allow_incomplete_fold = True
    )


class ForecasterMsExog:

    def __init__(self, data: ForecasterMsDataset, regressor=None, scaler=MinMaxScaler, exog_scaler=MinMaxScaler, scale=True):
        self.data: ForecasterMsDataset = data
        self.model = get_default_regressor() if regressor is None else regressor
        self.forecaster = self.get_forecaster(scaler, exog_scaler, scale)
        self.search_space = None
        self.cv = None

    def bayesian_search(self, verbose=False, search_space=None, cv=None, n_trials=5):
        self.search_space = get_default_search_space if search_space is None else search_space
        self.cv = get_default_fold_generator(self.data.get_size(self.data.series_dict)) if cv is None else cv
        results = bayesian_search_forecaster_multiseries(
            forecaster=self.forecaster,  # ForecasterRecursiveMultiSeries
            series=self.data.series_dict_train,  # target series
            exog=self.data.exog_dict_train,  # exogenous variables
            search_space=self.search_space,  # search space
            n_trials=n_trials,  # number of trials
            verbose=verbose,
            show_progress=True,
            return_best=True,
            metric="mean_absolute_error",
            n_jobs="auto",
            cv=cv,
        )
        return results

    def get_forecaster(self, scaler, exog_scaler, scale):
        scaler_s = scaler if scale else None
         #MinMaxScaler() if scale else None
        scaler_e = exog_scaler if scale else None
        #MinMaxScaler() if scale else None
        forecaster = ForecasterRecursiveMultiSeries(
            regressor=self.model,
            lags=365,
            encoding="ordinal",  # "ordinal" "onehot", # "ordinal_category",
            dropna_from_series=False,
            transformer_series=scaler_s,
            transformer_exog=scaler_e,
            # differentiation=1,
        )
        return forecaster

    def plot_forecast(self):
        forecast = self.forecaster.predict(
            # series_id=series_id,
            steps=self.data.test_size,
            exog=self.data.exog_dict_test,
        )
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        for k, v in self.data.series_dict_test.items():
            v.plot(style="-", ax=ax[0], legend=True, label=k + "-test", linewidth=0.8)
            forecast[k].plot(ax=ax[0], style="--", legend=True, linewidth=0.8, label=k + "-forecast")
        # calculate and plot residuals
        fig, bx = plt.subplots(1, 3, figsize=(15, 5))
        c = 0
        for k, v in self.data.series_dict_test.items():
            residuals = forecast[k].subtract(v)
            residuals.plot(ax=ax[1], style="-", legend=True, label=k + "-residuals", linewidth=0.8)
            residuals.hist(bins=20, legend=True, ax=bx[c])
            c = c + 1

        # print(forecast)
        # print(self.data.series_dict_test)
        # residuals.plot(ax=ax, style="-", legend=True, label="residuals", linewidth=0.8)

    def create_train_xy(self):
        (X, y) = self.forecaster.create_train_X_y(self.data.series_dict, exog=self.data.exog_dict)
        # split the data by the data.split_time
        split_time = self.data.split_time
        X_train = X.loc[X.index <= split_time]
        y_train = y.loc[y.index <= split_time]
        X_test = X.loc[X.index > split_time]
        y_test = y.loc[y.index > split_time]
        return (X_train, y_train, X_test, y_test)

    def calculate_test_performance_metrics(self):
        from sklearn.metrics import (
            r2_score,
            mean_absolute_error,
            mean_absolute_percentage_error,
            mean_squared_error,
        )

        y_pred: pd.DataFrame = (
            self.forecaster.predict(steps=self.data.test_size, exog=self.data.exog_dict_test)
            .melt(ignore_index=False)
            .reset_index()
            .rename(columns={"variable": "variable", "value": "predicted"})
        )

        y_test = (
            pd.DataFrame(self.data.series_dict_test)
            .melt(ignore_index=False)
            .reset_index()
            .rename(columns={"variable": "variable", "value": "real"})
        )

        y_pred = y_pred.merge(y_test, on=["variable", "index"])

        metrics = {
            "MAE": mean_absolute_error,
            "MSE": mean_squared_error,
            "R2": r2_score,
            "MAPE": mean_absolute_percentage_error,
        }

        results = {name: func(y_pred["real"], y_pred["predicted"]) for name, func in metrics.items()}
        results = pd.DataFrame.from_dict(results, orient="index")

        group_results = {
            k: {name: func(y_pred_k["real"], y_pred_k["predicted"]) for name, func in metrics.items()}
            for k, y_pred_k in y_pred.groupby("variable")
        }
        group_results = pd.DataFrame.from_dict(group_results, orient="index")

        return results, group_results
