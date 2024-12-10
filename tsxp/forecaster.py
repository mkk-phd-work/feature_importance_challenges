# from dataset import ForecasterMsDataset
from tsxp.dataset import ForecasterMsDataset
from lightgbm import LGBMRegressor
from skforecast.model_selection import bayesian_search_forecaster_multiseries
from skforecast.recursive import ForecasterRecursiveMultiSeries
from sklearn.preprocessing import MinMaxScaler
from skforecast.model_selection import TimeSeriesFold
import matplotlib.pyplot as plt


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
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 1),
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

    def __init__(self, data: ForecasterMsDataset, regressor=None, scale=False):
        self.data: ForecasterMsDataset = data
        self.model = get_default_regressor() if regressor is None else regressor
        self.forecaster = self.get_forecaster(scale=scale)
        self.search_space = None
        self.cv = None

    def bayesian_search(self, verbose=False, search_space=None, cv=None, n_trials=5):
        self.search_space = (
            get_default_search_space if search_space is None else search_space
        )
        self.cv = (
            get_default_fold_generator(self.data.get_size(self.data.series_dict))
            if cv is None
            else cv
        )
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

    def get_forecaster(self, scale=True):
        scaler_s = MinMaxScaler() if scale else None
        scaler_e = MinMaxScaler() if scale else None
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
            steps=365,
            exog=self.data.exog_dict_test,
        )
        # plot original series
        # self.data.plot_series()
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        # for k, v in self.data.series_dict_train.items():
        #     v.plot(title=k, style="-", ax=ax)
        for k, v in self.data.series_dict_test.items():
            v.plot(style="-", ax=ax, legend=True)
            # label=k+"-test")
        # plot forecast
        forecast.plot(ax=ax, style="--", label="forecast", legend=True)

    def create_train_xy(self):
        (X, y) = self.forecaster.create_train_X_y(
            self.data.series_dict, exog=self.data.exog_dict
        )
        # split the data by the data.split_time
        split_time = self.data.split_time
        X_train = X.loc[X.index < split_time]
        y_train = y.loc[y.index < split_time]
        X_test = X.loc[X.index >= split_time]
        y_test = y.loc[y.index >= split_time]
        return (X_train, y_train, X_test, y_test)
