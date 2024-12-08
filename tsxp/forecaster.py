# from dataset import ForecasterMsDataset
from tsxp.dataset import ForecasterMsDataset
from lightgbm import LGBMRegressor
from skforecast.model_selection import bayesian_search_forecaster_multiseries
from skforecast.recursive import ForecasterRecursiveMultiSeries
from sklearn.preprocessing import MinMaxScaler


def get_default_regressor(self):

    return LGBMRegressor(
        n_estimators=300,
        device="gpu",
        gpu_platform_id=1,
        gpu_device_id=0,
        random_state=42,
        # seed=42,
    )


class ForecasterMsExog:

    def __init__(
        self,
        data: ForecasterMsDataset,
        regressor=None,
        scale=False,
        # search_space=None
    ):
        self.data: ForecasterMsDataset = data
        self.model = get_default_regressor() if regressor is None else regressor
        self.forecaster = self.get_forecaster(scale=scale) 
        self.search_space = None

    def bayesian_search(self, verbose=False, search_space=None, n_trials=5):
        def default_search_space(trial):
            return {
                "lags": trial.suggest_categorical("lags", [4, [1, 2, 4, 5]]),
                "n_estimators": trial.suggest_int("n_estimators", 526, 526),
                "max_depth": trial.suggest_int("max_depth", 8, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.01),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 1),
            }

        self.search_space = default_search_space if search_space is None else search_space
        results = bayesian_search_forecaster_multiseries(
            forecaster=self.forecaster,
            y=self.data.series_dict,
            exog=self.data.exog_dict,
            param_grid=self.search_space,
            n_trials=n_trials,
            verbose=verbose,
            show_progress=True,
        )
        return results

    def get_forecaster(self, scale=True):
        scaler_s = MinMaxScaler() if scale else None
        scaler_e = MinMaxScaler() if scale else None
        forecaster = ForecasterRecursiveMultiSeries(
            regressor=self.model,
            lags=4,
            encoding="ordinal",  # "ordinal" "onehot", # "ordinal_category",
            dropna_from_series=False,
            transformer_series=scaler_s,
            transformer_exog=scaler_e,
            # differentiation=1,
        )
        return forecaster
