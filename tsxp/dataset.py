from skforecast.preprocessing import series_long_to_dict, exog_long_to_dict

import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame
import logging


def split_long_format(series_dict, split_ts="2016-01-01"):
    series_dict_train = {k: v.loc[:split_ts,] for k, v in series_dict.items()}
    series_dict_test = {k: v.loc[split_ts:,] for k, v in series_dict.items()}
    print(series_dict_train)
    return series_dict_train, series_dict_test


class Dataset:
    @staticmethod
    def get_size(series):
        key = list(series.keys())[0]
        size = series[key].shape[0]
        return size

    @staticmethod
    def get_dates(series):
        key = list(series.keys())[0]
        return series[key].index.min(), series[key].index.max()


class DatasetMapping:
    def __init__(self, series_id: str, index: str, series_value: str, freq: str = "D"):
        self.series_id = series_id
        self.index = index
        self.series_value = series_value
        self.freq = freq


class ForecasterMsDataset(Dataset):
    def __init__(
        self,
        series: DataFrame,
        exog: DataFrame,
        mapping: DatasetMapping,
        split_time: pd.Timestamp,
    ):
        self.series = series
        self.exog = exog
        self.mapping = mapping
        self.split_time = split_time
        self.series_dict: dict = series_long_to_dict(
            series,
            series_id=self.mapping.series_id,
            index=self.mapping.index,
            values=self.mapping.series_value,
            freq=mapping.freq,
        )
        self.exog_dict = exog_long_to_dict(
            self.exog,
            series_id=self.mapping.series_id,
            index=self.mapping.index,
            freq=self.mapping.freq,
        )
        self.series_size = Dataset.get_size(self.series_dict)
        # logging.info(f"Series size: {self.series_size}")
        self.series_dict_train, self.series_dict_test = split_long_format(
            self.series_dict, split_ts=self.split_time
        )
        # Splitting the exog data
        self.exog_dict_train, self.exog_dict_test = split_long_format(
            self.exog_dict, split_ts=self.split_time
        )

        self.train_size = Dataset.get_size(self.series_dict_train)
        self.test_size = Dataset.get_size(self.series_dict_test)

    def show_sizes(self):
        print(
            f"Dataset dates      : {Dataset.get_dates(self.series_dict)}"
            f"  (n={Dataset.get_size(self.series_dict)})"
        )
        print(
            f"Train dates      : {Dataset.get_dates(self.series_dict_train)}"
            f"  (n={Dataset.get_size(self.series_dict_train)})"
        )
        print(
            f"Test dates      : {Dataset.get_dates(self.series_dict_test)}"
            f"  (n={Dataset.get_size(self.series_dict_test)})"
        )

    def plot_series(self):
        # create a single plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        for k, v in self.series_dict_train.items():
            v.plot(title=k, style="-", ax=ax)
        for k, v in self.series_dict_test.items():
            v.plot(title=k, style="--", ax=ax)
