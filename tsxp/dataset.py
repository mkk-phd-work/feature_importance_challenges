from skforecast.preprocessing import series_long_to_dict, exog_long_to_dict

import pandas as pd
from pandas import DataFrame
import logging


def split_long_format(series_dict, split_ts="2016-01-01"):
    series_dict_train = {k: v.loc[:split_ts,] for k, v in series_dict.items()}
    series_dict_test = {k: v.loc[split_ts:,] for k, v in series_dict.items()}
    return series_dict_train, series_dict_test


class Dataset:
    @staticmethod
    def get_size(series):
        key = list(series.keys())[0]
        size = series[key].shape[0]
        return size


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
        logging.info(f"Series size: {self.series_size}")
