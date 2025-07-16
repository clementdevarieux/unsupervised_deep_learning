import numpy as np
import pandas as pd


class DataNormalizer:
    def __init__(self, method=None):
        self.method = method
        self.min_val = None
        self.max_val = None
        self.is_fitted = False

    def _to_numpy(self, data):
        if isinstance(data, (pd.DataFrame, pd.Series)):
            return data.values
        elif hasattr(data, 'values'):
            return data.values
        else:
            return np.array(data)

    def fit(self, data):
        data_array = self._to_numpy(data)

        self.min_val = data_array.min()
        self.max_val = data_array.max()
        self.is_fitted = True
        return self

    def transform(self, data):
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before transforming data")

        data_array = self._to_numpy(data)

        if self.method == "sigmoid" or self.method == "relu":
            return (data_array - self.min_val) / (self.max_val - self.min_val)
        elif self.method == "tanh":
            normalized = (data_array - self.min_val) / (self.max_val - self.min_val)
            return 2 * normalized - 1
        return data_array

    def fit_transform(self, data):
        return self.fit(data).transform(data)

    def inverse_transform(self, data):
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before inverse transforming data")

        data_array = self._to_numpy(data)

        if self.method == "sigmoid" or self.method == "relu":
            return data_array * (self.max_val - self.min_val) + self.min_val
        elif self.method == "tanh":
            normalized = (data_array + 1) / 2
            return normalized * (self.max_val - self.min_val) + self.min_val
        return data_array