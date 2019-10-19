import logging
from typing import Union

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import OneHotEncoder

from .base import FitTransformMixin

logger = logging.getLogger('pipeline')


class OneHotEncodingToBinaryEncoding(FitTransformMixin):
    def transform(self, mat:csr_matrix) -> np.array:
        rows = mat.shape[0]
        binary_digits = len(bin(mat.shape[1])[2:])
        res = np.zeros([rows, binary_digits], dtype=np.int8)
        for i in range(rows):
            for index in mat[i].indices:
                for j, bit in enumerate(bin(index)[2:]):
                    if bit == '0':
                        continue
                    res[i, j] = 1
        return res

    @property
    def params(self):
        return {}


class OneHotEncoderAdapter(FitTransformMixin):
    def __init__(self, categories, sparse=False, input_is_numerical=False, dtype='uint8'):
        self._dtype = dtype
        self._is_numerical = input_is_numerical        
        self._encoder = OneHotEncoder(sparse=sparse, dtype=dtype)

        categories = pd.Series(categories)
        if not self._is_numerical:
            categories = categories.astype(str).str.lower()

        self._encoder.fit(categories.values.reshape(-1,1))

    def transform(self, X:pd.Series) -> Union[np.ndarray, csr_matrix]:
        logging.debug('OneHotEncoderAdapter::transform - Start')
        try:
            categories = self._encoder.categories_[0]
            if not self._is_numerical:
                X = X.str.lower().fillna('')
                X[~X.isin(categories)] = ''
            return self._encoder.transform(X.values.reshape(-1,1))
        finally:
            logging.debug('OneHotEncoderAdapter::transform - Done')

    @property
    def params(self):
        return {
            'categories': sorted(self._encoder.categories_[0].tolist()),
            'dtype': self._dtype,
            'sparse': self._encoder.sparse,
            'input_is_numerical': self._is_numerical
        }
