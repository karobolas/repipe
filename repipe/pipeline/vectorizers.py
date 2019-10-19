import logging

import pandas as pd

from joblib import Parallel, delayed
from scipy.sparse import csr_matrix, vstack
from sklearn.feature_extraction.text import HashingVectorizer

from .base import FitTransformMixin


logger = logging.getLogger('pipeline')


class HashingVectorizerAdapter(FitTransformMixin):
    def __init__(self, **kwargs):
        self._encoder = HashingVectorizer(**kwargs)

    def transform(self, X: pd.Series) -> csr_matrix:
        logging.debug('HashingVectorizerAdapter::transform - Start')
        try:
            if len(X) > 1000:
                logging.debug('HashingVectorizerAdapter::transform - Executing parallel transform')
                sub_parts = Parallel(n_jobs=-1, max_nbytes='512K', mmap_mode='w+')(
                    delayed(self._encoder.transform)(X.iloc[i:i + 1000])
                    for i in range(0, len(X), 1000)
                )

                logging.debug('HashingVectorizerAdapter::transform - Concatenating sub-parts')
                final = vstack(sub_parts)
            else:
                final = self._encoder.transform(X)

            return final
        finally:
            logging.debug('HashingVectorizerAdapter::transform - Done')

    @property
    def params(self):
        return {
            'n_features': self._encoder.n_features,
            'lowercase': self._encoder.lowercase,
            'analyzer': self._encoder.analyzer,
            'ngram_range': list(self._encoder.ngram_range)
        }

