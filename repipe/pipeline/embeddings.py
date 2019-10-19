import os
import logging
from functools import lru_cache

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

from .base import FitTransformMixin


logger = logging.getLogger('pipeline')


class WordVectorEmbedder(FitTransformMixin):
    def __init__(self, path: str, max_embedding_len: int, dtype='float32'):
        super().__init__()

        self._path = path
        self._max_embedding_len = max_embedding_len
        self._dtype = dtype

        script_path = os.path.dirname(os.path.abspath(__file__))
        root_path = os.path.abspath(script_path + '/../')

        if not os.path.exists(path):
            path = root_path + '/' + path

        self._model = KeyedVectors.load(path)

        # Generate special tokens
        special_tokens = ['<MIS>', '<EOS>', '<PAD>']
        M = np.identity(len(special_tokens), dtype=dtype)
        base_vector = np.zeros(self._model.vector_size, dtype=dtype)

        self._special_tokens = {}
        for i, token in enumerate(special_tokens):
            self._special_tokens[token] = np.append(base_vector, M[i])

    @lru_cache(maxsize=1000000)
    def _get_vector(self, token: str):
        try:
            return np.append(self._model.wv.get_vector(token), [0, 0, 0])
        except KeyError:
            return self._special_tokens['<MIS>']

    def transform(self, series: pd.Series) -> np.array:
        def _transform(X: pd.Series) -> np.array:
            N = len(X)
            M = self._max_embedding_len
            K = self._model.wv.vector_size

            embeddings = np.zeros([N, M, K + 3], dtype=self._dtype)

            for i, text in enumerate(X.str.lower().str.split()):
                L = len(text)
                if L + 1 < M:
                    embeddings[i, L + 0] = self._special_tokens['<EOS>']
                if L + 2 < M:
                    embeddings[i, L + 1:] = self._special_tokens['<PAD>']

                if L > 0:
                    embedding = [self._get_vector(token) for token in text[:M]]
                    embeddings[i, :len(embedding)] = embedding

            return embeddings

        logger.debug('WordVectorEmbedder::transform - Start')
        try:
            return _transform(series)
        finally:
            logger.debug('WordVectorEmbedder::transform - Done')

    @property
    def params(self):
        return {
            'path': self._path,
            'dtype': self._dtype,
            'max_embedding_len': self._max_embedding_len,
        }
