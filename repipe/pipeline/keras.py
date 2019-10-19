import logging
from typing import List

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.text import hashing_trick
from keras_preprocessing.sequence import pad_sequences

from .base import FitTransformMixin


logger = logging.getLogger('pipeline')


class KerasTokenizerAdapter(FitTransformMixin):
    def __init__(self, **kwargs):
        kwargs['oov_token'] = '<mis>'

        tok = Tokenizer(oov_token=kwargs['oov_token'])
        word_counts = kwargs.pop('word_counts', tok.word_counts)
        word_docs = kwargs.pop('word_docs', tok.word_docs)
        index_docs = kwargs.pop('index_docs', tok.index_docs)
        index_word = kwargs.pop('index_word', tok.index_word)
        word_index = kwargs.pop('word_index', tok.word_index)

        self._encoder = Tokenizer(**kwargs)
        self._encoder.word_counts = word_counts
        self._encoder.word_docs = word_docs
        self._encoder.index_docs = index_docs
        self._encoder.word_index = word_index
        self._encoder.index_word = index_word

    def fit(self, texts):
        self._encoder.fit_on_texts(['<eos> <pad>'])
        self._encoder.fit_on_texts(texts)

        special = {'<pad>': 0, '<mis>': 1, '<eos>': 2}
        vocab = sorted(
            [(k, f,) for k, f in self._encoder.word_counts.items() if k not in special],
            reverse=True,
            key=lambda x: x[1]
        )

        first_slot = max(special.values()) + 1

        word_index = special.copy()
        word_index.update({k: idx + first_slot for idx, (k, _) in enumerate(vocab)})
        self._encoder.word_index = word_index
        self._encoder.index_word = {idx: k for k, idx in word_index.items()}

    def transform(self, X: pd.Series) -> List[List[int]]:
        def _transform(X: pd.Series) -> List[List[int]]:
            eos = self._encoder.word_index['<eos>']
            tokens = self._encoder.texts_to_sequences(X)

            for i in range(len(tokens)):
                tokens[i].append(eos)

            return tokens

        logging.debug('KerasTokenizerAdapter::transform - Start')
        try:
            return _transform(X)
        finally:
            logging.debug('KerasTokenizerAdapter::transform - Done')

    @property
    def params(self):
        return {
            'num_words': self._encoder.num_words,
            'filters': self._encoder.filters,
            'lower': self._encoder.lower,
            'split': self._encoder.split,
            'char_level': self._encoder.char_level,
            'oov_token': self._encoder.oov_token,
            'document_count': self._encoder.document_count,
            'word_counts': dict(self._encoder.word_counts),
            'word_docs': dict(self._encoder.word_docs),
            'index_docs': dict(self._encoder.index_docs),
            'index_word': self._encoder.index_word,
            'word_index': self._encoder.word_index
        }


class KerasTextHasher(FitTransformMixin):
    def __init__(self, hash_slots: int):
        self._hash_slots = hash_slots

    def transform(self, series: pd.Series) ->  List[List[int]]:
        def _transform(X: pd.Series) -> List[List[int]]:
            return [
                hashing_trick(text, n=self._hash_slots)
                for i, text in enumerate(X.str.lower())
            ]

        logging.debug('TextHasher::transform - Start')
        try:
            if len(series > 1000):
                logging.debug('TextHasher::transform - Executing parallel transform')
                sub_parts = Parallel(n_jobs=-1, max_nbytes='512K', mmap_mode='w+')(
                    delayed(_transform)(series.iloc[i:i + 1000])
                    for i in range(0, len(series), 1000)
                )

                logging.debug('TextHasher::transform - Concatenating sub-parts')
                final = sum(sub_parts, [])
            else:
                final = _transform(series)

            return final
        finally:
            logging.debug('TextHasher::transform - Done')

    @property
    def params(self):
        return {
            'hash_slots': self._hash_slots
        }


class KerasPadSequencesAdapter(FitTransformMixin):
    def __init__(self, **kwargs):
        self._params = kwargs

    def transform(self, X: List[List[int]]) -> np.array:
        logging.debug('KerasPadSequencesAdapter::transform - Start')
        try:
            return pad_sequences(X, **self._params)
        finally:
            logging.debug('KerasPadSequencesAdapter::transform - Done')

    @property
    def params(self):
        return {
            **self._params
        }
