import re
import logging
from typing import List
from functools import reduce

import nltk
import pandas as pd
from joblib import Parallel, delayed

from .base import FitTransformMixin


nltk.download('punkt', quiet=True)
logger = logging.getLogger('pipeline')


class TextScrubber(FitTransformMixin):
    def __init__(
            self,
            lower=False,
            tokenize=False,
            strip_line_break=True,
            filters='()[]{}<>$&%#|-+=*_─…•—–"\'’/\\“ °´®”̈~¿'
         ):
        self._lower = lower
        self._filters = filters
        self._tokenize = tokenize
        self._strip_line_break = strip_line_break

        self._scrubbers = []

        if strip_line_break:
            self._scrubbers.append((re.compile('[\r\n]+'), '. ',))

        self._scrubbers.extend([
            (re.compile('[' + re.escape(filters) + ']+'), ' '),
            (re.compile('\.[\. ]*'), ' . '),
            (re.compile('[0-9][0-9- ]*'), ' __NUM__ ')
        ])

    def transform(self, series: pd.Series) -> pd.Series:
        def _transform(X: pd.Series):
            if self._lower:
                X = X.str.lower()

            for regex, subs in self._scrubbers:
                X = X.str.replace(regex, subs)

            result = [
                [tok for tok in nltk.word_tokenize(text) if len(tok)]
                for text in X
            ]

            if not self._tokenize:
                result = list(
                    map(
                        lambda tokens: ' '.join(tokens),
                        result
                    )
                )

            return pd.Series(result)

        logger.debug('TextScrubber::transform - Start')
        try:
            if len(series) > 1000:
                logger.debug('TextScrubber::transform - Executing parallel transform')
                sub_parts = Parallel(n_jobs=-1, max_nbytes='512K', mmap_mode='w+')(
                    delayed(_transform)(series.iloc[i:i + 1000])
                    for i in range(0, len(series), 1000)
                )

                logger.debug('TextScrubber::transform - Concatenating sub-parts')
                final = pd.concat(sub_parts, axis=0)
            else:
                final = _transform(series)

            return final
        finally:
            logger.debug('TextScrubber::transform - Done')

    @property
    def params(self):
        return {
            'lower': self._lower,
            'tokenize': self._tokenize,
            'strip_line_break': self._strip_line_break,
            'filters': self._filters
        }


class TextFieldUnion(FitTransformMixin):
    def __init__(self, separator=' . '):
        self._sep = separator

    def transform(self, *series: List[pd.Series]) -> pd.Series:
        logger.debug('TextFieldUnion::transform - Start')
        try:
            return reduce(lambda a, b: (a.fillna('') + self._sep + b.fillna('')), series)
        finally:
            logger.debug('TextFieldUnion::transform - Done')

    @property
    def params(self):
        return {
            'separator': self._sep
        }