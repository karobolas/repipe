import logging
import pandas as pd
from .base import FitTransformMixin

logger = logging.getLogger('pipeline')


class DateTimePartExtractor(FitTransformMixin):
    def __init__(self, part):
        self._part = part

    def transform(self, series:pd.Series) -> pd.Series:
        return series.astype('datetime64[ns]').dt.__getattribute__(self._part)

    @property
    def params(self):
        return {
            'part': self._part
        }