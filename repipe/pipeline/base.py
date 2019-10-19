import logging
from abc import ABCMeta, abstractmethod
from typing import List, Dict, Any, Union


import pandas as pd

from ..utils import Timer
from ..serializeable import Serializable


logger = logging.getLogger('pipeline')


class FitTransformMixin(Serializable, metaclass=ABCMeta):
    def fit(self, *args):
        pass

    @abstractmethod
    def transform(self, X):
        pass


class TransformStep(FitTransformMixin):
    def __init__(
            self,
            out_field: str,
            in_fields: Union[str, List[str]],
            transform: FitTransformMixin
    ):
        super().__init__()

        if type(in_fields) is not list:
            in_fields = [in_fields]

        self._out_field = out_field
        self._in_fields = in_fields
        self._transformer = transform

    def fit(self, obj: Dict[str, Union[pd.Series, Any]]) -> None:
        with Timer() as t:
            fields = [obj[name] for name in self._in_fields]
            self._transformer.fit(*fields)
        logger.info(f'Finished fit-step {self._out_field}  in {int(t.elapsed)} ms')

    def transform(self, obj: Dict[str, Union[pd.Series, Any]]) -> Dict[str, Union[pd.Series, Any]]:
        with Timer() as t:
            fields = [obj[name] for name in self._in_fields]
            obj[self._out_field] = self._transformer.transform(*fields)
        logger.info(f'Finished step {self._out_field}  in {int(t.elapsed)} ms')
        return obj

    @property
    def params(self):
        return {
            'out_field': self._out_field,
            'in_fields': self._in_fields,
            'transform': self._transformer.to_dict()
        }


class FeatureSelector(FitTransformMixin):
    def __init__(self, features: List[str]):
        super().__init__()
        self._features = features

    def transform(self, obj: Dict[str, Any]) -> List[Any]:
        return [
            obj[name]
            for name in self._features
        ]

    @property
    def params(self):
        return {
            'features': self._features
        }


class Pipeline(FitTransformMixin):
    def __init__(self, steps: List[FitTransformMixin]):
        super().__init__()
        self._steps = steps

    def fit(self, df: pd.DataFrame) -> None:
        obj = {name: series for name, series in df.iteritems()}
        for step in self._steps:
            step.fit(obj)
            obj = step.transform(obj)

        return obj

    def transform(self, df: pd.DataFrame) -> Any:
        obj = {name: series for name, series in df.iteritems()}
        for step in self._steps:
            obj = step.transform(obj)

        return obj

    @property
    def params(self):
        return {
            'steps': [step.to_dict() for step in self._steps]
        }
