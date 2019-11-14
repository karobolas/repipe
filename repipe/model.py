import os
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from .pipeline import Pipeline
from .serializeable import Serializable


class ModelOutputMapper(Serializable):
    def __init__(
            self,
            classes:Dict[str,List[Dict[str, Any]]],
            mean_f1:float,
            fallback_class:str
    ):
        self._classes = {}
        self._class_types = {}

        for name, mappings in classes.items():
            parts = name.split(':')
            name = parts[-1]
            typ = 'multi-class' if len(parts) == 1 else parts[0]

            # Important to sort by f1 in descending order
            df = pd.DataFrame(mappings).set_index('class_id').sort_values('f1_score', ascending=False)

            # Select the (maximum) number of classes that achieves
            # the minimum average F1 score
            df2 = pd.DataFrame([
                {
                    'mean_f1': df.iloc[:i]['f1_score'].mean(),
                    'coverage': df.iloc[:i]['support'].sum(),
                    'classes': i
                }
                for i in range(1, len(df)+1)
            ])
            selection = df2[df2['mean_f1'] >= mean_f1].iloc[-1].round(5)

            # Map all classes not covered to the fallback class
            df['mapped_to_class'] = df.class_name
            df.loc[df.iloc[int(selection.classes):].index, 'mapped_to_class'] = fallback_class

            self._classes[name] = df.sort_values('class_id')
            self._class_types[name] = typ

            # Stats
            print(f'----{name}----')
            print(df[df.mapped_to_class != 'other'].__repr__())
            print(f'\n\tMean F1: {round(selection.mean_f1,4)}, Coverage:{round(selection.coverage,4)}')
            print()

        # Save
        self._fallback_class = fallback_class
        self._mean_f1 = mean_f1

    def predictions_to_classes(self, y:Dict[str, np.array]) -> Dict[str, List[Dict[str, Any]]]:
        def _f(name):
            classes = self._classes[name]

            def single_label(yp):
                return [
                    {
                        'prediction': classes.loc[id].mapped_to_class,
                        'actual_prediction': classes.loc[id].class_name,
                        'confidence': round(float(confidence), 6)
                    }
                    for id, confidence in zip(yp.argmax(axis=1), yp.max(axis=1))
                ]

            def multi_label(yp):
                return [
                    list(filter(
                        lambda p:p['prediction'] != 'other',
                        [
                            {
                                'prediction': classes.loc[id].mapped_to_class,
                                'actual_prediction': classes.loc[id].class_name,
                                'confidence': round(float(confidence), 6)
                            }
                            for id, confidence in zip(np.where(p)[0], ys[p])
                        ]
                    ))
                    for p, ys in zip(yp.round().astype(bool), yp)
                ]

            if self._class_types[name] == 'multi-label':
                return multi_label
            else:
                return single_label

        return {
            class_name:_f(class_name)(yp)
            for class_name, yp in y.items()
        }


    @property
    def params(self):
        cols = ['class_id', 'class_name', 'f1_score', 'precision', 'recall', 'support']

        return {
            'classes': {
                f'{self._class_types[name]}:{name}':[
                    r.to_dict()
                    for _, r in cls.reset_index(level=0)[cols].sort_values('class_id').iterrows()
                ]
                for name, cls in self._classes.items()
            },
            'mean_f1': round(float(self._mean_f1),3),
            'fallback_class': self._fallback_class
        }


class Model(Serializable):
    def __init__(
            self,
            path,
            pipeline:Pipeline,
            output_mapper:ModelOutputMapper
    ):
        import keras.backend as K
        from keras.models import load_model

        self._path = path
        self._model = load_model(path)
        self._pipeline = pipeline
        self._mapper = output_mapper
        self._tf_graph = K.get_session().graph

        self._labeler = self._map_multi if isinstance(self._model.output_shape, list) else self._map_single

    def _map_single(self, Y):
        Y = {self._model.output_names[0]:Y}
        return self._mapper.predictions_to_classes(Y)

    def _map_multi(self, Y):
        Y = dict(zip(self._model.output_names, Y))
        return self._mapper.predictions_to_classes(Y)

    def predict(self, obj):
        with self._tf_graph.as_default():
            X = self._pipeline.transform(obj)
            return self._labeler(self._model.predict(X, batch_size=1000))

    @property
    def params(self):
        return {
            'path': self._path,
            'pipeline': self._pipeline.to_dict(),
            'output_mapper': self._mapper.to_dict()
        }
