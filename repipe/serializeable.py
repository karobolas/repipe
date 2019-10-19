import logging
import importlib
from abc import ABCMeta, abstractproperty
from typing import Union, List, Dict, Any


logger = logging.getLogger('serializable')


def locate_class(path):
    parts = path.split('.')
    mod = importlib.import_module('.'.join(parts[:-1]))
    return getattr(mod, parts[-1])


class Serializable(object, metaclass=ABCMeta):
    @abstractproperty
    def params(self):
        pass

    def to_dict(self) -> Dict[str, Any]:
        return {
            'instance': {
                'cls': '.'.join([self.__class__.__module__, self.__class__.__name__]),
                'params': self.params
            }
        }

    @staticmethod
    def load(config:Union[List, Dict]):
        def load_instance(cls, params={}):
            cls = locate_class(cls)
            if 'KerasTokenizerAdapter' not in str(cls):
                params = Serializable.load(params)
            return cls(**params)

        if type(config) is dict:
            if 'instance' in config:
                return load_instance(**config['instance'])

            return {k: Serializable.load(v) for k, v in config.items()}

        elif type(config) is list:
            return [Serializable.load(v) for v in config]

        else:
            return config
