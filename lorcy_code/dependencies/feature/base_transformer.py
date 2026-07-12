from dataclasses import dataclass

from dataclasses_json import dataclass_json
import pandas as pd

from .features import Features


@dataclass_json
@dataclass
class BaseTransform(object):

    def __init__(self):
        # self.init_features = None
        self.features = None

    def fit(self, X: pd.DataFrame):
        self.fit_transform(X)

    def fit_transform(self, X: pd.DataFrame):
        pass

    def transform(self, X: pd.DataFrame):
        pass

    def dump(self, path):
        pass

    # def initialize_features(self, init_features:Features ):
    #     self.init_features = init_features
    #     # self.features = copy_features(self.init_features)


def copy_features(features:Features):
    return Features(features.get_features(deepcopy=True))
