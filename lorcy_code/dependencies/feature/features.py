from enum import Enum
import copy
from collections import OrderedDict
from typing import Optional, Dict, List, Tuple
from typing import OrderedDict as TypeOrderedDict

import pandas as pd
import numpy as np

from dataclasses import dataclass
from dataclasses import field
from dataclasses_json import dataclass_json

@dataclass_json
@dataclass
class BaseEncoder(object):

    def __init__(self):
        pass

    def fit(self, X):
        pass

    def transform(self, X):
        pass


@dataclass_json
@dataclass
class Normalization(BaseEncoder):
    name: str = 'normlization'
    mean: float = 0
    std: float = 0
    variance: float = 0
    nums: int = 0 # 统计 fit 时的样本数

    def fit(self, X: pd.Series, y=None):
        self.mean = float(X.mean())
        self.std = float(X.std())
        self.variance = self.std**2
        self.nums = int(len(X) - X.isnull().sum())
        return self

    def fit_list(self, X: list, y=None):
        X = np.array(X)
        self.mean = float(np.mean(X))
        self.std = float(np.std(X))
        self.variance = float(np.var(X))
        self.nums = int(len(X))
        return self

    def transform(self, X: pd.Series, y=None, default_value=np.nan):
        return X.map(lambda x: (x - self.mean) / self.std if not np.isnan(x) else default_value)

    def transform_list(self, X: pd.Series, y=None, default_value=np.nan, padding:Optional[int]=None):
        def normalize_list(x, padding):
            result_list = [(x_i - self.mean) / self.std if not np.isnan(x_i) else default_value for x_i in x]
            if padding is None:
                return result_list
            elif len(result_list) >= padding:
                return result_list[:padding]
            else:
                result_list += [0.0]*(padding-len(result_list))
                return result_list

        return X.apply(lambda x: normalize_list(x, padding))


@dataclass_json
@dataclass
class StringLookup(BaseEncoder):
    name: str = 'string_code'
    code_book: dict = field(default_factory=dict)
    decode_book: dict = field(default_factory=dict)
    size: int = 0 # code_book的规模
    offset: int = 0 # 编码的开始位置，例如可以设置为1

    def __len__(self):
        return self.size

    def __getitem__(self, key):
        return self.transform(key)

    def transform(self, key, default=None):
        return self.code_book.get(key, default)
    
    def transform_series(self, X: pd.Series, default=None):
        def map_value_to_id(value, key2id_dict, default_id):
            return key2id_dict.get(value, default_id)
        return X.apply(map_value_to_id, args=(self.code_book, default))
    
    def transform_list(self, X: pd.Series, defaul=None, padding:Optional[int]=None, mask:Optional[Tuple[float, float]]=None):
        def map_list_to_id(list, key2id_dict, default_id, padding, mask):
            result_list = [key2id_dict.get(value, default_id) for value in list]
            if padding is None:
                pass
            elif len(result_list) >= padding:
                result_list = result_list[:padding]
                if mask is None:
                    pass
                else:
                    mask_prob, mask_frac = mask
                    if np.random.rand() > mask_prob:
                        pass
                    else:
                        mask_num = int(padding * mask_frac)
                        mask_pos = np.random.choice(padding, mask_num)
                        result_list = [0 if i in mask_pos else x for i, x in enumerate(result_list)]
            else:
                result_list += [default_id]*(padding-len(result_list))

            return result_list

        return X.apply(map_list_to_id, args=(self.code_book, defaul, padding, mask))

    def inverse_transform(self, val, default=None):
        return self.decode_book.get(str(val), default)

    def fit(self, X: list):
        keys = X
        self.code_book = {key: i + self.offset for i, key in enumerate(keys)}
        self.decode_book = {code: key for key, code in self.code_book.items()}
        self.size = len(self.code_book)


class FeatureType(Enum): 
    Continuous = 1
    Categorical = 2
    KeyValueSequence = 3 # eg [("a", 1.0),("b", 2.0)] 
    CategoricalSequence = 4 # eg [100,201,302,401] 或 ["a","b","d"]
    ContinuousSequence = 5  # eg [12.5,21.2] 


class FeatureDtype(Enum):
    Int32 = 1
    Float32 = 2
    String = 3
    IntList = 4
    FloatList = 5
    StringList = 6
    KeyInt2ValueFloat = 7
    KeyInt2ValueInt = 8
    KeyStr2ValueInt = 9


@dataclass_json
@dataclass
class Feature(object):
    name: str = 'feature' # 对应 样本数据集中列名（dataframe）
    domain: str = 'domain'  # 特征/特征值归属主题
    feature_type: FeatureType = FeatureType.Continuous  # 特征列类型，注意声明的特征类型要和对应特征数据集（dataframe）中对应的列保持一致
    dtype: FeatureDtype = FeatureDtype.Float32 # 特征值类型， 注意声明的特征值类型要和对应特征数据集（dataframe）中对应的列中原始的类型，格式保持一致
    ctegorical_count: int = 0  # Categorical， KeyValueSequence， CategoricalSequence 类特征，类别特征值数量，包含默认值

    normalization: Optional[Normalization] = None # 对Continuous 有意义，用于记录特征列的均值或方差
    category_encode: Optional[StringLookup] = None # 对Categorical， KeyValueSequence， CategoricalSequence 类特征有意义


@dataclass_json
@dataclass
class Features(object):
    features: TypeOrderedDict[str, Feature] = field(default_factory=OrderedDict)
    keys: Optional[List[str]] = field(default_factory=list)  # index:int -> key:str
    key2id: Optional[TypeOrderedDict[str, int]] = field(default_factory=OrderedDict)  # key:str -> index:int

    def __post_init__(self):
        self._update_index()

    def _update_index(self):
        self._update_id2key()
        self._update_key2id()

    def _update_id2key(self):
        self.keys = [k for k, _ in self.features.items()]

    def _update_key2id(self):
        key_id_pair = [(k, i) for i, k in enumerate(self.keys)]
        self.key2id = OrderedDict(key_id_pair)  # {k:i for i, k in enumerate(self.id2key)}

    def get_idtkey(self):
        return self.keys

    def items(self):
        return self.features.items()

    def get_features(self, deepcopy=False) -> OrderedDict:
        if deepcopy:
            return copy.deepcopy(self.features)  # {k:copy.deepcopy(feature) for k,feature in self.features.items()}
        else:
            return self.features

    def add(self, feature: Feature, name=None):
        if name is None:
            self.features[feature.name] = feature
        else:
            self.features[name] = feature

        self._update_index()

    def pop(self, key):
        feature = self.features.pop(key)
        self._update_index()
        return feature

    def __len__(self):
        return len(self.features)

    def __getitem__(self, name):
        return self.features.get(name, None)

    def get_features_keys(self):
        return self.features.keys()


if __name__ == "__main__":
    pass
