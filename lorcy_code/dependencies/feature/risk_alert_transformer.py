from dataclasses import dataclass, field
from typing import List, Dict

from dataclasses_json import dataclass_json
import pandas as pd

from .features import FeatureDtype, Features, Feature, FeatureType
from .features import StringLookup, Normalization
from .base_transformer import BaseTransform

from lorcy_code.dependencies.config import base_config

@dataclass_json
@dataclass
class RiskAlertTransform(BaseTransform):
    
    features: Features = None  # 必须有
    categorical_feature_names: List[str] = None
    continuous_feature_names: List[str] = None
    unchanged_feature_names: List[str] = None
    transform_feature_names: List[str] = None
    # 新增：特征分组配置，用于将相关特征一起标准化
    feature_groups: Dict[str, List[str]] = field(default_factory=dict)

    def __post_init__(self):
        # maybe do something
        self.categorical_feature_names = base_config.DISCRETE_COLUMNS
        self.continuous_feature_names = base_config.CONTINUOUS_COLUMNS
        self.unchanged_feature_names = base_config.INVARIANT_COLUMNS
        # 初始化特征分组字典，如果没有传入
        if self.feature_groups is None:
            self.feature_groups = {}

    def _fit_categorical_feature(self, feature_name: str, input_dataset: pd.DataFrame, column_name: str):
        feature = Feature(name=feature_name, feature_type=FeatureType.Categorical)
        # 注意编码要从1开始，0代表未见的id。获取长度的时候记得要len(feature.category_encode)+1
        feature.category_encode = StringLookup(name=feature_name, offset=1)
        feature.category_encode.fit(input_dataset[column_name].unique().tolist())
        self.features.add(feature)

    def _transform_categorical_feature(self, feature_name: str, input_dataset: pd.DataFrame, column_name: str, output_dataset: pd.DataFrame):
        feature = self.features[feature_name]
        # transform_series = input_dataset[column_name].map(feature.category_encode.code_book).rename(column_name)
        transform_series = feature.category_encode.transform_series(input_dataset[column_name], default=0).rename(column_name)
        new_df = pd.DataFrame(transform_series)
        return pd.concat([output_dataset.reset_index(drop=True), new_df.reset_index(drop=True)], axis=1)

    def _fit_transform_categorical_feature(self, feature_name: str, input_dataset: pd.DataFrame, column_name: str, output_dataset: pd.DataFrame):
        self._fit_categorical_feature(feature_name, input_dataset, column_name)
        return self._transform_categorical_feature(feature_name, input_dataset, column_name, output_dataset=output_dataset)

    def _fit_continuous_feature(self, feature_name: str, input_dataset: pd.DataFrame, column_name: str):
        feature = Feature(name=feature_name, feature_type=FeatureType.Continuous)
        feature.normalization = Normalization(name=feature_name)
        feature.normalization.fit(input_dataset[column_name])
        self.features.add(feature)

    def _transform_continuous_feature(self, feature_name: str, input_dataset: pd.DataFrame, column_name: str, output_dataset: pd.DataFrame):
        feature = self.features[feature_name]
        transform_series = feature.normalization.transform(input_dataset[column_name]).rename(column_name)
        new_df = pd.DataFrame(transform_series)
        return pd.concat([output_dataset.reset_index(drop=True), new_df.reset_index(drop=True)], axis=1)

    def _fit_transform_continuous_feature(self, feature_name: str, input_dataset: pd.DataFrame, column_name: str, output_dataset: pd.DataFrame):
        self._fit_continuous_feature(feature_name, input_dataset, column_name)
        return self._transform_continuous_feature(feature_name, input_dataset, column_name, output_dataset=output_dataset)

    # 添加新方法：对特征组进行拟合
    def _fit_grouped_continuous_features(self, group_name: str, feature_names: List[str], input_dataset: pd.DataFrame):
        """将多个相关特征合并后一起进行标准化拟合"""
        # 创建一个共享的特征对象
        shared_feature = Feature(name=f"group_{group_name}", feature_type=FeatureType.Continuous)
        shared_feature.normalization = Normalization(name=f"group_{group_name}")
        
        # 拼接所有相关特征的数据
        combined_data = pd.Series()
        for feature_name in feature_names:
            combined_data = pd.concat([combined_data, input_dataset[feature_name]])
        
        # 用拼接后的数据拟合标准化器
        shared_feature.normalization.fit(combined_data)
        
        # 为组内的每个特征添加指向同一个标准化器的引用
        for feature_name in feature_names:
            feature = Feature(name=feature_name, feature_type=FeatureType.Continuous)
            # 所有特征共享同一个标准化器
            feature.normalization = shared_feature.normalization
            self.features.add(feature)

    # 添加新方法：转换特征组
    def _transform_grouped_continuous_feature(self, feature_name: str, input_dataset: pd.DataFrame, column_name: str, output_dataset: pd.DataFrame):
        """使用共享标准化参数转换特征"""
        # 与普通连续特征转换相同，因为我们已经共享了标准化器
        return self._transform_continuous_feature(feature_name, input_dataset, column_name, output_dataset)

    def fit_transform(self, input_dataset: pd.DataFrame):
        output_dataframe = pd.DataFrame()
        
        # 处理分类特征
        for name in self.categorical_feature_names:
            output_dataframe = self._fit_transform_categorical_feature(
                feature_name=name, input_dataset=input_dataset, column_name=name, output_dataset=output_dataframe)
        
        # 处理连续特征：先处理分组特征
        processed_continuous_features = set()
        for group_name, feature_names in self.feature_groups.items():
            # 拟合分组特征
            self._fit_grouped_continuous_features(group_name, feature_names, input_dataset)
            # 转换每个特征
            for name in feature_names:
                if name in self.continuous_feature_names:  # 确保只处理连续特征
                    output_dataframe = self._transform_continuous_feature(
                        feature_name=name, input_dataset=input_dataset, column_name=name, output_dataset=output_dataframe)
                    processed_continuous_features.add(name)
        
        # 处理剩余的未分组连续特征
        for name in self.continuous_feature_names:
            if name not in processed_continuous_features:
                output_dataframe = self._fit_transform_continuous_feature(
                    feature_name=name, input_dataset=input_dataset, column_name=name, output_dataset=output_dataframe)
        
        # 处理不变特征
        for name in self.unchanged_feature_names:
            output_dataframe[name] = input_dataset[name]

        # 选择最终输出的特征
        output_dataframe = output_dataframe[self.transform_feature_names]

        return output_dataframe

    def transform(self, input_dataset: pd.DataFrame):
        output_dataframe = pd.DataFrame()
        
        # 处理分类特征
        for name in self.categorical_feature_names:
            output_dataframe = self._transform_categorical_feature(
                feature_name=name, input_dataset=input_dataset, column_name=name, output_dataset=output_dataframe)
        
        # 处理连续特征：包括分组和非分组特征
        for name in self.continuous_feature_names:
            output_dataframe = self._transform_continuous_feature(
                feature_name=name, input_dataset=input_dataset, column_name=name, output_dataset=output_dataframe)
        
        # 处理不变特征
        for name in self.unchanged_feature_names:
            output_dataframe[name] = input_dataset[name]

        # 选择最终输出的特征
        output_dataframe = output_dataframe[self.transform_feature_names]

        return output_dataframe
@dataclass_json
@dataclass
class RiskAlertTransformPipeline(BaseTransform):
    features: Features = None
    trans: RiskAlertTransform = None
    transform_feature_names: List[str] = None
    # 新增：特征分组配置，用于将相关特征一起标准化
    feature_groups: Dict[str, List[str]] = field(default_factory=dict)

    def __post_init__(self):
        if self.trans is None:
            self.trans = RiskAlertTransform(features=Features(), transform_feature_names=self.transform_feature_names, feature_groups=self.feature_groups)

    def fit_transform(self, input_dataset: pd.DataFrame):
        X1:pd.DataFrame = self.trans.fit_transform(input_dataset)
        self.features = self.trans.features # 获得transform
        return X1

    def transform(self, input_dataset: pd.DataFrame):
        X1:pd.DataFrame = self.trans.transform(input_dataset)
        return X1
