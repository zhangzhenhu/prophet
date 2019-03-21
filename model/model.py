#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright (c) 2014 Hal.com, Inc. All Rights Reserved
#
"""
模块用途描述

Authors: zhangzhenhu(acmtiger@163.com)
Date:    2018/8/13 15:05
"""
from feature import Feature
from evaluation import Evaluation
import pandas as pd
import numpy as np
import random


class Model(object):
    # 模型名称
    name = "abstract"  # type: str
    # 模型参数K
    param = {}
    # 简单描述
    description = "模型的简单介绍"  # type: str
    # 使用的特征列表
    feature_names = []
    train_x = None  # type: pd.Dataframe
    train_y = None  # type: np.ndarray
    train_y_pred = None  # type: np.ndarray
    test_x = None  # type: pd.Dataframe
    test_y = None  # type: np.ndarray
    test_y_pred = None  # type: np.ndarray
    # 评估结果
    train_ev = None  # type: dict
    test_ev = None  # type: dict

    def __init__(self, feature: Feature, evaluation: Evaluation = None, param=None):
        self.feature = feature
        self.model = None
        # self.train_x = None
        # self.train_y = None
        # self.test_x = None
        # self.test_y = None
        self.y_pred = None
        self.y_true = None
        if evaluation is None:
            self.evaluation = Evaluation(model=self)
        else:
            self.evaluation = evaluation
        if isinstance(param, dict):
            self.param.update(param)

    def select_features(self, df_x: pd.DataFrame, feature_list=None) -> pd.DataFrame:
        if feature_list is None:
            return df_x.filter(regex='^uf_|if_|sf_|kf_', axis=1).copy()
        else:
            return df_x[feature_list]

    def tf_sample(self, df_x: pd.DataFrame, df_y: pd.DataFrame):
        """
        调整正负样本比例
        :param df_x:
        :param df_y:
        :return:
        """
        se_1 = [i for i in range(len(df_y)) if int(df_y[i]) == 1]
        ratio = (1 - df_y.mean()) / df_y.mean()
        len_del = len(se_1) * (1 - ratio) * 0
        random.shuffle(se_1)
        se_1_sub = se_1[:int(len_del)]
        df_x = df_x[~df_x.index.isin(se_1_sub)]
        df_y = [df_y[i] for i in range(len(df_y)) if i not in se_1_sub]
        return df_x, df_y

    def fit(self):
        raise NotImplemented

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        raise NotImplemented

    def save(self):
        raise NotImplemented

    def load(self):
        raise NotImplemented

    def evaluate(self, threshold: float = 0.5):
        return self.evaluation.evaluate(y_true=self.y_true, y_pred=self.y_pred, threshold=threshold)

    def test(self, **kwargs):
        """
        评估测试集上的效果
        :return:
        """
        feature_list = kwargs.get('feature_list', None)
        self.test_x = self.select_features(self.feature.features_test, feature_list)
        self.test_y = self.feature.label_test.values
        self.test_y_pred = self.predict(self.test_x)
        self.test_ev = self.evaluation.evaluate(y_true=self.test_y, y_pred=self.test_y_pred, threshold=0.5)
