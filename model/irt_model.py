#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright (c) 2014 Hal.com, Inc. All Rights Reserved
#
"""
模块用途描述

Authors: mengshuai(mengshuai@100tal.com)
Date:    2018/8/13 15:04
"""
import argparse
import numpy as np
import pandas as pd
from scipy.special import expit as sigmod
from .model import Model


class Irt(Model):
    name = "irt_model"
    # 模型参数
    description = "irt_model"

    def select_features(self, df_x: pd.DataFrame, feature_list=None):
        return df_x[['sf_theta', 'if_difficulty', 'sf_no']]

    @staticmethod
    def predict_by_theta(df_data):
        """

        Parameters
        ----------
        theta
        items

        Returns
        -------

        """
        # type(row)
        # print(row)
        D = 1.702
        b = df_data['if_difficulty']
        theta = df_data['sf_theta']
        z = D * (theta - b)
        prob = sigmod(z)
        return prob.values

    def fit(self, **kwargs) -> Model:
        feature_list = kwargs.get('feature_list', None)
        if not feature_list:
            self.name = self.name+'(-irt)'
        self.train_x = self.select_features(self.feature.features_train, feature_list)
        self.train_y = self.feature.label_train.values
        self.feature_names = self.train_x.columns
        # 评估训练集上的效果
        self.train_y_pred = self.predict(self.train_x)
        self.train_ev = self.evaluation.evaluate(y_true=self.train_y, y_pred=self.train_y_pred, threshold=0.5)

        return self

    def predict(self, x: pd.DataFrame = None) -> np.ndarray:
        if x is None:
            x = self.feature.features_test

        y_pred = self.predict_by_theta(x)
        return y_pred


if __name__ == "__main__":
    import feature

    ft = feature.Feature()
    ft.fit()
    # ft.select()

    model = Irt(ft)
    model.predict()
    model.evaluate()
