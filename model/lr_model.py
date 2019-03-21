#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright (c) 2014 Hal.com, Inc. All Rights Reserved
#
"""
模块用途描述

Authors: mengshuai(mengshuai@100tal.com)
Date:    2018/8/14 14:39
"""

import pandas as pd
import numpy as np
import random
import argparse
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from .model import Model, Feature, Evaluation
from sklearn.decomposition import PCA, KernelPCA
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline, FeatureUnion

class LR(Model):
    name = "LogisticRegression"
    # 模型参数
    param = {}
    description = "逻辑回归"

    def select_features(self, df_x: pd.DataFrame, feature_list=None):
        df_new = super(LR, self).select_features(df_x, feature_list)
        return df_new

    def fit(self, **kwargs) -> Model:

        feature_list = kwargs.get('feature_list', None)
        if not feature_list:
            self.name = self.name+'(-irt)'
        solver_name = kwargs.get('solver_name', 'lbfgs')
        penalty = kwargs.get('penalty', 'l2')
        k_single = kwargs.get('k_single', 0)
        k_pca = kwargs.get('k_pca', 1)
        self.train_x = self.select_features(self.feature.features_train, feature_list)
        self.train_y = self.feature.label_train.values
        self.feature_names = self.train_x.columns

        self.train_x, self.train_y = self.tf_sample(self.train_x, self.train_y)

        # 数据归一化
        scaler = preprocessing.StandardScaler()
        self.scalar_ = scaler.fit(self.train_x)

        # pca
        selection = SelectKBest(k=k_single)
        n_components = int(len(self.feature_names) * k_pca)
        pca = PCA(n_components=n_components)
        # pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True)
        combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])
        self.pca = combined_features.fit(self.train_x, self.train_y)
        self.pca = PCA(n_components=n_components).fit(self.train_x)


        self.model = LogisticRegression(penalty=penalty, C=1, solver=solver_name)
        fit_data = self.train_x.copy()
        fit_data = self.scalar_.transform(fit_data)
        fit_data = self.pca.transform(fit_data)
        self.model.fit(fit_data, self.train_y)

        # 评估训练集上的效果
        self.train_y_pred = self.predict(self.train_x)
        self.train_y = np.array(self.train_y)
        self.train_y_pred = np.array(self.train_y_pred)
        self.train_ev = self.evaluation.evaluate(y_true=self.train_y, y_pred=self.train_y_pred, threshold=0.5)

        return self

    def adjust_params(self, params=None, CV=10, scoring='accuracy', n_iter=10):

        self.cvmodel = GridSearchCV(self.model, params, cv=CV, scoring=scoring, n_jobs=-1)
        self.cvmodel.fit(self.train_x, self.train_y)
        self.best_params = self.cvmodel.best_params_
        self.best_score_ = self.cvmodel.best_score_
        print('%s在%s参数下，训练集准确率最高为%s:' % (self.model_name, self.best_params, self.best_score_))

        # 赋值模型以最优的参数
        self.model = LogisticRegression(C=1., solver='lbfgs')
        self.model.fit(self.train_x, self.train_y)
        return self.model

    def predict(self, x: pd.DataFrame = None) -> np.ndarray:
        if x is None:
            x = self.feature.features_test
        # x = self.select_features(x)
        x = self.scalar_.transform(x)
        x = self.pca.transform(x)

        y_prob = self.model.predict_proba(x)
        y_prob = y_prob.tolist()
        y_prob = [item[1] for item in y_prob]
        y_prob = np.array(y_prob)

        return y_prob


if __name__ == "__main__":
    import feature

    ft = feature.Feature()
    ft.fit()
    model = LR(ft)
    model.fit()
    model.predict()
    model.evaluate()
