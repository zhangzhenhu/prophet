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
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from .model import Model, Feature, Evaluation
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline, FeatureUnion

class SVC_Model(Model):

    name = "SVC"
    # 模型参数
    param = {}
    description = "支持向量机分类"

    def __init__(self, feature: Feature, evaluation: Evaluation = None, kernel='rbf'):
        super(SVC_Model, self).__init__(feature=feature, evaluation=evaluation)
        self.kernel = kernel
        self.description = kernel + self.description

    def select_features(self, df_x: pd.DataFrame, feature_list=None):
        df_new = super(SVC_Model, self).select_features(df_x, feature_list)
        return df_new

    def fit(self, **kwargs) -> Model:
        feature_list = kwargs.get('feature_list', None)
        if not feature_list:
            self.name = self.name+'(-irt)'
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
        combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])
        self.pca = combined_features.fit(self.train_x, self.train_y)
        self.pca = PCA(n_components=n_components).fit(self.train_x)

        self.model = SVC(kernel=self.kernel)
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
        self.model = SVC(kernel=self.kernel)
        self.model.fit(self.train_x, self.train_y)
        return self.model

    def predict(self, x: pd.DataFrame = None) -> np.ndarray:
        if x is None:
            x = self.feature.features_test
        # x = self.select_features(x)
        x = self.scalar_.transform(x)
        x = self.pca.transform(x)

        y_prob = self.model.predict(x)
        '''
        y_prob = y_prob.tolist()
        y_prob = [item[1] for item in y_prob]
        y_prob = np.array(y_prob)
        '''
        return y_prob

class SVR_Model(Model):
    name = "SVR"
    # 模型参数
    param = {}
    description = "支持向量机回归"

    def __init__(self, feature: Feature, evaluation: Evaluation = None, kernel='rbf'):
        super(SVR_Model, self).__init__(feature=feature, evaluation=evaluation)
        self.kernel = kernel
        self.description = kernel + self.description

    def select_features(self, df_x: pd.DataFrame):
        df_new = super(SVR_Model, self).select_features(df_x)
        return df_new

    def fit(self, method_name=None, k_single=0, k_pca=None) -> Model:

        self.train_x = self.select_features(self.feature.features_train)
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
        combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])
        self.pca = combined_features.fit(self.train_x, self.train_y)
        self.pca = PCA(n_components=n_components).fit(self.train_x)

        self.model = SVR(kernel=self.kernel)
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
        self.model = SVR(kernel=self.kernel)
        self.model.fit(self.train_x, self.train_y)
        return self.model

    def predict(self, x: pd.DataFrame = None) -> np.ndarray:
        if x is None:
            x = self.feature.features_test
        x = self.select_features(x)
        x = self.scalar_.transform(x)
        x = self.pca.transform(x)

        y_prob = self.model.predict(x)
        '''
        y_prob = y_prob.tolist()
        y_prob = [item[1] for item in y_prob]
        y_prob = np.array(y_prob)
        '''
        return y_prob

if __name__ == "__main__":
    import feature

    ft = feature.Feature()
    ft.fit()
    model = SVR_Model(ft)
    model.fit()
    model.predict()
    model.evaluate()
