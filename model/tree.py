#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright (c) 2014 Hal.com, Inc. All Rights Reserved
#
"""
模块用途描述

Authors: zhangzhenhu(acmtiger@163.com)
Date:    2018/8/13 15:04
"""
import xgboost as xgb
import argparse
import numpy as np
from .model import Model
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from xgboost.sklearn import XGBClassifier, XGBRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import os


class Xgboost(Model):
    name = "xgboost"
    # 模型参数
    # param = {'max_depth': 6, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
    param = {'max_depth': 4, 'min_child_weight': 4, 'booster': 'gbtree', 'n_estimators': 500, 'learning_rate': 0.02,
             'gamma': 0.2, 'silent': 1, 'objective': 'binary:logistic', 'subsample': 0.8, 'colsample_bytree': 0.8}
    # min_child_weight:, max_depth:, gamma:,
    description = "单纯的xgboost模型"

    def select_features(self, df_x: pd.DataFrame, feature_list=None):
        df_new = super(Xgboost, self).select_features(df_x, feature_list)
        # 填充缺失值为列均值
        for column in df_new.columns:
            mean_ = df_new[column].mean()
            if np.isnan(mean_):
                mean_ = 0
            df_new = df_new.fillna({column: mean_})

        return df_new

    def fit(self, **kwargs) -> Model:
        feature_list = kwargs.get('feature_list', None)
        if not feature_list:
            self.name = self.name+'(-irt)'
        self.train_x = self.select_features(self.feature.features_train, feature_list)
        self.train_y = self.feature.label_train.values
        self.feature_names = self.train_x.columns

        # dtrain = xgb.DMatrix(self.train_x, label=self.train_y)

        num_round = 10
        self.model = XGBClassifier(**self.param)
        # params = {'max_depth': range(2, 6, 1)}
        # params = {'min_child_weight': range(1, 6, 1)}
        # params = {'gamma': [0, 0.1, 0.2, 0.3]}
        # self.adjust_params(params)
        self.model.fit(self.train_x, self.train_y, eval_metric='auc')

        # 评估训练集上的效果
        self.train_y_pred = self.predict(self.train_x)
        self.train_y = np.array(self.train_y)
        self.train_y_pred = np.array(self.train_y_pred)
        self.train_ev = self.evaluation.evaluate(y_true=self.train_y, y_pred=self.train_y_pred, threshold=0.5)

        return self

    def adjust_params(self, params=None, CV=10, scoring='accuracy', n_iter=10):

        self.cvmodel = GridSearchCV(self.model, params, cv=CV, scoring=scoring, n_jobs=os.cpu_count())
        self.cvmodel.fit(self.train_x, self.train_y)
        self.best_params = self.cvmodel.best_params_
        self.best_score_ = self.cvmodel.best_score_
        print('%s在%s参数下，训练集准确率最高为%s:' % (self.name, self.best_params, self.best_score_))
        # 赋值模型以最优的参数
        # self.model = XGBClassifier(**self.param, max_depth=self.best_params['max_depth'])
        # self.model = XGBClassifier(**self.param, min_child_weight=self.best_params['min_child_weight'])
        self.model = XGBClassifier(**self.param, gamma=self.best_params['gamma'])
        return self.model

    def predict(self, x: pd.DataFrame = None) -> np.ndarray:
        if x is None:
            x = self.feature.features_test
        # x = self.select_features(x)

        y_prob = self.model.predict_proba(x)
        y_prob = y_prob.tolist()
        y_prob = [item[1] for item in y_prob]
        y_prob = np.array(y_prob)

        # self.y_pred = y_prob

        return y_prob


class GradientBoostingTree(Model):
    name = "GBDT"
    # 模型参数
    param = {'n_estimators': 500, 'learning_rate': 0.01, 'max_depth': 5, 'min_samples_split': 140,
             'min_samples_leaf': 70, 'max_features': 23, 'subsample': 0.8}

    description = "GBDT"

    def select_features(self, df_x: pd.DataFrame, feature_list=None):
        df_new = super(GradientBoostingTree, self).select_features(df_x, feature_list)
        # 填充缺失值为列均值
        for column in df_new.columns:
            mean_ = df_new[column].mean()
            if np.isnan(mean_):
                mean_ = 0
            df_new = df_new.fillna({column: mean_})

        return df_new

    def fit(self, **kwargs) -> Model:
        feature_list = kwargs.get('feature_list', None)
        if not feature_list:
            self.name = self.name+'(-irt)'
        self.train_x = self.select_features(self.feature.features_train, feature_list)
        self.train_y = self.feature.label_train.values
        self.feature_names = self.train_x.columns

        self.train_x, self.train_y = self.tf_sample(self.train_x, self.train_y)

        self.model = GradientBoostingClassifier(**self.param)
        # params = {'max_depth': range(3, 7, 1), 'min_samples_split': range(100, 200, 20)}
        # params = {'n_estimators': range(100, 1000, 100)}
        # 'learning_rate': [float(i/100) for i in range(1, 11, 1)]
        # params = {'min_samples_leaf': range(40, 100, 10)}
        # params = {'max_features': range(5, 44, 2)}
        # params = {'subsample': [0.75, 0.8, 0.85, 0.9, 0.95]}
        # self.adjust_params(params)
        self.model.fit(self.train_x, self.train_y)
        # 评估训练集上的效果
        self.train_y_pred = self.predict(self.train_x)
        self.train_y = np.array(self.train_y)
        self.train_y_pred = np.array(self.train_y_pred)
        self.train_ev = self.evaluation.evaluate(y_true=self.train_y, y_pred=self.train_y_pred, threshold=0.5)

        return self

    def adjust_params(self, params=None, CV=10, scoring='accuracy', n_iter=10):

        self.cvmodel = GridSearchCV(self.model, params, cv=CV, scoring=scoring, n_jobs=os.cpu_count())
        self.cvmodel.fit(self.train_x, self.train_y)
        self.best_params = self.cvmodel.best_params_
        self.best_score_ = self.cvmodel.best_score_
        print('%s在%s参数下，训练集准确率最高为%s:' % (self.name, self.best_params, self.best_score_))
        # 赋值模型以最优的参数
        # self.model = GradientBoostingClassifier(**self.param, max_depth=self.best_params['max_depth'],
        #                                         min_samples_split=self.best_params['min_samples_split'])
        self.model = GradientBoostingClassifier(**self.param, subsample=self.best_params['subsample'])

        # self.model.fit(self.train_x, self.train_y)
        return self.model

    def predict(self, x: pd.DataFrame = None) -> np.ndarray:
        if x is None:
            x = self.feature.features_test
        # x = self.select_features(x)

        # self.test_x = self.feature.features_test
        # self.test_y = self.feature.label_test.values
        # self.y_true = self.test_y
        y_prob = self.model.predict_proba(x)
        y_prob = y_prob.tolist()
        y_prob = [item[1] for item in y_prob]
        y_prob = np.array(y_prob)

        # self.y_pred = y_prob

        return y_prob

class GBDT_LR(Model):
    name = "GBDT_LR"
    # 模型参数
    param = {'n_estimators': 500, 'learning_rate': 0.01, 'max_depth': 5, 'min_samples_split': 140,
             'min_samples_leaf': 70, 'max_features': 23, 'subsample': 0.8}

    description = "GBDT_LR"

    def select_features(self, df_x: pd.DataFrame, feature_list=None):
        df_new = super(GBDT_LR, self).select_features(df_x, feature_list)
        # 填充缺失值为列均值
        for column in df_new.columns:
            mean_ = df_new[column].mean()
            if np.isnan(mean_):
                mean_ = 0
            df_new = df_new.fillna({column: mean_})

        return df_new

    def fit(self, **kwargs) -> Model:
        feature_list = kwargs.get('feature_list', None)
        if not feature_list:
            self.name = self.name+'(-irt)'
        self.train_x = self.select_features(self.feature.features_train, feature_list)
        self.train_y = self.feature.label_train.values
        self.feature_names = self.train_x.columns

        self.train_x, self.train_y = self.tf_sample(self.train_x, self.train_y)

        grd = GradientBoostingClassifier(**self.param)
        grd_enc = OneHotEncoder()
        grd_lm = LogisticRegression(penalty='l2', C=1, solver='lbfgs')
        grd.fit(self.train_x, self.train_y)
        grd_enc.fit(grd.apply(self.train_x)[:, :, 0])
        grd_lm.fit(grd_enc.transform(grd.apply(self.train_x)[:, :, 0]), self.train_y)
        self.grd = grd
        self.grd_enc = grd_enc
        self.model = grd_lm

        # 评估训练集上的效果
        self.train_y_pred = self.predict(self.train_x)
        self.train_y = np.array(self.train_y)
        self.train_y_pred = np.array(self.train_y_pred)
        self.train_ev = self.evaluation.evaluate(y_true=self.train_y, y_pred=self.train_y_pred, threshold=0.5)

        return self

    def predict(self, x: pd.DataFrame = None) -> np.ndarray:
        if x is None:
            x = self.feature.features_test
        # x = self.select_features(x)

        # self.test_x = self.feature.features_test
        # self.test_y = self.feature.label_test.values
        # self.y_true = self.test_y
        y_prob = self.model.predict_proba(self.grd_enc.transform(self.grd.apply(x)[:, :, 0]))
        y_prob = y_prob.tolist()
        y_prob = [item[1] for item in y_prob]
        y_prob = np.array(y_prob)

        # self.y_pred = y_prob

        return y_prob

class RF_LR(Model):
    name = "RF_LR"
    # 模型参数
    param = {'n_estimators': 35, 'n_jobs': -1, 'max_depth': 9, 'min_samples_split': 60,
             'min_samples_leaf': 30, 'max_features': 23}

    description = "RF_LR"

    def select_features(self, df_x: pd.DataFrame, feature_list=None):
        df_new = super(RF_LR, self).select_features(df_x, feature_list)
        # 填充缺失值为列均值
        for column in df_new.columns:
            mean_ = df_new[column].mean()
            if np.isnan(mean_):
                mean_ = 0
            df_new = df_new.fillna({column: mean_})

        return df_new

    def fit(self, **kwargs) -> Model:
        feature_list = kwargs.get('feature_list', None)
        if not feature_list:
            self.name = self.name+'(-irt)'
        self.train_x = self.select_features(self.feature.features_train, feature_list)
        self.train_y = self.feature.label_train.values
        self.feature_names = self.train_x.columns

        self.train_x, self.train_y = self.tf_sample(self.train_x, self.train_y)

        rf = RandomForestClassifier(**self.param)
        rf_enc = OneHotEncoder()
        rf_lm = LogisticRegression(penalty='l2', C=1, solver='lbfgs')
        rf.fit(self.train_x, self.train_y)
        rf_enc.fit(rf.apply(self.train_x))
        rf_lm.fit(rf_enc.transform(rf.apply(self.train_x)), self.train_y)
        self.rf = rf
        self.rf_enc = rf_enc
        self.model = rf_lm

        # 评估训练集上的效果
        self.train_y_pred = self.predict(self.train_x)
        self.train_y = np.array(self.train_y)
        self.train_y_pred = np.array(self.train_y_pred)
        self.train_ev = self.evaluation.evaluate(y_true=self.train_y, y_pred=self.train_y_pred, threshold=0.5)

        return self

    def predict(self, x: pd.DataFrame = None) -> np.ndarray:
        if x is None:
            x = self.feature.features_test
        # x = self.select_features(x)

        # self.test_x = self.feature.features_test
        # self.test_y = self.feature.label_test.values
        # self.y_true = self.test_y
        y_prob = self.model.predict_proba(self.rf_enc.transform(self.rf.apply(x)))
        y_prob = y_prob.tolist()
        y_prob = [item[1] for item in y_prob]
        y_prob = np.array(y_prob)

        # self.y_pred = y_prob

        return y_prob

class RandomForest(Model):
    name = "random_forest"
    # 模型参数
    param = {'n_estimators': 35, 'n_jobs': -1, 'max_depth': 9, 'min_samples_split': 60,
             'min_samples_leaf': 30, 'max_features': 23}
    # 'max_depth': None, 'min_samples_split':,  'min_samples_leaf': 95
    description = "随机森林"

    def select_features(self, df_x: pd.DataFrame, feature_list=None):
        df_new = super(RandomForest, self).select_features(df_x, feature_list)
        # 填充缺失值为列均值
        for column in df_new.columns:
            mean_ = df_new[column].mean()
            if np.isnan(mean_):
                mean_ = 0
            df_new = df_new.fillna({column: mean_})

        return df_new

    def fit(self, **kwargs) -> Model:
        feature_list = kwargs.get('feature_list', None)
        if not feature_list:
            self.name = self.name+'(-irt)'
        self.train_x = self.select_features(self.feature.features_train, feature_list)
        self.train_y = self.feature.label_train.values
        self.feature_names = self.train_x.columns

        self.model = RandomForestClassifier(**self.param)
        # params = {'n_estimators': range(20, 40, 5)}
        # params = {'min_samples_leaf': range(10, 60, 10)}
        # params = {'max_features': range(5, 44, 2)}
        # params = {'max_depth': range(4, 12, 1), 'min_samples_split': range(40, 260, 20)}
        # self.adjust_params(params)
        self.model.fit(self.train_x, self.train_y)
        # 评估训练集上的效果
        self.train_y_pred = self.predict(self.train_x)
        self.train_ev = self.evaluation.evaluate(y_true=self.train_y, y_pred=self.train_y_pred, threshold=0.5)

        return self

    def adjust_params(self, params=None, CV=10, scoring='accuracy', n_iter=10):

        self.cvmodel = GridSearchCV(self.model, params, cv=CV, scoring=scoring, n_jobs=-1)
        self.cvmodel.fit(self.train_x, self.train_y)
        self.best_params = self.cvmodel.best_params_
        self.best_score_ = self.cvmodel.best_score_
        print('%s在%s参数下，训练集准确率最高为%s:' % (self.name, self.best_params, self.best_score_))

        # 赋值模型以最优的参数
        # self.model = RandomForestClassifier(**self.param, n_estimators=self.best_params['n_estimators'])
        # self.model = RandomForestClassifier(**self.param, max_depth=self.best_params['max_depth'],
        #                                     min_samples_split=self.best_params['min_samples_split'])
        self.model = RandomForestClassifier(**self.param, max_features=self.best_params['max_features'])
        self.model.fit(self.train_x, self.train_y)
        return self.model

    def predict(self, x: pd.DataFrame = None) -> np.ndarray:
        if x is None:
            x = self.feature.features_test
        # x = self.select_features(x)

        # self.test_x = self.feature.features_test
        # self.test_y = self.feature.label_test.values
        # self.y_true = self.test_y
        y_prob = self.model.predict_proba(x)
        y_prob = y_prob.tolist()
        y_prob = [item[1] for item in y_prob]
        y_prob = np.array(y_prob)

        # self.y_pred = y_prob

        return y_prob


class DecisionTree(Model):
    name = "DecisionTree"
    # 模型参数
    param = {'max_depth': 9, 'min_samples_split': 220, 'min_samples_leaf': 18, 'criterion': 'gini',
             'max_features': None}
    #
    description = "决策树"

    def select_features(self, df_x: pd.DataFrame, feature_list=None):
        df_new = super(DecisionTree, self).select_features(df_x, feature_list)
        # 填充缺失值为列均值
        for column in df_new.columns:
            mean_ = df_new[column].mean()
            if np.isnan(mean_):
                mean_ = 0
            df_new = df_new.fillna({column: mean_})
        return df_new

    def fit(self, **kwargs) -> Model:
        feature_list = kwargs.get('feature_list', None)
        if not feature_list:
            self.name = self.name+'(-irt)'
        self.train_x = self.select_features(self.feature.features_train, feature_list)
        self.train_y = self.feature.label_train.values
        self.feature_names = self.train_x.columns
        self.model = DecisionTreeClassifier(**self.param)
        # params = {'min_samples_leaf': range(14, 26, 2)}
        # params = {'max_features': range(5, 44, 2)}
        # params = {'max_depth': range(4, 12, 1), 'min_samples_split': range(40, 260, 20)}
        # params = {'criterion': ['gini', 'entropy']}
        # self.adjust_params(params)
        self.model.fit(self.train_x, self.train_y)
        # 评估训练集上的效果
        self.train_y_pred = self.predict(self.train_x)
        self.train_ev = self.evaluation.evaluate(y_true=self.train_y, y_pred=self.train_y_pred, threshold=0.5)

        return self

    def adjust_params(self, params=None, CV=10, scoring='accuracy', n_iter=10):

        self.cvmodel = GridSearchCV(self.model, params, cv=CV, scoring=scoring, n_jobs=os.cpu_count())
        self.cvmodel.fit(self.train_x, self.train_y)
        self.best_params = self.cvmodel.best_params_
        self.best_score_ = self.cvmodel.best_score_
        print('%s在%s参数下，训练集准确率最高为%s:' % (self.name, self.best_params, self.best_score_))

        # 赋值模型以最优的参数
        # self.model = DecisionTreeClassifier(**self.param, max_depth=self.best_params['max_depth'],
        #                                     min_samples_split=self.best_params['min_samples_split'])
        # self.model = DecisionTreeClassifier(**self.param, criterion=self.best_params['criterion'])
        self.model = DecisionTreeClassifier(**self.param, min_samples_leaf=self.best_params['min_samples_leaf'])
        # self.model.fit(self.train_x, self.train_y)
        return self.cvmodel

    def predict(self, x: pd.DataFrame = None) -> np.ndarray:
        if x is None:
            x = self.feature.features_test
        # self.x = self.select_features(x)

        # self.test_x = self.feature.features_test
        # self.test_y = self.feature.label_test.values
        # self.y_true = self.test_y
        y_prob = self.model.predict_proba(x)
        y_prob = y_prob.tolist()
        y_prob = [item[1] for item in y_prob]
        y_prob = np.array(y_prob)

        # self.y_pred = y_prob

        return y_prob


if __name__ == "__main__":
    import feature

    ft = feature.Feature()
    ft.fit()
    # ft.select()

    model = RandomForest(ft)
    model.fit()
    model.predict()
    model.evaluate()
