#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright (c) 2014 Hal.com, Inc. All Rights Reserved
#
"""
模块用途描述

Authors: zhangzhenhu(acmtiger@163.com)
Date:    2018/8/20 17:20
"""
import xgboost as xgb
import argparse
import numpy as np
from .model import Model
import pandas as pd
import sys
from scipy.stats import binom
from scipy.stats import chi2_contingency
from concurrent import futures
from multiprocessing import Pool as MP
from itertools import repeat


def chi2_test(data: pd.DataFrame, alpha=0.2):
    """
    卡方检验，用于判断独立性
    :param data:
    :param alpha:
    :return: True 代表非独立
    """
    if data.iloc[0, 0] == 0 or data.iloc[1, 1] == 0:
        return False
    g, p, dof, expctd = chi2_contingency(data)
    # P 越大说明独立可能性越大
    if p <= alpha:
        return True
    return False


def conditional_test(data: pd.DataFrame, p=0.85, alpha=0.2):
    """
    条件概率检验
    :param data:
    :param p:
    :param alpha:
    :return:
    """
    n = data.iloc[1, 0] + data.iloc[1, 1]
    pe = 0
    for k in range(int(data.iloc[1, 0]) + 1):
        pe += binom.pmf(k=k, n=n, p=(1 - p))

    if pe > alpha:
        return False
    n = data.iloc[0, 0] + data.iloc[1, 0]
    pe = 0
    for k in range(int(data.iloc[1, 0]) + 1):
        pe += binom.pmf(k=k, n=n, p=(1 - p))

    if pe > alpha:
        return False

    return True


def initializer(args):
    global df_matrix
    df_matrix = args


def process(item):
    global df_matrix
    i, j = item
    # 题目i和题目j进行配对
    # 只有一个人，同时作答题目i和题目j才可以，所以去掉作答记录空的记录
    df_m = df_matrix.iloc[:, [i, j]].dropna().astype(int)
    # 如果同时作答 题目i和题目j 的人数不足20 就跳过，不算
    if len(df_m) < 20:
        return
        # 生成 矩阵
    stat_table = np.zeros((2, 2))
    stat_table[0, 0] = ((df_m.iloc[:, 0] == 0) & (df_m.iloc[:, 1] == 0)).sum()
    stat_table[1, 1] = ((df_m.iloc[:, 0] == 1) & (df_m.iloc[:, 1] == 1)).sum()
    # 如果不存在两题都答错（答对）就跳过不算
    if stat_table[0, 0] == 0 or stat_table[1, 1] == 0:
        return

    stat_table[1, 0] = ((df_m.iloc[:, 0] == 1) & (df_m.iloc[:, 1] == 0)).sum()
    stat_table[0, 1] = ((df_m.iloc[:, 0] == 0) & (df_m.iloc[:, 1] == 1)).sum()

    a, b = df_m.columns
    d = pd.DataFrame(stat_table, dtype=np.int)
    d.columns.name = b  # 列是题目B
    d.index.name = a  # 行是题目A
    # 独立性检验
    if not chi2_test(d):
        return None
    # 条件概率检验
    if conditional_test(d):
        return d
    # A与B互换，再来一次
    d = d.transpose()
    if conditional_test(d):
        return d
    return None


class POKS(Model):
    """
        Partial order knowledge structures 偏序知识结构
    """
    name = "poks-implication"
    # 模型参数
    param = {'chi2_alpha': 0.2, "conditional_p": 0.85, 'conditional_alpha': 0.2}
    description = "关系模型-包含关系"
    # P(B|A),行是A,列是B
    model_b_a = None
    # P(A|-B),行是A,列是B
    model_a_nb = None

    n_pairs = 0
    user_history = None

    @staticmethod
    def co_matrix(df_response):
        """
        生成两两题目之间的 contingency 矩阵
        :param df_response:
        :return:
        """
        # 统计每个题目的作答次数
        df_item_count = df_response.groupby('item_id').agg({'answer': ['count', 'mean']})
        df_item_count.columns = ['count', 'acc']
        # 去掉作答次数不足20次的题目
        df_item_valid = df_item_count[df_item_count['count'] >= 20]
        df_response = df_response[df_response['item_id'].isin(df_item_valid.index)]
        # 生成 学生-题目 作答矩阵
        df_matrix = df_response.drop_duplicates(['user_id', 'item_id']).pivot(index='user_id', columns='item_id',
                                                                              values='answer')
        r_count, c_count = df_matrix.shape

        hehe = [(i, j) for i in range(c_count) for j in range(i + 1, c_count)]

        # 双循环遍历题目，两两进行配对

        # results = []
        with MP(initializer=initializer, initargs=(df_matrix,)) as pool:
            results = [result for result in pool.map(process, hehe, chunksize=10000) if result is not None]
        return results

    @staticmethod
    def estimate(data: np.ndarray):
        a = data.sum(axis=1)
        p_xa = (a + 1) / (a.sum() + 2)
        # 计算 O(A) = P(A) / P(-A)
        o_a = p_xa[1] / p_xa[0]

        b = data.sum(axis=0)
        p_xb = (b + 1) / (b.sum() + 2)
        # 计算 O(B) = P(B) / P(-B)
        o_b = p_xb[1] / p_xb[0]

        # 计算P(B|A) 和 P(-B|A)
        p_xb_a = (data[1, :] + 1) / (data[1, :].sum() + 2)
        # 计算 O(B|A) = P(B|A) / P(-B|A)
        o_b_a = p_xb_a[1] / p_xb_a[0]

        # 计算P(A|-B) 和 P(-A|-B)
        p_xa_b = (data[:, 0] + 1) / (data[:, 0].sum() + 2)
        # 计算 O(A|-B) = P(A|-B) / P(-A/-B)
        o_a_nb = p_xa_b[1] / p_xa_b[0]

        w_ab = o_b_a / o_b
        w_nbna = o_a_nb / o_a

        return (w_ab, w_nbna)

    @staticmethod
    def get_p(data: np.ndarray):
        # 计算P(B|A) 和 P(-B|A)
        p_xb_a = (data[1, :] + 1) / (data[1, :].sum() + 2)

        # 计算P(A|-B) 和 P(-A|-B)
        p_xa_nb = (data[:, 0] + 1) / (data[:, 0].sum() + 2)

        # P(B|A),P(A|-B)
        return p_xb_a[1], p_xa_nb[1]

    def fit(self) -> Model:

        self.user_history = self.feature.response_history[['user_id', 'item_id', 'answer']].set_index('user_id')
        df_response = self.feature.response_history
        # 统计每个题目的作答次数
        df_item_count = df_response.groupby('item_id').agg({'answer': ['count', 'mean']})
        df_item_count.columns = ['count', 'acc']
        # 去掉作答次数不足20次的题目
        df_item_valid = df_item_count[df_item_count['count'] >= 20]
        df_response = df_response[df_response['item_id'].isin(df_item_valid.index)]

        # 生成 学生-题目 作答矩阵
        df_matrix = df_response.drop_duplicates(['user_id', 'item_id']).pivot(index='user_id', columns='item_id',
                                                                              values='answer')
        r_count, c_count = df_matrix.shape

        # 双循环遍历题目，两两进行配对
        hehe = [(i, j) for i in range(c_count) for j in range(i + 1, c_count)]
        with MP(initializer=initializer, initargs=(df_matrix,)) as pool:
            results = [result for result in pool.map(process, hehe, chunksize=10000) if result is not None]
        # return results

        # 关系对的数量
        self.n_pairs = len(results)
        self.description += ",挖掘出%d关系对" % self.n_pairs

        items_a = list(set([data.index.name for data in results]))
        items_b = list(set([data.columns.name for data in results]))
        n_rows = len(items_a)
        n_columns = len(items_b)

        x = np.ndarray(shape=(n_rows, n_columns))
        x[:, :] = np.nan
        # P(B|A),行是A,列是B
        self.model_b_a = pd.DataFrame(x, index=items_a, columns=items_b)

        # P(A|-B),行是A,列是B
        self.model_a_nb = self.model_b_a.copy()

        for data in results:
            p_b_a, p_a_nb = self.get_p(data.values)
            a = data.index.name
            b = data.columns.name
            self.model_b_a.loc[a, b] = p_b_a
            self.model_a_nb.loc[a, b] = p_a_nb

        # 转成系数矩阵
        self.model_b_a = self.model_b_a.to_sparse()
        self.model_a_nb = self.model_a_nb.to_sparse()

        self.train_x = self.select_features(self.feature.features_train)
        # self.feature.features_train[['user_id', 'item_id', 'sf_no']]
        self.train_y = self.feature.label_train.values
        self.feature_names = self.train_x.columns

        # 评估训练集上的效果
        self.train_y_pred = self.predict(self.train_x)
        self.train_ev = self.evaluation.evaluate(y_true=self.train_y, y_pred=self.train_y_pred, threshold=0.5)

        return self

    # def get_user_history(self, user_id):
    #    self.feature.response_history[]

    def predict(self, x: pd.DataFrame = None) -> np.ndarray:
        if x is None:
            x = self.feature.features_test
        # x = self.select_features(x)
        y_pred = []
        for _, row in x.iterrows():
            user_id = row['user_id']
            try:
                # 确保df_uh是DataFrame
                df_uh = self.user_history.loc[[user_id], :]
            except KeyError:
                y_pred.append(np.NAN)
                continue
            if df_uh.empty:
                y_pred.append(np.NAN)
                continue
            item_id = row['item_id']
            # 确保df_uh是DataFrame
            correct_items = df_uh.loc[df_uh['answer'] == 1, 'item_id']
            wrong_items = df_uh.loc[df_uh['answer'] == 0, 'item_id']
            correct_items = list(set(correct_items).intersection(self.model_b_a.index))
            wrong_items = list(set(wrong_items).intersection(self.model_a_nb.index))

            if len(correct_items) == 0:
                y1 = pd.Series([])
            else:
                try:
                    y1 = self.model_b_a.loc[correct_items, item_id]
                except KeyError:
                    y1 = pd.Series([])

            if len(wrong_items) == 0:
                y2 = pd.Series([])
            else:
                try:

                    y2 = self.model_a_nb.loc[wrong_items, item_id]
                except KeyError:
                    y2 = pd.Series([])
            # 暂时先采用最大值
            # 概率传播算法后续再加入

            y_pred.append(max(y1.max(), y2.max()))

        return np.array(y_pred)

    def select_features(self, df_x: pd.DataFrame, feature_list=None) -> pd.DataFrame:
        return df_x[['user_id', 'item_id', 'sf_no']].copy()




def init_option():
    """
    初始化命令行参数项
    Returns:
        OptionParser 的parser对象
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input",
                        help=u"输入文件；默认标准输入设备")
    parser.add_argument("-o", "--output", dest="output",
                        help=u"输出文件；默认标准输出设备")
    return parser


def main(options):
    pass


if __name__ == "__main__":

    parser = init_option()
    options = parser.parse_args()

    if options.input:

        options.input = open(options.input)
    else:
        options.input = sys.stdin
    main(options)
