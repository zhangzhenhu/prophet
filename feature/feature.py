#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright (c) 2014 Hal.com, Inc. All Rights Reserved
#
"""
模块用途描述

Authors: zhangzhenhu(acmtiger@163.com)
Date:    2018/8/10 16:13
"""
import pandas as pd
from . import impala_sql
import os
import numpy as np
from . import irt
from joblib import Parallel, delayed
import joblib
import time
from IPython.display import display
import numpy as np
from multiprocessing import cpu_count, Pool
from datetime import datetime
from scipy.special import expit as sigmod
import math

cores = cpu_count()  # Number of CPU cores on your system
partitions = cores  # Define as many partitions as you want


def parallelize(data, func):
    data_split = np.array_split(data, partitions)
    pool = Pool(cores)
    data = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return data


# data = parallelize(data, work);


# _parallel = Parallel(n_jobs=os.cpu_count(), backend='multiprocessing')

def applyParallel(dfGrouped, func):
    retLst = Parallel(n_jobs=os.cpu_count())(delayed(func)(group, name) for name, group in dfGrouped)
    return pd.concat(retLst)


def irt_theta_sf(df_ku, name):
    """
    序列特征里加入IRT.
    (User,Knowledge)答题序列，理论上不会有重复题目
    :param name:
    :param df_ku:
    :return:
    """
    # ##############################
    # 序列化theta值作为特征
    # ##############################
    df_ku = df_ku.sort_values('date_time')
    total_count = len(df_ku)
    df_response = df_ku.copy()
    df_response.rename(columns={'difficulty': 'b'}, inplace=True)
    # df_response.drop_duplicates(['user_id', 'item_id'], inplace=True)
    df_theta = []
    for i in range(1, total_count + 1):
        response = df_response.iloc[:i, ].copy()
        model = irt.UIrt2PL()
        model.fit(response=response)
        model.estimate_theta()
        theta = model.user_vector[['theta']]
        df_theta.append(theta.iloc[0, 0])

    cum_count = list(range(1, total_count + 1))
    theta_series = pd.DataFrame({
        'knowledge_id': [name[0]] * total_count,
        'user_id': [name[1]] * total_count,
        'item_id': df_ku['item_id'].values,
        'date_time': df_ku['date_time'].values,
        # 'no': cum_count,  # 题号
        'sf_theta': [1.5] + df_theta[:-1],  # 第一题，默认能力值为1.5,默认大家都能答对1难度题目
    })

    D = 1.702
    z = D * (theta_series['sf_theta'].values - df_ku['difficulty'].values)
    theta_series['sf_irt_prob'] = sigmod(z)

    # # 第一题，
    # theta_series.fillna(0)
    return theta_series


def irt_theta_uf(df_u_response, user_id="0"):
    """
    用户画像特征里加入irt
    :return:
    """
    df_response = df_u_response[['item_id', 'difficulty', 'answer']].copy()
    df_response['user_id'] = user_id
    df_response.rename(columns={'difficulty': 'b'}, inplace=True)
    df_response.drop_duplicates(['user_id', 'item_id'], inplace=True)
    model = irt.UIrt2PL()
    model.fit(response=df_response)
    model.estimate_theta()
    theta = model.user_vector.drop(['iloc'], axis=1).set_index('user_id')
    theta.columns = ['uf_%s' % x for x in theta.columns]
    return theta


class Feature:
    _feature_names = pd.Series({'if_acc': "题目画像-正确率",
                                'if_answer_count': "题目画像-作答次数",
                                'if_correct_count': "题目画像-正确次数",
                                'if_difficulty': "题目画像-人工难度",
                                'kf_acc': "知识点画像-正确率",
                                'kf_answer_count': "知识点画像-作答次数",
                                'kf_correct_count': "知识点画像-正确次数",
                                'sf_acc': "序列特征-正确率",
                                'sf_theta': "序列特征-IRT能力值",
                                'sf_irt_prob': "序列特征-IRT预测概率",
                                'sf_answer_K': "序列特征-第k题作答结果",
                                'sf_difficulty_k': "序列特征-第k题难度",

                                # 'uf_acc': "学生画像-正确率",
                                # 'uf_answer_count': "学生画像-作答次数",
                                'uf_theta': "学生画像-IRT能力值",
                                'uf_right_k': '学生画像-k难度题目正确数量',
                                'uf_count_k': '学生画像-k难度题目作答数量',
                                'uf_accuracy_k': '学生画像-k难度题目正确率',
                                })

    def __init__(self):
        # pass
        # self.train_data = None
        # self.test_data = None
        self.start_time = 0
        self.finish_time = 0
        self.cost_time = 0
        # 实体的特征
        self.user_features = None  # type: pd.DataFrame
        self.item_features = None  # type: pd.DataFrame
        self.knowledge_features = None  # type: pd.DataFrame
        # 作答序列特征
        self.series_features = None  # type: pd.DataFrame
        # 作答记录数据
        self.response_raw = None  # type: pd.DataFrame
        self.response_history = None  # type: pd.DataFrame
        self.response_train = None  # type: pd.DataFrame
        self.response_test = None  # type: pd.DataFrame
        # 训练数据的特征矩阵
        self.features_train = None  # type: pd.DataFrame
        self.features_test = None  # type: pd.DataFrame
        self.label_train = None  # type: pd.DataFrame
        self.label_test = None  # type: pd.DataFrame
        # 实体的静态标签数据,(数据库中直接读取到的信息)
        self.item_data = None  # type: pd.DataFrame

    @property
    def feature_names(self):
        pass
        return self._feature_names

    def download_data(self):
        self.response_raw = impala_sql.ai_response()
        self.item_data = impala_sql.aistudy_knowledge_question()
        self.item_data.drop_duplicates('item_id', inplace=True)
        self.item_data.set_index('item_id', inplace=True)

    def _split_by_day(self, data, day1="2018-07-20", day2="2018-08-01"):
        """
        按照某个时间点，将数据分成两部分
        :param data:
        :param day:
        :return:
        """
        d1 = data[data['date_time'] <= day1]
        d2 = data[(data['date_time'] > day1) & (data['date_time'] <= day2)]
        d3 = data[data['date_time'] > day2]
        return d1, d2, d3

    def _user_feature(self):
        """
        生成用户特征. 前缀 uf_
        :return:
        """
        # user_g = self.response_history.groupby('user_id')
        # user_acc = user_g.agg({'answer': ['mean', 'sum', 'count']})
        # user_acc.columns = ['uf_acc', 'uf_correct_count', 'uf_answer_count']
        # self.user_features = user_acc
        # df_theta = applyParallel(user_g, irt_theta_uf)

        self.user_features = self._uf_irt_theta(self.response_history)
        # self.user_features = self.user_features.join(df_theta, how='left')

    def _item_feature(self):
        """
        生成题目特征. 前缀 if_
        :return:
        """
        epsilon = 0.005
        item_acc = self.response_history.groupby('item_id').agg({'answer': ['mean', 'sum', 'count']})
        item_acc.columns = ['if_acc', 'if_correct_count', 'if_answer_count']
        item_acc['if_acc_new'] = (item_acc['if_correct_count']+1)/(item_acc['if_answer_count']+2)
        item_acc.drop(columns=['if_acc'], inplace=True)
        self.item_features = self.item_data[['difficulty']].join(item_acc, how='left')
        # self.item_features = item_acc.join(self.items_data[['difficulty']], how='left')
        self.item_features.rename(columns={'difficulty': 'if_difficulty'}, inplace=True)
        # self.items_data

    def _knowledge_feature(self):
        """
        生成知识点特征. 前缀 kf_
        :return:
        """
        epsilon = 0.005
        df_acc = self.response_history.groupby('knowledge_id').agg({'answer': ['mean', 'sum', 'count']})
        df_acc.columns = ['kf_acc', 'kf_correct_count', 'kf_answer_count']
        df_acc['kf_acc_new'] = (df_acc['kf_correct_count'] + 1) / (df_acc['kf_answer_count'] + 2)
        df_acc.drop(columns=['kf_acc'], inplace=True)
        self.knowledge_features = df_acc

    def fit(self, split_day1="2018-07-20", split_day2="2018-08-01"):
        self.start_time = time.time()
        self.download_data()
        self.response_history, self.response_train, self.response_test = self._split_by_day(data=self.response_raw,
                                                                                            day1=split_day1,
                                                                                            day2=split_day2)

        # self.response_train,self.response_predict
        self._item_feature()
        self._user_feature()
        self._knowledge_feature()

        self._series_features()
        self.update_feature()
        #
        # def select(self, user_feature_selected=['uf_acc', 'uf_correct_count', 'uf_answer_count'],
        #            item_feature_selected=['if_difficulty', 'if_acc', 'if_correct_count', 'if_answer_count'],
        #            knowledge_feature_selected=['kf_acc', 'kf_correct_count', 'kf_answer_count'],
        #            series_feature_selected=['sf_acc']):
        self.finish_time = time.time()
        self.cost_time = self.finish_time - self.start_time
        return self

    def update_feature(self):
        df_train = self.response_train[['user_id', 'item_id', 'knowledge_id']].copy()
        df_test = self.response_test[['user_id', 'item_id', 'knowledge_id']].copy()
        self.label_train = self.response_train['answer'].copy()
        self.label_test = self.response_test['answer'].copy()
        user_feature_selected = [col for col in self.user_features.columns if col.startswith('uf_')]
        item_feature_selected = [col for col in self.item_features.columns if col.startswith('if_')]
        knowledge_feature_selected = [col for col in self.knowledge_features.columns if col.startswith('kf_')]
        series_feature_selected = [col for col in self.series_features.columns if col.startswith('sf_')]
        # user_feature_selected = []
        if user_feature_selected:
            df_train = df_train.join(self.user_features[user_feature_selected], on='user_id', how='left')
            df_test = df_test.join(self.user_features[user_feature_selected], on='user_id', how='left')

        if item_feature_selected:
            df_train = df_train.join(self.item_features[item_feature_selected], on='item_id', how='left')
            df_test = df_test.join(self.item_features[item_feature_selected], on='item_id', how='left')

        # print(1, len(df_train))
        if knowledge_feature_selected:
            df_train = df_train.join(self.knowledge_features[knowledge_feature_selected], on='knowledge_id', how='left')
            df_test = df_test.join(self.knowledge_features[knowledge_feature_selected], on='knowledge_id', how='left')
        # print(df_train.columns)
        # print(2, len(df_train))
        if series_feature_selected:
            # 增加作答序列特征
            on = ['knowledge_id', 'user_id', 'item_id']
            df_train = df_train.merge(self.series_features[on + series_feature_selected], left_on=on, right_on=on,
                                      how='left')
            df_test = df_test.merge(self.series_features[on + series_feature_selected], left_on=on, right_on=on,
                                    how='left')
        # print(3, len(df_train))
        self.features_train = df_train
        self.features_test = df_test

        # select the data satisfy the rule
        self.dataSplitByRule()
        # self.features_train.set_index(['knowledge_id', 'user_id', 'item_id'], inplace=True)
        # self.features_test.set_index(['knowledge_id', 'user_id', 'item_id'], inplace=True)
        # self.features_train.sort_index(axis=1, inplace=True)
        # self.features_test.sort_index(axis=1, inplace=True)
        self.evaluate()

    def dataSplitByRule(self):

        print('原始训练集大小：', self.features_train.shape[0])
        self.features_train = self.features_train.loc[(self.features_train.if_answer_count > 20) & (self.features_train.uf_count_all > 5)]
        print('筛选训练集大小：', self.features_train.shape[0])

        print('原始测试集大小：', self.features_test.shape[0])
        self.features_test = self.features_test.loc[(self.features_test.if_answer_count > 20) & (self.features_test.uf_count_all > 5)]
        print('筛选测试集大小：', self.features_test.shape[0])

        on = ['knowledge_id', 'user_id', 'item_id']
        self.label_train = self.response_train.merge(self.features_train, left_on=on, right_on=on, how='inner')['answer'].copy()
        self.label_test = self.response_test.merge(self.features_test, left_on=on, right_on=on, how='inner')['answer'].copy()

    def fillna(self):

        rule = {'sf_acc': -1,
                'if_acc': -1,
                'if_answer_count': -1,
                'if_correct_count': -1,
                'uf_acc': -1,
                'uf_answer_count': -1,
                'uf_correct_count': -1,
                'kf_acc': -1,
                'kf_answer_count': -1,
                'kf_correct_count': -1,
                }
        self.features_train.fillna(rule, inplace=True)
        self.features_test.fillna(rule, inplace=True)
        self.features_train.fillna(0, inplace=True)
        self.features_test.fillna(0, inplace=True)

    @staticmethod
    def __series_features(df_ku: pd.DataFrame):
        df_ku = df_ku.sort_values('date_time')
        total_count = len(df_ku)
        cum_right = df_ku['answer'].cumsum()
        cum_count = list(range(1, total_count + 1))
        cum_acc = cum_right / cum_count

        # ##############################
        # 前k道题目的答题结果和难度作为特征
        # ##############################
        k = 10
        ds_ta = df_ku['answer'][:k].values.copy()
        ds_td = df_ku['difficulty'][:k].values.copy()
        # 做错的设置为-1
        ds_ta[ds_ta == 0] = -1

        # 前k题目的作答序列
        ds_ta = ds_ta.reshape((1, ds_ta.size)).repeat(total_count, axis=0)
        # 前k题目的难度序列
        ds_td = ds_td.reshape((1, ds_td.size)).repeat(total_count, axis=0)
        # 如果不够k道题，补足，默认值都为0（未作答）
        if total_count < k:
            attach = np.zeros((total_count, k - total_count))
            ds_ta = np.hstack((ds_ta, attach))
            ds_td = np.hstack((ds_td, attach))
        # 时间上属于未来的题目 应该是未作答状态，设置为0
        for i in range(total_count):
            ds_ta[i, i:] = 0
        # ##############################

        # ##############################
        # 作答记录
        # ##############################
        # ds_his=df_ku['item_id'] + "_" + df_ku['answer'].astype(str)
        # ds
        # ##############################

        # 列名带有前缀 sf_（series_features） 的是序列特征
        base = pd.DataFrame({
            'item_id': df_ku['item_id'].values,  # 一定要用values，否则会出现异常结果
            'date_time': df_ku['date_time'].values,
            'sf_no': cum_count,  # 题号
            'sf_acc': [0] + cum_acc.tolist()[:-1],  # 第一题，默认正确率是0
            'sf_acc_new': [0.5] + [abs(acc-0.5) if i<2 else acc for i, acc in enumerate(cum_acc.tolist()[:-1])]
        }, index=np.arange(total_count))
        # 注意，concat(axis=1) 要求 index 必须一致
        return pd.concat([
            base,
            pd.DataFrame(ds_ta, columns=["sf_answer_%d" % i for i in range(1, k + 1)]),
            pd.DataFrame(ds_td, columns=["sf_difficulty_%d" % i for i in range(1, k + 1)]),
        ], axis=1)

    def _series_features(self):
        """
        作答序列特征。前缀 sf_
        当前应试者在当前知识点的时间序列特征。
        :return:
        """

        df_response = self.response_raw

        g = df_response.groupby(['knowledge_id', 'user_id'])
        self.series_features = g.apply(self.__series_features).reset_index()
        df_theta = self._sf_irt_theta(df_response=df_response)
        on_keys = ['knowledge_id', 'user_id', 'item_id', 'date_time']
        self.series_features = self.series_features.merge(df_theta, left_on=on_keys, right_on=on_keys, how='left')

    @staticmethod
    def _uf_irt_theta(df_response):
        today = datetime.today().strftime("%Y-%m-%d")
        file_name = "uf_irt_theta_%s.pkl" % today
        # df_response = self.response_test
        if os.path.exists(file_name):
            df_theta = joblib.load(file_name)
        else:
            g = df_response.groupby(['user_id'])
            df_theta = applyParallel(g, irt_theta_uf)
            # df_theta = g.apply(irt_theta_uf)
            # print(df_theta.columns)
            joblib.dump(df_theta, file_name)
        return df_theta

    @staticmethod
    def _sf_irt_theta(df_response):
        """
        训练irt的theta值，作为特征。
        每个作答样本有一个theta值，当前作答样本的theta值来源于其所属序列（knowledge_id', 'user_id'）前面的题训练而成。
        :param df_response:
        :return:
        """
        today = datetime.today().strftime("%Y-%m-%d")
        file_name = "sf_irt_theta_%s.pkl" % today
        # df_response = self.response_test
        if os.path.exists(file_name):
            df_theta = joblib.load(file_name)
        else:
            g = df_response.groupby(['knowledge_id', 'user_id'])
            df_theta = applyParallel(g, irt_theta_sf)
            joblib.dump(df_theta, file_name)
        return df_theta

    def describe(self) -> pd.DataFrame:

        def hehe(ds_label):
            _vc = ds_label.value_counts().to_dict()
            _total = sum(_vc.values())

            return {"数量总量": _total,
                    '正样本数': _vc[1],
                    '负样本数': _vc[0],
                    '正样本比例': _vc[1] / float(_total),
                    }

        record1 = hehe(self.label_test)
        record1['name'] = "测试数据"
        record2 = hehe(self.label_train)
        record2['name'] = "训练数据"
        record3 = hehe(self.response_history['answer'])
        record3['name'] = "画像数据"
        df_ret = pd.DataFrame([record1, record2, record3])
        df_ret.set_index('name', inplace=True)
        # display(df_ret)
        return df_ret

    def evaluate(self):
        """
        评估一下特征的覆盖情况。比如有多少比例样本缺少用户特征(新用户)
        :return:
        """
        pass

    def save(self, filename="feature.pkl"):
        joblib.dump(self, filename)

    @staticmethod
    def load(filename="feature.pkl"):
        return joblib.load(filename)
