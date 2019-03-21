#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright (c) 2014 Hal.com, Inc. All Rights Reserved
#
"""
模块用途描述

Authors: mengshuai(mengshuai@100tal.com)
Date:    2018/8/15 19:44
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.special import expit as sigmod
from scipy.optimize import minimize
import tempfile
import logging



class UIrt2PL:
    # logger = logging.getLogger()

    def __init__(self, D=1.702, logger=logging.getLogger()):
        self.D = D
        self.k = 1
        self.user_vector = None
        self.item_vector = None
        self.logger = logger
        self.model = None
        self.response_sequence = None

    def fit(self, response: pd.DataFrame, sequential=True):
        """
        :param response_df: 作答数据，必须包含三列 user_id item_id answer
        D=1.702
        """
        assert response is not None
        if 'new_difficulty' in response.columns and 'b' not in response.columns:
            response.rename(columns={'new_difficulty': 'b'}, inplace=True)
        elif 'difficulty' in response.columns and 'b' not in response.columns:
            response.rename(columns={'difficulty': 'b'}, inplace=True)

        if 'discrimination' in response.columns and 'a' not in response.columns:
            response.rename(columns={'discrimination': 'a'}, inplace=True)
        elif 'a' not in response.columns:
            response.loc[:, 'a'] = 1

        if sequential:
            labels = {'user_id', 'item_id', 'answer', 'a', 'b', 'create_time', 'knowledge_id'}.intersection(
                response.columns)
            self.response_sequence = response[list(labels)]

            self.response_matrix = self.response_sequence.pivot(index="user_id", columns="item_id", values='answer')

        else:
            self.response_matrix = response.copy()
            self.response_matrix.index.name = 'user_id'
            # 矩阵形式生成序列数据
            self.response_sequence = pd.melt(self.response_matrix.reset_index(), id_vars=['user_id'],
                                             var_name="item_id",
                                             value_name='answer')
            # 去掉空值
            self.response_sequence.dropna(inplace=True)

        #
        self._init_model()
        labels = set(response.columns).intersection({'a', 'b', 'c'})
        if sequential and labels:
            item_info = response[['item_id'] + list(labels)].drop_duplicates(subset=['item_id'])
            item_info.set_index('item_id', inplace=True)
            self.set_abc(item_info, columns=list(labels))

        ret = self.estimate_theta()
        self.model = self.user_vector
        return ret

    def _init_model(self):
        assert self.response_sequence is not None
        user_ids = list(self.response_matrix.index)
        user_count = len(user_ids)
        item_ids = list(self.response_matrix.columns)
        item_count = len(item_ids)
        self.user_vector = pd.DataFrame({
            'iloc': np.arange(user_count),
            'user_id': user_ids,
            'theta': np.zeros(user_count)},
            index=user_ids)
        self.item_vector = pd.DataFrame(
            {'iloc': np.arange(item_count),
             'item_id': item_ids,
             'a': np.ones(item_count),
             'b': np.zeros(item_count),
             'c': np.zeros(item_count)}, index=item_ids)

        self.response_sequence = self.response_sequence.join(self.user_vector['iloc'].rename('user_iloc'), on='user_id',
                                                             how='left')
        self.response_sequence = self.response_sequence.join(self.item_vector['iloc'].rename('item_iloc'), on='item_id',
                                                             how='left')
        # 统计每个应试者的作答情况
        # user_stat = self.response_sequence.groupby('user_id')['answer'].aggregate(['count', 'sum']).rename(
        #     columns={'sum': 'right'})
        # 注意：难度是浮点数，需要先转换为整型，然后在统计每个难度的分布
        x = self.response_sequence.astype({'b': 'int32'}).groupby(['user_id', 'b']).aggregate(
            {'answer': ['count', 'sum']})
        y = x.unstack()
        y.columns = [(col[1] + "_" + str(int(col[2]))).strip().replace('sum', 'right') for col in y.columns.values]

        for i in range(1, 6):
            i = int(i)
            if 'right_%s' % i in y.columns:
                y.loc[:, 'accuracy_%s' % i] = y['right_%s' % i] / y['count_%s' % i]

        y.loc[:, 'count_all'] = y.filter(regex='^count_', axis=1).sum(axis=1)
        y.loc[:, 'right_all'] = y.filter(regex='^right_', axis=1).sum(axis=1)
        y.loc[:, 'accuracy_all'] = y['right_all'] / y['count_all']

        self.user_vector = self.user_vector.join(y, how='left')
        # self.user_vector.fillna({'count': 0, 'right': 0}, inplace=True)
        # self.user_vector['accuracy'] = self.user_vector['right'] / self.user_vector['count']

        # 统计每个项目的作答情况
        item_stat = self.response_sequence.groupby('item_id')['answer'].aggregate(['count', 'sum']).rename(
            columns={'sum': 'right'})
        self.item_vector = self.item_vector.join(item_stat, how='left')
        self.item_vector.fillna({'count': 0, 'right': 0}, inplace=True)
        self.item_vector['accuracy'] = self.item_vector['right'] / self.item_vector['count']

    def set_theta(self, values):
        """

        Parameters
        ----------
        values

        Returns
        -------

        """
        assert isinstance(values, pd.DataFrame) or isinstance(values,
                                                              np.ndarray), "values的类型必须是pandas.DataFrame或numpy.ndarray"

        if self.user_vector is None:
            assert isinstance(values, pd.DataFrame), "values的类型必须是pandas.DataFrame"
            user_count = len(values)
            user_ids = list(values.index)

            self.user_vector = pd.DataFrame({
                'iloc': np.arange(user_count),
                'user_id': user_ids,
                'theta': values.loc[:, 'theta'].values.flatten(),
            },
                index=user_ids)

        else:
            if isinstance(values, pd.DataFrame):
                # self.user_vector = values
                self.user_vector.loc[values.index, 'theta'] = values.loc[:, 'theta'].values.flatten()

            elif isinstance(values, np.ndarray):
                self.user_vector.loc[:, 'theta'] = values.flatten()

            else:
                raise TypeError("values的类型必须是pandas.DataFrame 或numpy.ndarray")

    def set_abc(self, values, columns=None):
        """
        values 可以是pandas.DataFrame 或者 numpy.ndarray
        当values:pandas.DataFrame,,shape=(n,len(columns))，一行一个item,
        pandas.DataFrame.index是item_id,columns包括a,b,c。

        当values:numpy.ndarray,shape=(n,len(columns)),一行一个item,列对应着columns参数。
        Parameters
        ----------
        values
        columns 要设置的列

        Returns
        -------

        """

        assert isinstance(values, pd.DataFrame) or isinstance(values,
                                                              np.ndarray), "values的类型必须是pandas.DataFrame或numpy.ndarray"
        if columns is None:
            if isinstance(values, pd.DataFrame):
                columns = [x for x in ['a', 'b', 'c'] if x in values.columns]
            else:
                raise ValueError("需要指定columns")

        if self.item_vector is None:
            assert isinstance(values, pd.DataFrame), "values的类型必须是pandas.DataFrame"
            item_count = len(values)
            item_ids = list(values.index)

            self.item_vector = pd.DataFrame({
                'iloc': np.arange(item_count),
                'item_id': item_ids,
                'a': np.ones(item_count),
                'b': np.zeros(item_count),
                'c': np.zeros(item_count),

            },
                index=item_ids)

            self.item_vector.loc[:, columns] = values.loc[:, columns].values

        else:
            if isinstance(values, pd.DataFrame):
                # self.user_vector = values
                self.item_vector.loc[values.index, columns] = values.loc[:, columns].values

            elif isinstance(values, np.ndarray):
                self.item_vector.loc[:, columns] = values

            else:
                raise TypeError("values的类型必须是pandas.DataFrame或numpy.ndarray")

    def set_items(self, items: pd.DataFrame):
        self.item_vector = items

    def set_users(self, users: pd.DataFrame):
        self.user_vector = users

    def predict_s(self, users, items):
        n = len(users)
        m = len(items)
        assert n == m, "should length(users)==length(items)"

        user_v = self.user_vector.loc[users, ['theta']]

        if isinstance(items, pd.DataFrame) and set(items.columns).intersection({'a', 'b'}):
            item_v = items.loc[:, ['a', 'b']]
        else:
            item_v = self.item_vector.loc[items, ['a', 'b']]

        z = self.D * item_v['a'].values * (user_v['theta'].values - item_v['b'].values)
        # z = alpha * (theta - beta)
        e = np.exp(z)
        s = e / (1.0 + e)
        return s

    def predict_x(self, users, items):
        if isinstance(items, pd.DataFrame):
            self.set_items(items)
        if isinstance(users, pd.DataFrame):
            self.set_theta(users)

        user_count = len(users)
        item_count = len(items)
        theta = self.user_vector.loc[users, 'theta'].values.reshape((user_count, 1))
        a = self.item_vector.loc[items, 'a'].values.reshape((1, item_count))
        b = self.item_vector.loc[items, 'b'].values.reshape((1, item_count))
        # c = self.item_vector.loc[items, 'c'].values.reshape((1, item_count))
        # c = c.repeat(user_count, axis=0)
        z = self.D* a.repeat(user_count, axis=0) * (
                theta.repeat(item_count, axis=1) - b.repeat(user_count, axis=0))
        prob_matrix = sigmod(z)
        # e = np.exp(z)
        # s =   e / (1.0 + e)
        return prob_matrix

    def predict_simple(self, user_id, items: pd.DataFrame):
        """

        Parameters
        ----------
        theta
        items

        Returns
        -------

        """
        if user_id not in self.user_vector.index:
            self.logger.warning('IRT ' + str(user_id) + ' no_irt_theta')
            return pd.Series(data=[np.nan] * len(items), index=items.index), None

        if 'new_difficulty' in items.columns and 'b' not in items.columns:
            # candidate_items.rename(columns={'new_difficulty': 'b'}, inplace=True)
            items['b'] = items['new_difficulty']
        elif 'difficulty' in items.columns and 'b' not in items.columns:
            # candidate_items.rename(columns={'difficulty': 'b'}, inplace=True)
            items['b'] = items['difficulty']
        if 'discrimination' in items.columns and 'a' not in items.columns:
            # candidate_items.rename(columns={'discrimination': 'a'}, inplace=True)
            items['a'] = items['discrimination']
        elif 'a' not in items.columns:
            items.loc[:, 'a'] = 1

        theta = self.user_vector.loc[user_id, 'theta']
        a = items.loc[:, ['a']].values
        b = items.loc[:, ['b']].values
        z = self.D * a * (theta - b)
        prob = sigmod(z)
        # items['irt'] = prob
        return pd.Series(data=prob.flatten(), index=items.index), theta

    def predict_by_theta(self, theta, items: pd.DataFrame):
        """

        Parameters
        ----------
        theta
        items

        Returns
        -------

        """
        b = items.loc[:, ['b']].values
        z = self.D * (theta - b)
        prob = sigmod(z)
        # items['irt'] = prob
        return pd.Series(data=prob.flatten(), index=items.index), theta

    def _prob(self, theta: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray = None):
        """

        Parameters
        ----------
        theta shape=(n,1) n是学生数量
        a  shape=(1,m) m是题目数量
        b  shape=(1,m) m是题目数量
        c  shape=(1,m) m是题目数量

        Returns
        -------

        """

        z = self.D * a * (theta.reshape(len(theta), 1) - b)
        # print(type(z))
        if c is None:
            return sigmod(z)
        return c + (1 - c) * sigmod(z)

    def _object_func(self, theta: np.ndarray, y: np.ndarray, a: np.ndarray = None, b: np.ndarray = None,
                     c: np.ndarray = None):
        """
        .. math::
            Object function  = - \ln L(x;\theta)=-(\sum_{i=0}^n ({y^{(i)}} \ln P + (1-y^{(i)}) \ln (1-P)))
        Parameters
        ----------
        theta
        a
        b
        c

        Returns
        -------
        res : OptimizeResult

        The optimization result represented as a OptimizeResult object.
        Important attributes are: x the solution array, success a Boolean flag indicating if the optimizer
        exited successfully and message which describes the cause of the termination.
        See OptimizeResult for a description of other attributes.
        """

        # 预测值
        y_hat = self._prob(theta=theta, a=a, b=b)
        # 答题记录通常不是满记录的，里面有空值，对于空值设置为0，然后再求sum，这样不影响结果
        # 如果 y_hat 中有0或者1的值 无法求log
        # if (y_hat == 1.0).any() or (y_hat == 0).any():
        #     print('dfsfdf')
        # np.seterr(divide='ignore')
        # try:
        obj = y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)
        # except RuntimeWarning:
        #     print('dfdf')

        # obj[np.isneginf(obj)] = 0
        # np.seterr(divide='warn')
        # 用where处理不了空值，如果是空值，where认为是真
        # obj = - np.sum(np.where(y, np.log(y_hat), np.log(1 - y_hat)))
        # print('obj', obj)
        # 目标函数没有求平均
        return - np.sum(np.nan_to_num(obj, copy=False))

    def _jac_theta(self, theta: np.ndarray, y: np.ndarray, a: np.ndarray = None, b: np.ndarray = None,
                   c: np.ndarray = None):
        # 预测值
        y_hat = self._prob(theta=theta, a=a, b=b)
        # 一阶导数
        # 每一列是一个样本，求所有样本的平均值
        _all = self.D * a * (y_hat - y)

        # 答题记录通常不是满记录的，里面有空值，对于空值设置为0，然后再求sum，这样不影响结果
        grd = np.sum(np.nan_to_num(_all, copy=False), axis=1)
        # grd = grd.reshape(len(grd), 1)
        # print(grd.shape, file=sys.stderr)
        return grd

    def estimate_theta(self, tol=None, options=None, bounds=None):
        """
        已知题目参数的情况下，估计学生的能力值。
        优化算法说明参考 https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize

        Parameters
        ----------
        method 优化算法，可选 CG、Newton-CG、L-BFGS-B
        tol
        options
        bounds
        join join=True，所有学生一起估计；反之，每个学生独立估计

        Returns
        -------

        """
        item_count = len(self.item_vector)
        if 'a' in self.item_vector.columns:
            a = self.item_vector.loc[:, 'a'].values.reshape(1, item_count)
        else:
            a = None

        b = self.item_vector.loc[:, 'b'].values.reshape(1, item_count)
        # if 'c' in self.item_vector.columns:
        #     c = self.item_vector.loc[:, 'c'].values.reshape(1, item_count)
        # else:
        #     c = None

        success = []

        # self._es_res_theta = []

        # 每个人独立估计
        for index, row in self.response_matrix.iterrows():
            # 注意y可能有缺失值
            yy = row.dropna()
            # len(y) == len(y.dropna())
            # 全对的情况
            if yy.sum() == len(yy):
                theta = self.response_sequence.loc[self.response_sequence['user_id'] == index, 'b'].max() + 0.5
                success.append(True)
                # self._es_res_theta.append(res)
            elif yy.sum() == 0:
                # 全错的情况
                theta = self.response_sequence.loc[self.response_sequence['user_id'] == index, 'b'].min() - 0.5
                success.append(True)
            else:
                y = row.values.reshape(1, len(row))
                theta = self.user_vector.loc[index, 'theta']

                res = minimize(self._object_func, x0=[theta], args=(y, a, b), jac=self._jac_theta,
                               bounds=bounds, options=options, tol=tol)
                theta = res.x[0]
                success.append(res.success)

                # self._es_res_theta.append(res)
            # 全错估计值会小于0
            theta = 0 if theta < 0 else theta

            self.user_vector.loc[index, 'theta'] = theta

        return all(success)

    def to_dict(self):
        return self.user_vector['theta'].to_dict()

    @classmethod
    def from_dict(cls, serialize_data):
        obj = cls()
        index = []
        theta = []
        for key, value in serialize_data.items():
            index.append(key)
            theta.append(value)
        obj.set_theta(pd.DataFrame({'theta': theta}, index=index))
        return obj

    def to_pickle(self):
        fh = tempfile.TemporaryFile(mode='w+b')
        self.user_vector.to_pickle(path=fh)
        fh.seek(0)
        data = fh.read()
        fh.close()
        return data

    @classmethod
    def from_pickle(cls, data):
        fh = tempfile.TemporaryFile(mode='w+b')
        fh.write(data)
        fh.seek(0)
        user_vector = pd.read_pickle(fh)
        fh.close()
        obj = cls()
        # cf构造函数 需要把0-1作答结果转成-1，1的形式，
        # 这里不能用构造函数传入数据
        obj.user_vector = user_vector
        return obj

    def get_user_vecotor(self, user_id):
        if self.user_vector is not None and len(self.user_vector) > 0:
            if user_id in self.user_vector.index:
                return self.user_vector.loc[user_id, :]
        return None

    def get_user_info(self, user_id):
        vector = self.get_user_vector(user_id=user_id)
        if not vector:
            return None

        # theta = vector.get('theta',None)
        # theta = vector.get('accuracy_all',None)
        # theta = vector.get('count_all',None)
