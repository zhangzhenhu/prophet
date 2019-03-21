#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright (c) 2014 Hal.com, Inc. All Rights Reserved
#
"""
模块用途描述

Authors: zhangzhenhu(acmtiger@163.com)
Date:    2018/8/13 14:47
"""
from sklearn import metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import ceil


def accuracy_error(y_true: np.ndarray, y_pred: np.ndarray):
    return abs(y_true.mean() - y_pred.mean())


class Evaluation:
    def __init__(self, model=None):
        self.model = model
        self.key_index = ['uae', 'accuracy', 'precision', 'mae', 'mse', 'auc_score', 'f1_score', ]

    def evaluate(self, y_true: np.ndarray = None, y_pred: np.ndarray = None, threshold: float = 0.5) -> dict:
        # if y_true is None and self.model is not None:
        #     y_true = self.model.y_true
        # if y_pred is None and self.model is not None:
        #     y_pred = self.model.y_pred
        assert y_pred is not None
        assert y_true is not None
        y_pred_binary = y_pred.copy()

        valid_mask = ~np.isnan(y_pred_binary)
        # 非空值，覆盖率
        coverage = valid_mask.mean()
        if coverage == 0:
            accuracy = 0
            tpr = 0
            fpr = 0
            precision_0 = 0
            precision_1 = 0
            uae = 0
            mae = 0
            mse = 0
            auc_score = 0
            f1_score = 0

        else:
            y_pred = y_pred[valid_mask]
            y_pred_binary = y_pred_binary[valid_mask]
            y_true = y_true[valid_mask]
            # y_pred = y_pred_binary.copy()
            y_pred_binary[y_pred_binary >= threshold] = 1
            y_pred_binary[y_pred_binary < threshold] = 0
            accuracy = metrics.accuracy_score(y_true, y_pred_binary)
            tpr = metrics.precision_score(y_pred_binary, y_true, pos_label=1)
            try:
                fpr = 1-metrics.precision_score(y_pred_binary, y_true, pos_label=0)
            except:
                fpr = 0
            precision_1 = metrics.precision_score(y_true, y_pred_binary, pos_label=1)
            precision_0 = metrics.precision_score(y_true, y_pred_binary, pos_label=0)
            uae = accuracy_error(y_true, y_pred_binary)
            mae = metrics.mean_absolute_error(y_true, y_pred)
            mse = metrics.mean_squared_error(y_true, y_pred)
            try:
                auc_score = metrics.roc_auc_score(y_true=y_true, y_score=y_pred)
            except ValueError:
                auc_score = 0

            f1_score = metrics.f1_score(y_true, y_pred_binary)
        ret = {'accuracy': accuracy,
               'coverage': coverage,
               'tpr': tpr,
               'fpr': fpr,
               'precision_1': precision_1,
               'precision_0': precision_0,
               'uae': uae,
               'mae': mae,
               'mse': mse,
               'auc_score': auc_score,
               'f1_score': f1_score,
               }
        self.key_index = list(ret.keys())
        ret['name'] = self.model.name if self.model is not None else "NULL"
        ret['description'] = self.model.description if self.model is not None else "NULL"
        return ret

    def print_evaluate(self, y_true=None, y_pred=None, threshold=0.5):
        ret = self.evaluate(y_true, y_pred, threshold)

        for k in ['name', ] + self.key_index + ['description']:
            v = ret[k]
            if isinstance(v, float):
                print("%s: %0.5f" % (k, v), end=' ')
            else:
                print("%s: %s" % (k, v), end=' ')
        print("")

    def plot_auc(self, y_true=None, y_pred=None, gca=None):
        if y_true is None and self.model is not None:
            y_true = self.model.y_true
        if y_pred is None and self.model is not None:
            y_pred = self.model.y_pred
        assert y_pred is not None
        assert y_true is not None

        fpr, tpr, _ = metrics.roc_curve(y_true=y_true, y_score=y_pred)
        roc_auc = metrics.auc(fpr, tpr)
        # plt.figure()
        lw = 2
        if gca is None:
            gca = plt.figure().gca()
        gca.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        gca.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        gca.set_xlim([0.0, 1.0])
        gca.set_ylim([0.0, 1.05])
        gca.set_xlabel('False Positive Rate')
        gca.set_ylabel('True Positive Rate')
        gca.set_title(' %s roc curve' % self.model.name)
        gca.legend(loc="lower right")


class EvaluationMany(list):

    def evaluate_no(self, target="test") -> pd.DataFrame:
        if len(self) == 0:
            return pd.DataFrame()

        records = []
        key_index = self[0].evaluation.key_index.copy()
        key_index.append('description')
        for model in self:
            if target == "train":
                x = model.train_x
                y_true = model.train_y
                y_pred = model.train_y_pred
            else:
                x = model.test_x
                y_true = model.test_y
                y_pred = model.test_y_pred

            for i in range(1, 15):
                y_true_selected = y_true[x['sf_no'] == i]
                y_pred_selected = y_pred[x['sf_no'] == i]
                record = model.evaluation.evaluate(y_true=y_true_selected, y_pred=y_pred_selected)
                record['题号'] = i
                records.append(record)

            y_true_selected = y_true[x['sf_no'] > i]
            y_pred_selected = y_pred[x['sf_no'] > i]
            record = model.evaluation.evaluate(y_true=y_true_selected, y_pred=y_pred_selected)
            record['题号'] = i + 1
            records.append(record)

        df = pd.DataFrame.from_records(data=records)
        df.set_index('name', inplace=True)
        df.sort_index(axis=1, )
        key_index.append('题号')
        return df[key_index].sort_values(key_index[0], ascending=False)

    def evaluate(self, target="test") -> pd.DataFrame:
        if len(self) == 0:
            return pd.DataFrame()

        key_index = self[0].evaluation.key_index.copy()
        key_index.append('description')
        if target == "train":
            records = [model.train_ev for model in self]
        else:
            records = [model.test_ev for model in self]

        df = pd.DataFrame.from_records(data=records)

        df.set_index('name', inplace=True)
        df.sort_index(axis=1, )
        return df[key_index].sort_values(key_index[0], ascending=False)

    def plot_separated(self, target="test"):

        rows = int(ceil(len(self) / 2))
        columns = 2
        fig, axes = plt.subplots(rows, columns, figsize=(columns * 5, rows * 5))
        if target == "train":
            for i, model in enumerate(self):
                x = i // 2
                y = i % 2
                # plt.subplot(rows, columns, i)
                model.evaluation.plot_auc(y_true=model.train_y, y_pred=model.train_y_pred, gca=axes[x, y])
        else:
            for i, model in enumerate(self):
                x = i // 2
                y = i % 2
                # plt.subplot(rows, columns, i)
                model.evaluation.plot_auc(y_true=model.test_y, y_pred=model.test_y_pred, gca=axes[x, y])

    def plot_allin(self, target="test"):

        data = []
        if target == "train":
            for model in self:
                valid_mask = ~np.isnan(model.train_y_pred)
                # 预测结果中存在空值
                if valid_mask.mean() == 0:
                    data.append((None, None, None))
                    continue
                score = metrics.roc_curve(y_true=model.train_y[valid_mask], y_score=model.train_y_pred[valid_mask])
                data.append(score)
        else:
            for model in self:
                valid_mask = ~np.isnan(model.test_y_pred)
                if valid_mask.mean() == 0:
                    data.append((None, None, None))
                    continue
                score = metrics.roc_curve(y_true=model.test_y[valid_mask], y_score=model.test_y_pred[valid_mask])
                data.append(score)

        gca = plt.figure(figsize=(10, 10)).gca()
        # roc_auc = metrics.auc(fpr, tpr)

        lw = 2
        for item, model in zip(data, self):
            fpr, tpr, _ = item
            if fpr is None:
                continue
            try:
                roc_auc = metrics.auc(fpr, tpr)
            except:
                pass
            plt.plot(fpr, tpr,  # color='darkorange',
                     lw=lw, label='%s (area = %0.2f)' % (model.name, roc_auc))

        gca.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        gca.set_xlim([0.0, 1.0])
        gca.set_ylim([0.0, 1.05])
        gca.set_xlabel('False Positive Rate')
        gca.set_ylabel('True Positive Rate')
        gca.set_title('roc curve')
        gca.legend(loc="lower right")
        plt.show()
