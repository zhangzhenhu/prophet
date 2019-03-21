#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright (c) 2014 Hal.com, Inc. All Rights Reserved
#
"""
模块用途描述

Authors: zhangzhenhu(acmtiger@163.com)
Date:    2018/8/13 19:35
"""
import sys
import argparse
from feature import Feature
from evaluation import EvaluationMany
import model as md

__version__ = 1.0


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


def main(options=None):
    pass

    ft = Feature()
    ft.fit()
    print(ft.feature_names)
    print("==============数据信息==============")
    print(ft.describe())
    print("--------------训练数据--------------")
    print(ft.features_train.describe().transpose())
    print("--------------测试数据--------------")
    print(ft.features_test.describe().transpose())
    ft.fillna()
    models = [
        # md.POKS(ft),
        md.CF(feature=ft, kind='user_base'),
        md.CF(feature=ft, kind='item_base'),
        md.Irt(ft),
        md.LR(ft),
        md.AdaBoost(ft),
        md.DecisionTree(ft),
        md.GradientBoostingTree(ft),
        md.RandomForest(ft),
        md.Xgboost(ft),
    ]
    for model in models:
        model.fit()
        model.test()
        # model.evaluation.print_evaluate()
        # model.evaluation.plot_auc()
    ev = EvaluationMany(models)
    print("==============模型效果==============")
    print(ev.evaluate())
    print(ev.evaluate_no())
    # ev.plot_allin()

    return ev


if __name__ == "__main__":

    parser = init_option()
    options = parser.parse_args()

    if options.input:

        options.input = open(options.input)
    else:
        options.input = sys.stdin
    main(options)
