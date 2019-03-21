import argparse
import numpy as np
import pandas as pd
from .model import Model, Feature, Evaluation


class CF(Model):
    name = "cf_model"
    # 模型参数
    description = "cf_model"

    def __init__(self, feature: Feature, evaluation: Evaluation = None, kind='user_base'):
        super(CF, self).__init__(feature=feature, evaluation=evaluation)
        self.kind = kind
        self.description = kind

    def select_features(self, df_x: pd.DataFrame):
        return df_x[['user_id', 'item_id', 'answer']].copy()

    def preProcess(self):

        self.data_ori.loc[self.data_ori['answer'] == 0, 'answer'] = -1
        # 删除作答数目小于5的题目、答题数小于5的同学
        count_lq = self.data_ori.groupby('user_id')['item_id'].count().reset_index()
        count_lq = count_lq.rename(columns={'item_id': 'count'})
        count_final = count_lq[count_lq['count'] >= 5]
        users_final = self.data_ori.merge(count_final, left_on='user_id', right_on='user_id', how='inner')
        users_final = pd.DataFrame(users_final, columns=['user_id', 'item_id', 'answer'])
        # 剔除答题人数小于设定要求的试题的答题数据
        count_stu = self.data_ori.groupby('item_id')['user_id'].count().reset_index()
        count_stu = count_stu.rename(columns={'user_id': 'count'})
        count_final = count_stu[count_stu['count'] >= 5]
        users_final = users_final.merge(count_final, left_on='item_id', right_on='item_id', how='inner')
        users_final = pd.DataFrame(users_final, columns=['user_id', 'item_id', 'answer'])
        # 转化为矩阵
        users_final = users_final.pivot_table('answer', 'user_id', 'item_id')
        users_final = users_final.fillna(0)
        self.user_name = users_final.index.tolist()
        self.item_name = users_final.columns.tolist()
        stu_ans = users_final.values
        sparsity = float(len(stu_ans.nonzero()[0]))
        sparsity /= (users_final.shape[0] * users_final.shape[1])
        sparsity *= 100
        # print('稠密度: {:4.2f}%'.format(sparsity))
        self.users_final = users_final
        return self.users_final

    def fast_similarity(self, kind, epsilon=1e-9):

        stu_ans = self.users_final.values
        if kind == 'user':
            sim = stu_ans.dot(stu_ans.T) + epsilon
        elif kind == 'item':
            sim = stu_ans.T.dot(stu_ans) + epsilon
        norms = np.array([np.sqrt(np.diagonal(sim))])
        return (sim / norms / norms.T)

    def fit(self):

        self.data_ori = self.select_features(self.feature.response_history)
        self.preProcess()
        self.user_similarity = self.fast_similarity(kind='user')
        self.item_similarity = self.fast_similarity(kind='item')

        # 评估训练集上的效果
        self.train_x = self.feature.features_train[['user_id', 'item_id', 'sf_no']].copy()
        self.train_y = self.feature.label_train.values
        # self.feature_names = ['answer']
        self.train_y_pred = self.predict(self.train_x)
        self.train_ev = self.evaluation.evaluate(y_true=self.train_y, y_pred=self.train_y_pred, threshold=0.3)

        return self

    def predict(self, x: pd.DataFrame = None, kind=None) -> np.ndarray:
        if kind is None:
            kind = self.kind
        self.description = kind

        if x is None:
            x = self.feature.response_test
        # response_test = x.copy()
        response_test = x[['user_id', 'item_id']].copy()
        k = 0.7
        response_test['predict'] = np.nan  # 预测值
        response_test['sign'] = np.nan  # 预测分支
        for index, row in response_test.iterrows():
            user_id = row['user_id']
            item_id = row['item_id']
            if kind == 'user_base':
                if user_id in self.user_name and item_id in self.item_name:
                    # 如果该题存在历史答题数据，则协同过滤
                    row_num = self.user_name.index(user_id)
                    col_num = self.item_name.index(item_id)
                    top_k_users = [i for i in range(len(self.user_similarity[:, row_num])) if
                                   self.user_similarity[i, row_num] > k and i != row_num]
                    if len(top_k_users) == 0:
                        # answer_pred = self.data_ori.loc[self.data_ori['item_id'] == item_id, 'answer'].mean()
                        response_test.loc[index, 'sign'] = 2
                    else:
                        if sum(self.users_final.values[:, col_num][top_k_users]) == 0:
                            pass
                        else:
                            answer_pred = self.user_similarity[row_num, :][top_k_users].dot(
                                self.users_final.values[:, col_num][top_k_users])
                            answer_pred /= np.sum(np.abs(self.user_similarity[row_num, :][top_k_users]))
                            response_test.loc[index, 'sign'] = 1
                            response_test.loc[index, 'predict'] = answer_pred
            else:
                if user_id in self.user_name and item_id in self.item_name:
                    # 如果该学生存在历史答题数据，则协同过滤
                    row_num = self.user_name.index(user_id)
                    col_num = self.item_name.index(item_id)
                    top_k_items = [i for i in range(len(self.item_similarity[:, col_num])) if
                                   self.item_similarity[i, col_num] > k and i != col_num]
                    if len(top_k_items) == 0:
                        # answer_pred = self.data_ori.loc[self.data_ori['user_id'] == user_id, 'answer'].mean()
                        response_test.loc[index, 'sign'] = 4
                    else:
                        if sum(self.users_final.values[row_num, :][top_k_items]) == 0:
                            pass
                        else:
                            answer_pred = self.item_similarity[col_num, :][top_k_items].dot(
                                self.users_final.values[row_num, :][top_k_items].T)
                            answer_pred /= np.sum(np.abs(self.item_similarity[:, col_num][top_k_items]))
                            response_test.loc[index, 'sign'] = 3
                            response_test.loc[index, 'predict'] = answer_pred

        y_pred = response_test['predict'].values
        assert y_pred.shape[0] == x.shape[0]
        return y_pred

    def test(self):
        """
        评估测试集上的效果
        :return:K
        """
        self.test_x = self.feature.features_test[['user_id', 'item_id', 'sf_no']]
        self.test_y = self.feature.label_test.values
        self.test_y_pred = self.predict(self.test_x)

        # self.test_y_pred = self.predict()
        # self.test_y = self.feature.response_test['answer'].copy()
        self.test_ev = self.evaluation.evaluate(y_true=self.test_y, y_pred=self.test_y_pred, threshold=0)
