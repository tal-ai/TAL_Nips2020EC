import numpy as np 
import pandas as pd
import os

# model_4:
# 水平: l^{T} 题目之间的影响corr: C 最终结果为 base + l^{T}C
# 采用active learning 的 uncertainty sampling尝试结果。


class MyModel:
    def __init__(self):
        """
        Simple example baseline model, which will always query the most answered question available during question 
        selection, and will always predict the most common answer for each question when a student hasn't answered it.
        """
        self.most_popular_answers = None
        self.num_answers = None

    def train_model(self):
        """
        Train a model.
        """
        # # For local evaluation
        # data_path = os.path.normpath('../../data/test_input/test_train_task_4.csv')

        # For full training
        data_path = os.path.normpath('../../data/train_data/train_task_3_4.csv') # for training

        df = pd.read_csv(data_path)
        data_array = df.pivot(index='UserId', columns='QuestionId', values='IsCorrect').to_numpy()
        observation_mask = 1 - np.isnan(data_array).astype(int)
        self.num_answers = observation_mask.sum(axis=0)

        self.most_popular_answers = []

        # Get most popular answer (right or wrong) for each question, to use for predictions.
        for column in data_array.T:
            col_answers = column[np.isnan(column)==False].astype(int)

            # 对应submit_002之前
            # most_popular = np.argmax(np.bincount(col_answers))
            most_popular = np.mean(col_answers)

            self.most_popular_answers.append(most_popular)

        np.save('model_task_4_most_popular.npy', self.most_popular_answers)
        np.save('model_task_4_num_answers.npy', self.num_answers)

    def load(self, most_popular_path, num_answers_path, corr_mat_path):
        """
        Load a model's state, by loading arrays containing the most popular answers.
        Args:
            most_popular_path (string or pathlike): Path to array containing the most popular answer to each question.
            num_answers_path (string or pathlike): Path to array containing the number of recorded answers to each 
                question.
        """
        self.most_popular_answers = np.load(most_popular_path)
        self.num_answers = np.load(num_answers_path)
        self.corr_mat = np.load(corr_mat_path)
        self.corr_mat[np.isnan(self.corr_mat)] = 0
        self.queried_history = []

    def update_queried_history(self, masked_data, masked_binary_data, this_time_queried_each_user):
        self.queried_history.append(this_time_queried_each_user)

    # model 4: update confidence of each question on each step
    def update_confidence(self, masked_data, masked_binary_data, can_query):
        # 在model 3之前，是每个人10道题全都做完，
        # 然后得到长度为10的水平向量，和对应10行的correlation.
        # 在model 4+，每答一道题，就把这道题的水平bias更新进这个人所在行的most_popular_answers_mat里。

        # 首先是每个人这次选了什么题目。
        # print(self.queried_history[-1].shape)  # 正常来说是(984,)的

        # 对每个人的baseline(也就是self.most_popular_answers_mat[i])进行更新
        for user_col in range(masked_binary_data.shape[0]):

            # 这个人回答了哪道题
            this_user_answer_this_q = self.queried_history[-1][user_col]
            # 这个人这些题回答对不对
            this_user_answer_binary = masked_binary_data[user_col, this_user_answer_this_q]
            # 这个人这些题和其他题的联系
            this_user_corr_mat = self.corr_mat[this_user_answer_this_q, :]
            # 大众在挑出的题目上的回答
            mv_selected_answer_binary = self.most_popular_answers[this_user_answer_this_q]
            # 计算自己和大众的差距
            relative_level = this_user_answer_binary - mv_selected_answer_binary  # 现在是一个scalar
            # 计算这个人信心的偏差值
            this_user_bias = np.dot(relative_level, this_user_corr_mat) * 0.4
            # print('更新结果')
            self.most_popular_answers_mat[user_col] += this_user_bias

    def select_feature(self, masked_data, can_query):
        """
        2020年09月16日15:04:23
        1. 优先选择准确率在0.5附近的题目
        """
        # 越接近0.5，区分度越大
        # 注意: most_popular_answers是小数
        # for model_4+, 每行维护一个独立的用户的信心向量
        # 只在time_step=0的时候初始化这个信心矩阵
        if np.all(masked_data == 0):
            print('初始化信心矩阵')
            self.most_popular_answers_mat = np.repeat(self.most_popular_answers.reshape(1, -1),
                                                      repeats=masked_data.shape[0], axis=0)
        else:
            # 如果不是第0轮刚开始，那么更新most_popular_answers_mat的任务交给update_confidence函数
            pass

        # 然后每个人独立选，分别选出自己目前信心最接近0.5的那道题。
        # 注意从第1轮开始，每个人的信心向量可能不同。
        # 如果使用区分度
        diversity = np.abs(self.most_popular_answers_mat - 0.5)

        masked_diversity = can_query * diversity # 广播了一下
        # print(masked_diversity[0, :])
        masked_diversity[masked_diversity == 0] = 1 # 这个值是越小越好的，所以=1就相当于被挖掉了。
        selections = np.argmin(masked_diversity, axis=1)

        # print(masked_diversity)
        return selections

    def predict(self, masked_data, masked_binary_data):
        """
        Produce binary predictions.
        """
        self.most_popular_answers_mat = np.round(self.most_popular_answers_mat)
        self.most_popular_answers_mat[self.most_popular_answers_mat > 1] = 1.
        self.most_popular_answers_mat[self.most_popular_answers_mat < 0] = 0.
        predictions = self.most_popular_answers_mat

        return predictions


