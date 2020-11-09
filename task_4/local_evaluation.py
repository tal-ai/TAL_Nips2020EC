from submission_model_task_4 import Submission
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
import os


def pivot_df(df, values):
    """
    Convert dataframe of question and answerrecords to pivoted array, filling in missing columns if some questions are 
    unobserved.
    """
    # 每个user在每个question上的作答情况，
    # data表里的作答情况是AnswerValue，也就是选项；
    # data_binary表里的作答情况是IsCorrect，也就是是否做对了。

    data = df.pivot(index='UserId', columns='QuestionId', values=values)

    # 加入没有被观察的题目列，用nan填充。输出的时候统一用-1代替。
    # Add rows for any questions not in the test set
    data_cols = data.columns
    all_cols = np.arange(948)
    missing = set(all_cols) - set(data_cols)
    for i in missing:
        data[i] = np.nan
    data = data.reindex(sorted(data.columns), axis=1)

    data = data.to_numpy()
    data[np.isnan(data)] = -1
    return data


if __name__ == "__main__":
    data_path = os.path.normpath('../../data/test_input/valid_task_4.csv')
    df = pd.read_csv(data_path)
    data = pivot_df(df, 'AnswerValue')
    '''
            q_0 q_1 q_2 ... q_947
    user_0   1   3   -1       2
    user_1   4   2    1       -1
    user_2   -1  -1   2        -1
    .
    .
    
    '''
    binary_data = pivot_df(df, 'IsCorrect')

    '''
            q_0 q_1 q_2 ... q_947
    user_0   -1  0   1       0
    user_1   1   1   1       -1
    user_2   -1  -1  1       -1
    .
    .
    '''
    # Array containing -1 for unobserved, 0 for observed and not target (can query), 1 for observed and target (held out
    # for evaluation).

    # -1是没有数据的，0是训练集，1是target，所以在
    targets = pivot_df(df, 'IsTarget')  # 984 * 948
    # print(targets)
    # print(targets.shape)
    # print('第一行的valid set 长度', sum(targets[0, :] > 0)) # 15
    # print('第二行的valid set 长度', sum(targets[1, :] > 0)) # 115
    # print('第三行的valid set 长度', sum(targets[2, :] > 0)) # 64
    # 每个人测试集(valid set)的大小不同

    observations = np.zeros_like(data)  # 984 * 948 大小的zeros
    masked_data = data * observations   # 初始状态，masked_data是全0
    masked_binary_data = binary_data * observations # masked_binary_data也是全0

    can_query = (targets == 0).astype(int) # 和前面的targets形状一样；只有可以被选择的题目被列在这里。
    # print(can_query)
    # print(can_query.shape)

    submission = Submission()

    # 记录每次queried question index.
    # queried_history = []  # 注意: 尽量不改变这个脚本做的事情。把queried_history转移到model内部记录。

    print('开始进行题目选择')
    for i in range(10):
        print('Feature selection step {}'.format(i+1))
        next_questions = submission.select_feature(masked_data, masked_binary_data, can_query)
        # Validate not choosing previously selected question here

        # print('第{}次选出的题目:'.format(i))
        # print(next_questions)

        # 看每个user的回答
        for i in range(can_query.shape[0]):
            # Validate choosing queriable target here

            # 验证这道题可不可以问
            assert can_query[i, next_questions[i]] == 1
            # 这道题目在当前这步问过了，之后不可被问了。
            can_query[i, next_questions[i]] = 0

            # 验证这道题目是不是可以被问的。第0轮之前，observation所有格子都是0.
            # Validate choosing unselected target here
            assert observations[i, next_questions[i]] == 0

            # 更新observations
            # 然后被问过了，以后不可以被问了。1代表问过了
            observations[i, next_questions[i]] = 1
            # masked_data 在第i轮有i个选项信息
            masked_data = data * observations
            # masked_binary_data在第i轮有i个正确/错误信息
            masked_binary_data = binary_data * observations
        
        # Update model with new data, if required
        # 这样每次把masked得到的数据输入模型来update
        submission.update_model(masked_data, masked_binary_data, can_query)
    # print(np.array(submission.model.queried_history).shape)
    # 10轮之后，我们有model, 有所有query过的数据，用这些来预测每个人在所有题目上的表现
    preds = submission.predict(masked_data, masked_binary_data)

    # 只有targets==1的位置是valid set,是我们要计算准确率的。
    pred_list = preds[np.where(targets == 1)]

    # 拿出binary(也就是IsCorrect矩阵)里valid set上的数据，是一个(n,)的array
    target_list = binary_data[np.where(targets==1)]
    # print(target_list)
    # 然后
    # print(target_list.dtype)
    acc = (pred_list == target_list).astype(int).sum()/len(target_list)
    print('Final accuracy: {}'.format(acc))
