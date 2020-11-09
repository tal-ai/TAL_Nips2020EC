# 因为有缺失值，所以只能两两pair-wise计算相关然后存下来
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
# from tqdm import tqdm

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



corr_mat = np.zeros((binary_data.shape[1], binary_data.shape[1]))
q_num = 948
for i in tqdm(range(q_num)):
    for j in range(q_num):
        target_mat_1 = binary_data[:, i]
        target_mat_2 = binary_data[:, j]
        index_unobserved_1 = np.where(target_mat_1 == -1)
        index_unobserved_2 = np.where(target_mat_2 == -1)
        # 任何一道题没做都认为是缺失值
        index_unobserved_any = np.unique(np.concatenate([index_unobserved_1[0], index_unobserved_2[0]], axis=0))
        # 反过来找到两道题都做了的人
        observed_both = np.array(list(set(range(target_mat_1.shape[0])) - set(index_unobserved_any)))
        
        # 检查是不是都做了
        try:
            assert (sum(target_mat_1[observed_both] == -1) == 0) & (sum(target_mat_2[observed_both] == -1) == 0)
            this_r = pearsonr(target_mat_2[observed_both], target_mat_1[observed_both])[0]
            corr_mat[i, j] = this_r
        except:
            corr_mat[i, j] = 0


np.save('model_task_4_corr_mat_by_questions.npy', corr_mat)