import pandas as pd
import numpy as np
import glob
files = glob.glob('/Users/haoyang/tmp/praise/nips比赛/t4/整理后/*.csv')


def get_weighted_votes(votes, weights):
    weighted_votes = np.zeros((votes.shape[0], sum(weights))).astype(np.int)
    index = 0
    for i in range(votes.shape[1]):
        weighted_votes[:, index: index + weights[i]] = np.tile(votes[:, i:i+1], weights[i])
        index += weights[i]
    return weighted_votes


# 目前效果好的:
useful_file_list = [
 '/Users/haoyang/tmp/praise/nips比赛/t4/整理后/best_now.csv', # 662
 '/Users/haoyang/tmp/praise/nips比赛/t4/整理后/onn_epoch_7.csv', # 661
 '/Users/haoyang/tmp/praise/nips比赛/t4/整理后/onn_epoch_10.csv', # 662
 '/Users/haoyang/tmp/praise/nips比赛/t4/整理后/kangyu_submit_task_2.csv', # 65+
 '/Users/haoyang/tmp/praise/nips比赛/t4/整理后/jiahao_64.csv', # 64+
 '/Users/haoyang/tmp/praise/nips比赛/t4/整理后/proba_lr_stack_0.csv', # 660
 '/Users/haoyang/tmp/praise/nips比赛/t4/整理后/jiahao_662.csv', # 662
 '/Users/haoyang/tmp/praise/nips比赛/t4/整理后/jiahao_663.csv', # 663
    '/Users/haoyang/tmp/praise/nips比赛/t4/submission_task_2.csv' # 653
]

weight_list =  [2, 2, 2, 1, 1, 2, 2, 3, 1]


df_list = []
for i in range(len(useful_file_list)):
    df_list.append(pd.read_csv(useful_file_list[i])[['UserId', 'QuestionId', 'AnswerValue']])

df_merge = pd.concat(df_list, axis=1)

votes = np.array([df_merge.iloc[:, 3 * col_num + 2] for col_num in range(len(useful_file_list))]).T

weighted_votes = get_weighted_votes(votes, weight_list)

voted_answer = []
for i in tqdm(range(weighted_votes.shape[0])):
    a = np.bincount(weighted_votes[i, :])
    voted_answer.append(np.argmax(a))


df_merge_7_final = df_merge.iloc[:, :2].copy()
df_merge_7_final['AnswerValue'] = voted_answer
df_merge_7_final.to_csv('/Users/haoyang/tmp/praise/nips比赛/t4/整理后/merged_result_9.csv', index=False)


# add task 1 best


train_df = pd.read_csv('/Users/haoyang/DevelopProjects/TAL/projects_2020/nips_task/data/train_data/train_task_1_2.csv')
q_no_dupli_df = train_df.drop_duplicates('QuestionId')[['QuestionId', 'CorrectAnswer']]
correct_answer_dict = {}

for i in range(27613):
    correct_answer_dict[q_no_dupli_df.iloc[i]['QuestionId']] = q_no_dupli_df.iloc[i]['CorrectAnswer']



task_1_best_path = '/Users/haoyang/DevelopProjects/TAL/projects_2020/nips_task/task_1/7679.csv'

task_1_best_df = pd.read_csv(task_1_best_path)
task_1_best_df = pd.merge(task_1_best_df, df_merge_7_final, on=['UserId', 'QuestionId'])


task_1_best_df['correct_answer'] = task_1_best_df.QuestionId.map(correct_answer_dict)

# label correction by task 1
task_1_best_df.AnswerValue = task_1_best_df.correct_answer * (task_1_best_df.probability > 0.6) + task_1_best_df.AnswerValue * (task_1_best_df.probability <= 0.6)


task_1_best_df[['UserId', 'QuestionId', 'AnswerValue']].to_csv('submission_task_2.csv', index=False)
