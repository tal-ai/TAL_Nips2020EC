import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import random
from deepctr.models import DeepFM,FLEN
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
import pickle
from tqdm import tqdm_notebook
from collections import Counter 


data_dir = '../data/离线实验_0923/'
os.makedirs(data_dir,exist_ok=True)

student_metadata =  pd.read_csv('../data/public_data/metadata/student_metadata_task_1_2.csv')
question_metadata =  pd.read_csv('../data/public_data/metadata/question_metadata_task_1_2.csv')
answer_metadata =  pd.read_csv('../data/public_data/metadata/answer_metadata_task_1_2.csv')
subject_metadata =  pd.read_csv('../data/public_data/metadata/subject_metadata.csv')


subject_metadata.index = subject_metadata['SubjectId']
subject_metadata.fillna(0,inplace=True)
subject_metadata['ParentId'] = subject_metadata['ParentId'].apply(int)

def add_date_feature(df, col):
    df[col].fillna("0000-08-01 00:00:00.000", inplace=True)
    df['{}_Split'.format(col)] = df[col].apply(
        lambda x: x.split()[0].split('-'))
    df['{}_Year'.format(col)] = df['{}_Split'.format(col)
                                   ].apply(lambda x: int(x[0]))
    df['{}_Month'.format(col)] = df['{}_Split'.format(col)
                                    ].apply(lambda x: int(x[1]))
    df['{}_Year_Month'.format(col)] = df.apply(lambda x:
                                               '{}_{}'.format(
                                                   x['{}_Year'.format(col)],
                                                   x['{}_Month'.format(col)]), axis=1)
    return df


question_metadata['SubjectId'] = question_metadata['SubjectId'].apply(eval)
question_metadata['subject_num'] = question_metadata['SubjectId'].apply(len)


que_to_subject = dict(zip(question_metadata['QuestionId'],question_metadata['SubjectId'])) 



n = 10
que_feature_map = {}
for que_id,subject_list in que_to_subject.items():
    feature_list = []
    feature_list = subject_list+[-1]*n
    que_feature_map[que_id] = feature_list[:n]

def add_subject_col(df,n=8):
    tmp = df['QuestionId'].map(que_feature_map)
    tmp = pd.DataFrame(np.array(tmp.tolist()),columns=["subject_{}".format(i) for i in range(1,n+1)])
    df = pd.concat([tmp,df],axis=1)
    return df


student_metadata['PremiumPupil'].fillna(2,inplace=True)
student_metadata = add_date_feature(student_metadata,'DateOfBirth')


answer_metadata = add_date_feature(answer_metadata,'DateAnswered') 

df_all_train = pd.read_csv('../data/public_data/train_data/train_task_1_2.csv')
## 用户能力预估
question_difficulty_dict = df_all_train.groupby('QuestionId')['IsCorrect'].agg(lambda x:x.mean()).to_dict()
user_ability_dict = df_all_train.groupby('UserId')['IsCorrect'].agg(lambda x:x.mean()).to_dict()


user_history_ability_dict = pickle.load(open('../data/离线实验/user_history_ability_dict.pkl','rb'))
user_history_subject_ability_dict = pickle.load(open('../data/离线实验/user_history_subject_ability_dict.pkl','rb'))


QuestionId2SubjectId = dict(zip(question_metadata['QuestionId'],question_metadata['SubjectId']))
len(QuestionId2SubjectId)


# 获取历史的能力
def get_history_ability_list(df):
    history_ability_list = []
    for ym,UserId in tqdm_notebook(zip(df['DateAnswered_YearMonth'],df['UserId'])):
        history_ability = user_history_ability_dict.get(ym, {}).get(UserId, 0.5)
        history_ability_list.append(history_ability)
    return history_ability_list
# 获取历史知识点的能力
def add_history_subject_ability_list(df,n = 10):
    history_ability_list = []
    for ym, UserId, QuestionId in tqdm_notebook(zip(df['DateAnswered_YearMonth'], df['UserId'], df['QuestionId'])):
        feature_list = []
        for i,SubjectId in enumerate(QuestionId2SubjectId[QuestionId],1):
            value = user_history_subject_ability_dict.get(
                ym, {}).get((UserId,SubjectId), 0.5)
            feature_list.append(value)
        feature_list = feature_list+[0.5]*n
        history_ability_list.append(feature_list[:n])
    tmp = pd.DataFrame(np.array(history_ability_list), columns=[
                       "history_subject_ability_{}".format(i) for i in range(1, n+1)])
    df = pd.concat([tmp,df],axis=1)
    return df


def merge_df(df_path, pkl_path):
    df = pd.read_csv(df_path)
    df = df.merge(answer_metadata, how='left')
    df_add_student_info = df.merge(student_metadata, how='left')
    df_add_subejct = add_subject_col(df_add_student_info, n=10)
    df_add_subejct['question_difficulty'] = df_add_subejct['QuestionId'].map(
        question_difficulty_dict)
    df_add_subejct['user_ability'] = df_add_subejct['UserId'].map(
        user_ability_dict)
    df_add_subejct['DateAnswered_YearMonth'] = df_add_subejct['DateAnswered_Year']*100+df_add_subejct['DateAnswered_Month']
    # 增加history_ability
    df_add_subejct['history_ability'] = get_history_ability_list(
        df_add_subejct)
    df_add_subejct = add_history_subject_ability_list(df_add_subejct,n=10)
    with open(pkl_path, 'wb') as f:
        pickle.dump(df_add_subejct, f)
    return df_add_subejct

# 获取lbe_dict


def get_lbe_dict(df, features):
    df[features] = df[features].fillna('-1', )
    lbe_dict = {}
    for feat in features:
        df[feat] = df[feat].apply(int)
        lbe = LabelEncoder()
        lbe.fit(df[feat])
        lbe_dict[feat] = lbe
    return lbe_dict


def encode_df(df, lbe_dict, feature_names):
    random.seed(0)
    for feat in tqdm_notebook(feature_names):
        df[feat] = df[feat].fillna('-1', )
        lbe = lbe_dict[feat]
        df[feat] = df[feat].apply(int)
        lbe_class_dict = dict(zip(lbe.classes_, lbe.classes_))
        df[feat] = df[feat].map(lambda s: random.choice(
            lbe.classes_) if s not in lbe_class_dict else s)
        df[feat] = lbe.transform(df[feat])
    return df


answer_features = ['GroupId', 'QuizId', 'SchemeOfWorkId','DateAnswered_Year', 'DateAnswered_Month','Confidence']
feature_names = ['subject_1', 'subject_2', 'subject_3', 'subject_4', 'subject_5',
                 'subject_6', 'subject_7', 'subject_8', 'subject_9', 'subject_10',
                 'QuestionId', 'UserId', 'Gender', 'PremiumPupil',
                 'DateOfBirth_Year', 'DateOfBirth_Month','DateAnswered_YearMonth']+answer_features


df_path = '../data/public_data/train_data/train_task_1_2.csv'
pkl_path = os.path.join(data_dir,'raw_train.pkl')
pkl_encode_path = os.path.join(data_dir,'raw_train_encode.pkl')

df = merge_df(df_path,pkl_path) 



df['DateAnswered_YearMonth'] = df['DateAnswered_Year']*100+df['DateAnswered_Month'] 



user_history_ability_dict = {}
for ym in tqdm_notebook(df['DateAnswered_YearMonth'].unique()):
    df_ym = df[df['DateAnswered_YearMonth']<ym].copy()
    if df_ym.shape[0]==0:
        continue
    user_history_ability = df_ym.groupby('UserId')['IsCorrect'].mean().to_dict()
    user_history_ability_dict[ym] = user_history_ability


with open('../data/离线实验/user_history_ability_dict.pkl','wb') as f:
    pickle.dump(user_history_ability_dict,f)


def get_history_ability_list(df):
    history_ability_list = []
    for ym,UserId in tqdm_notebook(zip(df['DateAnswered_YearMonth'],df['UserId'])):
        history_ability = user_history_ability_dict.get(ym, {}).get(UserId, 0.5)
        history_ability_list.append(history_ability)
    return history_ability_list

df['history_ability'] = get_history_ability_list(df) 


question_metadata =  pd.read_csv('../data/public_data/metadata/question_metadata_task_1_2.csv')
question_metadata['SubjectId'] = question_metadata['SubjectId'].apply(eval)
question_metadata['subject_num'] = question_metadata['SubjectId'].apply(len)


QuestionId2SubjectId = dict(zip(question_metadata['QuestionId'],question_metadata['SubjectId']))
len(QuestionId2SubjectId)

new_list = []
for _,row in question_metadata.iterrows():
    for SubjectId in row['SubjectId']:
        new_list.append({"SubjectId":SubjectId,"QuestionId":row['QuestionId']})

question_metadata_flatten = pd.DataFrame(new_list)

df_raw_add_subject = df_raw[['QuestionId', 'UserId', 'IsCorrect',
                             'DateAnswered_YearMonth']].merge(question_metadata_flatten)



user_history_subject_ability_dict = {}
for ym in tqdm_notebook(df_raw_add_subject['DateAnswered_YearMonth'].unique()):
    df_ym = df_raw_add_subject[df_raw_add_subject['DateAnswered_YearMonth']<ym].copy()
    if df_ym.shape[0]==0:
        continue
    user_history_subject_ability = df_ym.groupby(['UserId','SubjectId'])['IsCorrect'].mean().to_dict()
    user_history_subject_ability_dict[ym] = user_history_subject_ability

with open('../data/离线实验/user_history_subject_ability_dict.pkl','wb') as f:
    pickle.dump(user_history_subject_ability_dict,f)


def add_history_subject_ability_list(df,n = 10):
    history_ability_list = []
    for ym, UserId, QuestionId in tqdm_notebook(zip(df['DateAnswered_YearMonth'], df['UserId'], df['QuestionId'])):
        feature_list = []
        for i,SubjectId in enumerate(QuestionId2SubjectId[QuestionId],1):
            value = user_history_subject_ability_dict.get(
                ym, {}).get((UserId,SubjectId), 0.5)
            feature_list.append(value)
        feature_list = feature_list+[0.5]*n
        history_ability_list.append(feature_list[:n])
    tmp = pd.DataFrame(np.array(history_ability_list), columns=[
                       "history_subject_ability_{}".format(i) for i in range(1, n+1)])
    df = pd.concat([tmp,df],axis=1)
    return df


df_tmp = add_history_subject_ability_list(df_raw[:20000]) 