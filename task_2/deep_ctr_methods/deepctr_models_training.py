import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'


import sys
sys.path.append('/share/tabchen/')
sys.path.append('../')
from nips_utils.metrics_utils import get_model_metrics,get_multi_class_report

import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import random
from DeepCTR.deepctr.models import DeepFM,FLEN,ONN,FGCNN
from DeepCTR.deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
import pickle
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from tqdm import tqdm_notebook


data_dir = '../data/离线实验_0923/' 

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



question_metadata =  pd.read_csv('../data/public_data/metadata/question_metadata_task_1_2.csv')
question_metadata['SubjectId'] = question_metadata['SubjectId'].apply(eval)
question_metadata['subject_num'] = question_metadata['SubjectId'].apply(len)


que_to_subject = dict(zip(question_metadata['QuestionId'],question_metadata['SubjectId'])) 


que_feature_map = {}
n = 8
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


df_train = pd.read_csv(os.path.join(data_dir, 'train_task_1_2.csv')) 

df_train = df_train.merge(student_metadata,how='left')
df_train = df_train.merge(answer_metadata,how='left')

df_train['label'] = df_train['IsCorrect'] 

df_train = add_subject_col(df_train) 


with open(os.path.join(data_dir, 'train_task_1_2.pkl'),'wb') as f:
    pickle.dump(df_train,f)



df_dev = pd.read_csv(os.path.join(data_dir, 'dev_task_1_2.csv')) 

df_dev = df_dev.merge(student_metadata,how='left')
df_dev = df_dev.merge(answer_metadata,how='left')

df_dev['label'] = df_dev['IsCorrect'] 
df_dev = add_subject_col(df_dev) 
with open(os.path.join(data_dir, 'dev_task_1_2.pkl'),'wb') as f:
    pickle.dump(df_dev,f)

df_test = pd.read_csv('../data/starter_kit/submission_templates/submission_task_1_2.csv') 


df_test = df_test.merge(student_metadata,how='left')
df_test = add_subject_col(df_test)

with open(os.path.join(data_dir, 'test_task_1_2.pkl'),'wb') as f:
    pickle.dump(df_test,f)


# 获取lbe_dict
def get_lbe_dict(df,features):
    df[features] = df[features].fillna('-1', )
    lbe_dict = {}
    for feat in features:
        df[feat] = df[feat].apply(int)
        lbe = LabelEncoder()
        lbe.fit(df[feat])
        lbe_dict[feat] = lbe
    return lbe_dict


feature_names = ['QuestionId', 'UserId', 'Gender','PremiumPupil',
                 'DateOfBirth_Year', 'DateOfBirth_Month']+["subject_{}".format(i) for i in range(1,n+1)]



df_path = '../data/离线实验/raw_train.pkl'
df = pickle.load(open(df_path,'rb'))

lbe_dict = get_lbe_dict(df,feature_names) 

with open(os.path.join(data_dir, 'lbe_dict.pkl'),'wb') as f:
    pickle.dump(lbe_dict,f)


df_all_train = pd.read_csv('../data/public_data/train_data/train_task_1_2.csv') 


df_all_train = df_all_train.merge(df_all_train,how='left')
df_all_train = add_subject_col(df_all_train)

with open(os.path.join(data_dir, 'raw_train.pkl'),'wb') as f:
    pickle.dump(df_all_train,f)

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

def get_split_info(ids, n_splits=10, max_random_state=5):
    all_split_info = {}
    for random_state in range(max_random_state):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_info = {}
        for kf_i, (train_ids, test_ids) in enumerate(kf.split(ids)):
            train_ids, dev_ids = train_test_split(train_ids, test_size=0.1,random_state=random_state)
            split_info[kf_i] = {"train_ids": list(train_ids), "dev_ids": list(
                dev_ids), "test_ids": list(test_ids)}
        all_split_info[random_state] = split_info
    return all_split_info

def get_split_id(all_split_info,kf_i,random_state=0):
    split_info = all_split_info[random_state][kf_i]
    train_ids, dev_ids,test_ids = split_info['train_ids'],split_info['dev_ids'],split_info['test_ids']
    return train_ids, dev_ids,test_ids



class DataGet():
    def __init__(self, df_path,split_info_path=None,n_splits=10, max_random_state=5):
        self.df = pickle.load(open(df_path,'rb'))
        self.df['label'] = self.df['IsCorrect']
        self.id_col = 'AnswerId'
        ids = self.df[self.id_col]
        if not split_info_path is None and os.path.exists(split_info_path):
            self.split_info_path = pickle.load(open(split_info_path,'rb'))
        else:
            self.all_split_info = get_split_info(ids, n_splits, max_random_state)
            if not split_info_path is None:
                with open(split_info_path,'wb') as f:
                    pickle.dump(self.all_split_info,f)
                
    def get_data_index(self, kf_i, random_state):
        split_info = self.all_split_info[random_state][kf_i]
        train_ids, dev_ids, test_ids = split_info['train_ids'], split_info['dev_ids'], split_info['test_ids']
        return train_ids, dev_ids, test_ids


    def get_index_data(self, ids):
        df_seg = self.df[self.df[self.id_col].isin(ids)].copy()
        df_seg.index = range(len(df_seg))
        return df_seg

    def get_data(self, kf_i, random_state):
        train_ids, dev_ids, test_ids = self.get_data_index(kf_i=kf_i, random_state=random_state)
        df_train = self.get_index_data(train_ids)
        df_dev = self.get_index_data(dev_ids)
        df_test = self.get_index_data(test_ids)
        return df_train, df_dev, df_test



question_difficulty_dict = df_train.groupby('QuestionId')['IsCorrect'].agg(lambda x:x.mean()).to_dict()
user_ability_dict = df_train.groupby('UserId')['IsCorrect'].agg(lambda x:x.mean()).to_dict()



sparse_features = ['QuestionId', 'UserId', 'Gender','PremiumPupil',
                 'DateOfBirth_Year', 'DateOfBirth_Month']+["subject_{}".format(i) for i in range(1,n+1)]
dense_features = ['question_difficulty','user_ability']
target = ['label']


def encode_df(df, lbe_dict,feature_names):
    df = df.copy()
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


train_feature = encode_df(df_train, lbe_dict,sparse_features) 

test_feature = encode_df(df_dev, lbe_dict,sparse_features) 

train_feature['question_difficulty'] = train_feature['QuestionId'].map(question_difficulty_dict)
train_feature['user_ability'] = train_feature['UserId'].map(user_ability_dict)
test_feature['question_difficulty'] = test_feature['QuestionId'].map(question_difficulty_dict)
test_feature['user_ability'] = test_feature['UserId'].map(user_ability_dict)

fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=train_feature[feat].nunique(),embedding_dim=4)
                       for i,feat in enumerate(sparse_features)] + [DenseFeat(feat, 1,)
                      for feat in dense_features]

dnn_feature_columns = fixlen_feature_columns
linear_feature_columns = fixlen_feature_columns

feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)


train = pd.concat([train_feature,test_feature[20000:]])
# train = train_feature
train['AnswerValue'] = train['AnswerValue']-1
test = test_feature[:20000]
test['AnswerValue'] = test['AnswerValue']-1

train_model_input = {name: train_data[name] for name in feature_names}
dev_model_input = {name: dev_data[name] for name in feature_names}


answer_features = ['GroupId', 'QuizId', 'SchemeOfWorkId','DateAnswered_Year', 'DateAnswered_Month','Confidence'] 
sparse_features = ['QuestionId', 'UserId', 'Gender', 'PremiumPupil', 'DateAnswered_YearMonth',
                   'DateOfBirth_Year', 'DateOfBirth_Month']+["subject_{}".format(i) for i in range(1, n+1)]+answer_features
dense_features = ['question_difficulty', 'user_ability', 'history_ability']
target = ['label']


feature_names = sparse_features+dense_features 

train_data = df_train
dev_data = df_dev


field_info_map = {'QuestionId': "item",
                  'UserId': "user",
                  'Gender': "user",
                  'PremiumPupil': "user",
                  'DateOfBirth_Year': "user",
                  'DateOfBirth_Month': "user",
                  'history_subject_ability_1':"item",
                  'history_subject_ability_2':"item",
                  'history_subject_ability_3':"item",
                  'history_subject_ability_4':"item",
                  'history_subject_ability_5':"item",
                  'history_subject_ability_6':"item",
                  'history_subject_ability_7':"item",
                  'history_subject_ability_8':"item",
                  'history_subject_ability_9':"item",
                  'history_subject_ability_10':"item",
                  'subject_1': "item",
                  'subject_2': "item",
                  'subject_3': "item",
                  'subject_4': "item",
                  'subject_5': "item",
                  'subject_6': "item",
                  'subject_7': "item",
                  'subject_8': "item",
                  'subject_9': "item",
                  'subject_10': "item",
                  'GroupId': "user",
                  'QuizId': "user",
                  'SchemeOfWorkId': "user",
                  'DateAnswered_Year': "user",
                  'DateAnswered_Month': "user",
                  'DateAnswered_YearMonth':"user",
                  'Confidence': "user",
                  'question_difficulty': "user",
                  'user_ability': "user",
                  'history_ability': "user"}


emb_features = ['item_id_emb','user_id_emb'] 

field_info = [field_info_map[x] for x in feature_names] 

fixlen_feature_columns = [
    SparseFeat(feat,
                vocabulary_size=len(lbe_dict[feat].classes_),
                embedding_dim=4,
                group_name=field_info[i])
    for i, feat in enumerate(sparse_features)
] + [DenseFeat(
    feat,
    1,
) for feat in dense_features]
# +[DenseFeat(
#     feat,
#     100
# ) for feat in emb_features]

def get_input(df):
    return {name: np.vstack(df[name]) if 'emb' in name else df[name] for name in feature_names}

df_test = pickle.load(open(os.path.join(data_dir, 'final_test_encode.pkl'),'rb')) 

final_test_input = get_input(df_test) 



# models

model_dir = 'model/task2/ONN/更多特征_12'
os.makedirs(model_dir,exist_ok=True)
# 4.Define Model,train,predict and evaluate
model = ONN(linear_feature_columns, dnn_feature_columns,task='multiclass',num_labels=4)
model.compile("adam", "sparse_categorical_crossentropy",
              metrics=['sparse_categorical_crossentropy'])
target_col = 'label' 

history = model.fit(train_model_input, df_train[target_col],
                    batch_size=5120, epochs=5, verbose=2, 
#                     validation_data=(dev_model_input,dev_data[target_col]),
#                     callbacks=[early_stopping_monitor, checkpoint]
                   )

pred_ans = model.predict(final_test_input, batch_size=2014) 

pred_label_list = pred_ans.argmax(axis=1)+1 
df_test['AnswerValue'] = pred_label_list 

save_dir = '../data/提交/task2/'
save_path = os.path.join(save_dir,'submission_task_2.csv')  df_test[['UserId','QuestionId','AnswerValue']].to_csv(save_path,index=False)