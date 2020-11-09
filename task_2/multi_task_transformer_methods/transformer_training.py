import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

import random
import time
import warnings
import re
import gc
import pickle
import copy
import math
import datetime

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import transformers

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from tqdm.autonotebook import tqdm
import utils


EPOCHS = 300
MAX_LEN = 100

MODEL_NAME = "use_150_days+user+group+quiz+difficulty_512dims_1e-3"

OUTPUT_PATH = os.path.join("./KT_output_models", MODEL_NAME)

if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)


# encoding to index for deep model input
QUESTIONS2IDX = pickle.load(open("/share/kangyu/nips/KT_training/data/question2idx.pkl", 'rb'))
SUBJECTS2IDX = pickle.load(open("/share/kangyu/nips/KT_training/data/subject2idx.pkl", 'rb'))
GROUPS2IDX = pickle.load(open("/share/kangyu/nips/KT_training/data/group2idx.pkl", 'rb'))
QUIZS2IDX = pickle.load(open("/share/kangyu/nips/KT_training/data/quiz2idx.pkl", 'rb'))

# Average accuracy over all students for each question.
QUESTIONS_IDX2PERFORMANCE = pickle.load(open("/share/kangyu/nips/KT_training/data/question_idx2performance.pkl", 'rb'))

STU_NUMS = 118972

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore
   
    
set_seed(42)

##### MODEL

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)



class AttentionHead(nn.Module):
    def __init__(self, input_size, att_size):
        super().__init__()
        self.hatt_u = nn.Linear(input_size, att_size)
        self.hatt_w = nn.Linear(att_size, 1, bias=False)

        self.softmax = nn.Softmax(dim=2)
        self.tanh = nn.Tanh()

    def forward(self, input):
        u = self.hatt_u(input) # [batch, max_len, embed_nums, att_size]
        u = self.tanh(u)

        w = self.hatt_w(u)  # [batch, max_len, embed_nums, 1]
        a = self.softmax(w)  # [batch, max_len, embed_nums, 1]

        out = torch.matmul(a.transpose(2, 3), input).squeeze(dim=2)  # [batch, max_len, input_size]

        return out



class KTModel(nn.Module):
    def __init__(self, embed_dim=512, drop_rate=0.0):
        super(KTModel, self).__init__()

        self.user_embed = nn.Embedding(STU_NUMS, embed_dim)
        self.quiz_embed = nn.Embedding(len(QUIZS2IDX)+1, embed_dim)
        self.group_embed = nn.Embedding(len(GROUPS2IDX) + 1, embed_dim)

        self.subject_embed = nn.Embedding(len(SUBJECTS2IDX)+1, embed_dim)
        self.question_embed = nn.Embedding(len(QUESTIONS2IDX) + 1, embed_dim)
        self.position_embed = nn.Embedding(MAX_LEN, embed_dim)

        self.difficulty_embed = nn.Embedding(11, embed_dim)

        self.correctness_embed = nn.Embedding(3, embed_dim)
        self.choice_embed = nn.Embedding(5, embed_dim)

        self.attention = AttentionHead(embed_dim, embed_dim)

        self.layer_norm1 = nn.LayerNorm(embed_dim)

        self.mh_att1 = MultiHeadedAttention(h=8, d_model=embed_dim, dropout=drop_rate)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

        self.mh_att2 = MultiHeadedAttention(h=8, d_model=embed_dim, dropout=drop_rate)
        self.layer_norm3 = nn.LayerNorm(embed_dim)

        self.mh_att3 = MultiHeadedAttention(h=8, d_model=embed_dim, dropout=drop_rate)
        self.layer_norm4 = nn.LayerNorm(embed_dim)

        self.mh_att4 = MultiHeadedAttention(h=8, d_model=embed_dim, dropout=drop_rate)
        self.layer_norm5 = nn.LayerNorm(embed_dim)

        self.l0 = nn.Linear(embed_dim, 6)
        torch.nn.init.normal_(self.l0.weight, std=0.02)

        self.dropout = nn.Dropout(drop_rate)

    def forward(self, users, quizs, groups, difficulties,
                student_choices, correct_choices,
                correctness, att_mask, questions, positions,
                subject1, subject2, subject3, subject4, subject5, subject6):

        user_embedding = self.user_embed(users)
        quiz_embedding = self.quiz_embed(quizs)
        group_embedding = self.group_embed(groups)

        difficulty_embedding = self.difficulty_embed(difficulties)

        student_choice_embedding = self.choice_embed(student_choices)
        correct_choice_embedding = self.choice_embed(correct_choices)

        correctness_embedding = self.correctness_embed(correctness)
        question_embedding = self.question_embed(questions)
        position_embedding = self.position_embed(positions)

        subject1_embedding = self.subject_embed(subject1)
        subject2_embedding = self.subject_embed(subject2)
        subject3_embedding = self.subject_embed(subject3)
        subject4_embedding = self.subject_embed(subject4)
        subject5_embedding = self.subject_embed(subject5)
        subject6_embedding = self.subject_embed(subject6)

        attention_output = self.attention(
            torch.stack(
                (question_embedding, position_embedding,
                 quiz_embedding, group_embedding, difficulty_embedding,
                 subject1_embedding, subject2_embedding, subject3_embedding,
                 subject4_embedding, subject5_embedding, subject6_embedding),
                dim=2)
        )

        kv = correctness_embedding + student_choice_embedding + correct_choice_embedding + attention_output
        kv = self.layer_norm1(kv)

        q = correct_choice_embedding + attention_output
        q = self.layer_norm1(q)

        if self.training:
            kv = torch.mean(
                torch.stack(
                    [self.dropout(kv) for _ in range(10)],
                    dim=0,
                ),
                dim=0,
            )
            q = torch.mean(
                torch.stack(
                    [self.dropout(q) for _ in range(10)],
                    dim=0,
                ),
                dim=0,
            )

        att_mask = att_mask.unsqueeze(dim=1)  #[batch, 1, 100]

        mh_att_out1 = self.mh_att1(query=q, key=kv, value=kv, mask=att_mask)
        kv = kv + mh_att_out1
        kv = self.layer_norm2(kv)

        mh_att_out2 = self.mh_att2(query=q, key=kv, value=kv, mask=att_mask)
        kv = kv + mh_att_out2
        kv = self.layer_norm3(kv)

        mh_att_out3 = self.mh_att3(query=q, key=kv, value=kv, mask=att_mask)
        kv = kv + mh_att_out3
        kv = self.layer_norm4(kv)

        mh_att_out4 = self.mh_att4(query=q, key=kv, value=kv, mask=att_mask)
        kv = kv + mh_att_out4 + user_embedding
        kv = self.layer_norm5(kv)

        if self.training:
            logits = torch.mean(
                torch.stack(
                    [self.l0(self.dropout(kv)) for _ in range(10)],
                    dim=0,
                ),
                dim=0,
            )
        else:
            logits = self.l0(kv)

        return logits[:, :, :2], logits[:, :, 2:]



######### DATESET
class KT_TrainDataset(data.Dataset):
    def __init__(self, file_list, fold):
        
        self.file_list = file_list
        self.fold = fold

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        
        file = self.file_list[idx]
        path = os.path.join("/share/kangyu/nips/KT_training/data/all_features_all_sequence/train_npys", file)
        one = np.load(path)  #[features_nums, time_length]
        
        kfolds = one[0].astype(np.int)
        one = one[:, kfolds != -1]  #去除test部分
        
        kfolds = one[0].astype(np.int)
        one = one[:, kfolds != self.fold] #去除dev部分
        
        dates = one[14].tolist()
        
        start_index = [0]  #记录每个相隔不超过150天的序列的起始点
        previous = dates[0]
        for i in range(1, len(dates)):
            d = dates[i]

            year, month, day = d.split("-")
            year, month, day = int(year), int(month), int(day)
            d2 = datetime.datetime(year, month, day)

            year, month, day = previous.split("-")
            year, month, day = int(year), int(month), int(day)
            d1 = datetime.datetime(year, month, day)

            gap = (d2-d1).days

            if gap > 150:
                start_index.append(i)
                previous = d

        which_start = random.choice(range(len(start_index)))  #随机选取一段序列
        if which_start + 1 < len(start_index):
            one = one[:, start_index[which_start] : start_index[which_start+1]]
        else:
            one = one[:, start_index[which_start] : ]
        
        assert len(one) == 15
        
        length = one.shape[-1]
        
        if length > MAX_LEN:
            one = one[:, sorted(random.sample(range(length), MAX_LEN))] # 记得sort，防止顺序被打乱。
            
        length = one.shape[-1]
        
        questions_idx = one[1].astype(np.int).tolist()
        
        difficulties = []
        for q in questions_idx:
            performance = sum(QUESTIONS_IDX2PERFORMANCE[q]) / len(QUESTIONS_IDX2PERFORMANCE[q])
            performance = int(performance // 0.1)
            difficulties.append(performance)
                    
                
        quizs = one[2].astype(np.int).tolist()
        groups = one[3].astype(np.int).tolist()
        users = one[4].astype(np.int).tolist()
        
        subject1, subject2, subject3, subject4, subject5, subject6 = one[5:11].astype(np.int).tolist()
        
        is_correct = one[11].astype(np.float).tolist()     #要预测的，test没有。
        choices = (one[12].astype(np.float) - 1).tolist()  #要预测的，test没有。
        
        correct_choices = (one[13].astype(np.float) - 1).tolist()  #题目固有信息，test也能获得，可以直接传入。
        
        dates = one[14].tolist()
        
        loss_mask = 1 - np.random.binomial(1, 0.3, length) # 要算loss的是0，不算loss的是1。有15%的0。
        att_mask = loss_mask.tolist() # attention要注意的是1，不注意的是0。
        
        #建模用
        correctness = np.array(is_correct)  #用来建模, 要预测的部分的label要扔掉。
        correctness[loss_mask==0] = 2
        correctness = correctness.tolist()
        
        student_choices = np.array(choices)
        student_choices[loss_mask==0] = 4
        student_choices = student_choices.tolist()
                
        #算loss用
        labels = np.array(is_correct)  #用来算loss, 不预测的部分的loss不算。
        labels[loss_mask==1] = 2
        labels = labels.tolist()
        
        choice_labels = np.array(choices)
        choice_labels[loss_mask==1] = 4
        choice_labels = choice_labels.tolist()
        
        
        loss_mask = loss_mask.tolist()
        
        positions = [0]
        n = 0
        previous = dates[0]
        for d in dates[1:]:
            if d != previous:
                n += 1
                positions.append(n)
            else:
                positions.append(n)
            previous = d
        
        padding_length = MAX_LEN - length
        
        assert padding_length >= 0
        
        for _ in range(padding_length):
            loss_mask.append(1)
            att_mask.append(0)
            
            correctness.append(2)
            labels.append(2)
            
            student_choices.append(4)
            choice_labels.append(4)
            
            quizs.append(0)
            groups.append(0)
            users.append(118971)
            questions_idx.append(0)
            
            correct_choices.append(4)
            
            subject1.append(0)
            subject2.append(0)
            subject3.append(0)
            subject4.append(0)
            subject5.append(0)
            subject6.append(0)
            
            positions.append(0)
            difficulties.append(0)
        
        return {
                "users": torch.tensor(users, dtype=torch.long),
            
                "quizs": torch.tensor(quizs, dtype=torch.long),
                "groups": torch.tensor(groups, dtype=torch.long),
            
                "labels": torch.tensor(labels, dtype=torch.long),
                "correctness": torch.tensor(correctness, dtype=torch.long),
            
                "choice_labels": torch.tensor(choice_labels, dtype=torch.long),
                "student_choices": torch.tensor(student_choices, dtype=torch.long),

                "loss_mask": torch.tensor(loss_mask, dtype=torch.float),
                "att_mask": torch.tensor(att_mask, dtype=torch.float),
                
                "questions": torch.tensor(questions_idx, dtype=torch.long),
                "correct_choices": torch.tensor(correct_choices, dtype=torch.long),
            
                "subject1": torch.tensor(subject1, dtype=torch.long),
                "subject2": torch.tensor(subject2, dtype=torch.long),
                "subject3": torch.tensor(subject3, dtype=torch.long),
                "subject4": torch.tensor(subject4, dtype=torch.long),
                "subject5": torch.tensor(subject5, dtype=torch.long),
                "subject6": torch.tensor(subject6, dtype=torch.long),
            
                "positions": torch.tensor(positions, dtype=torch.long),
            
                "difficulties": torch.tensor(difficulties, dtype=torch.long),
            
               }
        

class KT_DevDataset(data.Dataset):
    def __init__(self, file_list, fold):
        
        self.file_list = file_list
        self.fold = fold

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        
        file = self.file_list[idx]
        path = os.path.join("/share/kangyu/nips/KT_training/data/all_features_all_sequence/dev_npys", file)
        one = np.load(path)
        
        kfolds = one[0].astype(np.int)
        one = one[:, kfolds != -1]  #去除test部分
        
        assert len(one) == 15
        
        length = one.shape[-1]
        
        if length > MAX_LEN:
            one = one[:, :MAX_LEN]
            
        length = one.shape[-1]
        
        kfolds = one[0].astype(np.int)
        questions_idx = one[1].astype(np.int).tolist()
        
        
        difficulties = []
        for q in questions_idx:
            performance = sum(QUESTIONS_IDX2PERFORMANCE[q]) / len(QUESTIONS_IDX2PERFORMANCE[q])
            performance = int(performance // 0.1)
            difficulties.append(performance)
                    
                      
        quizs = one[2].astype(np.int).tolist()
        groups = one[3].astype(np.int).tolist()
        users = one[4].astype(np.int).tolist()
        
        subject1, subject2, subject3, subject4, subject5, subject6 = one[5:11].astype(np.int).tolist()
        
        is_correct = one[11].astype(np.float).tolist()     #要预测的，test没有。
        choices = (one[12].astype(np.float) - 1).tolist()  #要预测的，test没有。
        
        correct_choices = (one[13].astype(np.float) - 1).tolist()  #题目固有信息，test也能获得，可以直接传入。
        
        dates = one[14].tolist()
        
        loss_mask = (kfolds!=self.fold).astype(np.int) # 要算loss的是0，不算loss的是1。dev是0，train是1
        att_mask = loss_mask.tolist() # attention要注意的是1，不注意的是0。
        
        #建模用
        correctness = np.array(is_correct)  #用来建模, dev部分的label要扔掉。
        correctness[loss_mask==0] = 2
        correctness = correctness.tolist()
        
        student_choices = np.array(choices)
        student_choices[loss_mask==0] = 4
        student_choices = student_choices.tolist()
                
        
        #算loss用
        labels = np.array(is_correct)  #用来算loss, train的部分的loss不算。
        labels[loss_mask==1] = 2
        labels = labels.tolist()
        
        choice_labels = np.array(choices)
        choice_labels[loss_mask==1] = 4
        choice_labels = choice_labels.tolist()
        
        
        loss_mask = loss_mask.tolist()
        
        positions = [0]
        n = 0
        previous = dates[0]
        for d in dates[1:]:
            if d != previous:
                n += 1
                positions.append(n)
            else:
                positions.append(n)
            previous = d
        
        padding_length = MAX_LEN - length
        
        assert padding_length >= 0
        
        for _ in range(padding_length):
            loss_mask.append(1)
            att_mask.append(0)
            
            correctness.append(2)
            labels.append(2)
            
            student_choices.append(4)
            choice_labels.append(4)
            
            quizs.append(0)
            groups.append(0)
            users.append(118971)
            questions_idx.append(0)
            
            correct_choices.append(4)
            
            subject1.append(0)
            subject2.append(0)
            subject3.append(0)
            subject4.append(0)
            subject5.append(0)
            subject6.append(0)
            
            positions.append(0)
            difficulties.append(0)
        
        return {
                "users": torch.tensor(users, dtype=torch.long),
            
                "quizs": torch.tensor(quizs, dtype=torch.long),
                "groups": torch.tensor(groups, dtype=torch.long),
            
                "labels": torch.tensor(labels, dtype=torch.long),
                "correctness": torch.tensor(correctness, dtype=torch.long),
            
                "choice_labels": torch.tensor(choice_labels, dtype=torch.long),
                "student_choices": torch.tensor(student_choices, dtype=torch.long),

                "loss_mask": torch.tensor(loss_mask, dtype=torch.float),
                "att_mask": torch.tensor(att_mask, dtype=torch.float),
                
                "questions": torch.tensor(questions_idx, dtype=torch.long),
                "correct_choices": torch.tensor(correct_choices, dtype=torch.long),
            
                "subject1": torch.tensor(subject1, dtype=torch.long),
                "subject2": torch.tensor(subject2, dtype=torch.long),
                "subject3": torch.tensor(subject3, dtype=torch.long),
                "subject4": torch.tensor(subject4, dtype=torch.long),
                "subject5": torch.tensor(subject5, dtype=torch.long),
                "subject6": torch.tensor(subject6, dtype=torch.long),
            
                "positions": torch.tensor(positions, dtype=torch.long),
            
                "difficulties": torch.tensor(difficulties, dtype=torch.long),
            
               }


###### loss_fn

def correct_loss_fn(logits, labels):
    loss_fct = nn.CrossEntropyLoss(ignore_index=2)
    
    loss = loss_fct(logits, labels)
    return loss

def choice_loss_fn(logits, labels):
    loss_fct = nn.CrossEntropyLoss(ignore_index=4)

    loss = loss_fct(logits, labels)
    return loss


def calc_metric(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1)
    recall = recall_score(y_true, y_pred, pos_label=1)
    f1 = f1_score(y_true, y_pred, pos_label=1)
    return acc, prec, recall, f1
    

######## train and eval

def train_fn(data_loader, model, optimizer, device, scheduler=None, epoch=0):
    model.train()
    
    losses = utils.AverageMeter()
    correct_losses = utils.AverageMeter()
    choice_losses = utils.AverageMeter()
    
    correct_accs = utils.AverageMeter()
    choice_accs = utils.AverageMeter()

    tk0 = tqdm(data_loader, total=len(data_loader))
    
    for bi, d in enumerate(tk0):
        
        users = d["users"].to(device, dtype=torch.long)
        
        quizs = d["quizs"].to(device, dtype=torch.long)
        groups = d["groups"].to(device, dtype=torch.long)
        
        difficulties = d["difficulties"].to(device, dtype=torch.long)
        
        labels = d["labels"].to(device, dtype=torch.long)
        correctness = d["correctness"].to(device, dtype=torch.long)
        
        choice_labels = d["choice_labels"].to(device, dtype=torch.long)
        student_choices = d["student_choices"].to(device, dtype=torch.long)
        
        loss_mask = d["loss_mask"].to(device, dtype=torch.float)
        att_mask = d["att_mask"].to(device, dtype=torch.float)
        
        questions = d["questions"].to(device, dtype=torch.long)
        correct_choices = d["correct_choices"].to(device, dtype=torch.long)
        
        positions = d["positions"].to(device, dtype=torch.long)
        
        subject1 = d["subject1"].to(device, dtype=torch.long)
        subject2 = d["subject2"].to(device, dtype=torch.long)
        subject3 = d["subject3"].to(device, dtype=torch.long)
        subject4 = d["subject4"].to(device, dtype=torch.long)
        subject5 = d["subject5"].to(device, dtype=torch.long)
        subject6 = d["subject6"].to(device, dtype=torch.long)

        optimizer.zero_grad()
        
        correct_logits, choice_logits = model(
            users=users, quizs=quizs, groups=groups, difficulties=difficulties, 
            student_choices=student_choices, correct_choices=correct_choices, 
            correctness=correctness, att_mask=att_mask, questions=questions, positions=positions, 
            subject1=subject1, subject2=subject2, subject3=subject3, 
            subject4=subject4, subject5=subject5, subject6=subject6, 
        )
        
        correct_logits = correct_logits.reshape(-1, 2)  #[batch * len, 2]
        choice_logits = choice_logits.reshape(-1, 4)  #[batch * len, 4]
        
        labels = labels.reshape(-1)  #[batch * len]
        choice_labels = choice_labels.reshape(-1)
        
        loss_mask = loss_mask.reshape(-1)  #[batch * len]
        
        
        correct_logits = correct_logits[loss_mask==0, :]
        choice_logits = choice_logits[loss_mask==0, :]
        
        labels = labels[loss_mask==0]
        choice_labels = choice_labels[loss_mask==0]
        
        
        correct_loss = correct_loss_fn(correct_logits, labels)
        choice_loss = choice_loss_fn(choice_logits, choice_labels)
        
        loss = correct_loss + choice_loss
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step()
        
        
        probs = torch.softmax(correct_logits, dim=1).cpu().detach().numpy()
        y_hat = np.argmax(probs, axis=1)
        
        acc = accuracy_score(labels.cpu().detach().numpy(), y_hat)
        
        correct_accs.update(acc, labels.size(0))
        
        
        probs = torch.softmax(choice_logits, dim=1).cpu().detach().numpy()
        y_hat = np.argmax(probs, axis=1)
        
        acc = accuracy_score(choice_labels.cpu().detach().numpy(), y_hat)
        
        choice_accs.update(acc, labels.size(0))
        
        
        losses.update(loss.item(), labels.size(0))
        correct_losses.update(correct_loss.item(), labels.size(0))
        choice_losses.update(choice_loss.item(), labels.size(0))
        
        
        tk0.set_postfix(loss=losses.avg, correct_loss=correct_losses.avg, choice_loss=choice_losses.avg, 
                        correct_acc=correct_accs.avg, choice_acc=choice_accs.avg)

        
 def eval_fn(data_loader, model, device):
    model.eval()
    
    losses = utils.AverageMeter()
    correct_losses = utils.AverageMeter()
    choice_losses = utils.AverageMeter()
    
    all_correct_logits = []
    all_correct_labels = []
    
    all_choice_logits = []
    all_choice_labels = []
 
    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader))
        for bi, d in enumerate(tk0):
            
            users = d["users"].to(device, dtype=torch.long)
            
            quizs = d["quizs"].to(device, dtype=torch.long)
            groups = d["groups"].to(device, dtype=torch.long)
            
            difficulties = d["difficulties"].to(device, dtype=torch.long)

            labels = d["labels"].to(device, dtype=torch.long)
            correctness = d["correctness"].to(device, dtype=torch.long)
            
            choice_labels = d["choice_labels"].to(device, dtype=torch.long)
            student_choices = d["student_choices"].to(device, dtype=torch.long)
        
            loss_mask = d["loss_mask"].to(device, dtype=torch.float)
            att_mask = d["att_mask"].to(device, dtype=torch.float)
            
            questions = d["questions"].to(device, dtype=torch.long)
            correct_choices = d["correct_choices"].to(device, dtype=torch.long)
            
            positions = d["positions"].to(device, dtype=torch.long)

            subject1 = d["subject1"].to(device, dtype=torch.long)
            subject2 = d["subject2"].to(device, dtype=torch.long)
            subject3 = d["subject3"].to(device, dtype=torch.long)
            subject4 = d["subject4"].to(device, dtype=torch.long)
            subject5 = d["subject5"].to(device, dtype=torch.long)
            subject6 = d["subject6"].to(device, dtype=torch.long)
            
            correct_logits, choice_logits = model(
                users=users, quizs=quizs, groups=groups, difficulties=difficulties, 
                student_choices=student_choices, correct_choices=correct_choices, 
                correctness=correctness, att_mask=att_mask, questions=questions, positions=positions, 
                subject1=subject1, subject2=subject2, subject3=subject3, 
                subject4=subject4, subject5=subject5, subject6=subject6, 
            )
            

            correct_logits = correct_logits.reshape(-1, 2)  #[batch * len, 2]
            choice_logits = choice_logits.reshape(-1, 4)  #[batch * len, 4]

            labels = labels.reshape(-1)  #[batch * len]
            choice_labels = choice_labels.reshape(-1)

            loss_mask = loss_mask.reshape(-1)  #[batch * len]


            correct_logits = correct_logits[loss_mask==0, :]
            choice_logits = choice_logits[loss_mask==0, :]

            labels = labels[loss_mask==0]
            choice_labels = choice_labels[loss_mask==0]
            
            
            correct_loss = correct_loss_fn(correct_logits, labels)
            choice_loss = choice_loss_fn(choice_logits, choice_labels)

            loss = correct_loss + choice_loss
            
            
            all_correct_logits.append(correct_logits)
            all_correct_labels.append(labels)
            
            all_choice_logits.append(choice_logits)
            all_choice_labels.append(choice_labels)
                        
            losses.update(loss.item(), labels.size(0))
            correct_losses.update(correct_loss.item(), labels.size(0))
            choice_losses.update(choice_loss.item(), labels.size(0))

            tk0.set_postfix(loss=losses.avg, correct_loss=correct_losses.avg, choice_loss=choice_losses.avg)
      
    
    all_correct_logits = torch.cat(all_correct_logits, dim=0)
    all_probs = torch.softmax(all_correct_logits, dim=1).cpu().detach().numpy()
    y_pred = np.argmax(all_probs, axis=1)
    
    all_correct_labels = torch.cat(all_correct_labels, dim=0).cpu().detach().numpy()
    
    acc_correct = accuracy_score(all_correct_labels, y_pred)
    auc = roc_auc_score(all_correct_labels, all_probs[:, 1])
    
    
    all_choice_logits = torch.cat(all_choice_logits, dim=0)
    all_probs = torch.softmax(all_choice_logits, dim=1).cpu().detach().numpy()
    y_pred = np.argmax(all_probs, axis=1)
    
    all_choice_labels = torch.cat(all_choice_labels, dim=0).cpu().detach().numpy()
    
    acc_choice = accuracy_score(all_choice_labels, y_pred)
    
    
    return acc_correct, auc, acc_choice



device = torch.device("cuda")

train_file_list = os.listdir("/share/kangyu/nips/KT_training/data/all_features_all_sequence/train_npys")

val_file_list = os.listdir("/share/kangyu/nips/KT_training/data/all_features_all_sequence/dev_npys")

# loaders
loaders = {
    "train": data.DataLoader(KT_TrainDataset(train_file_list, 0), 
                             batch_size=300, 
                             shuffle=True, 
                             num_workers=10, 
                             pin_memory=True, 
                             drop_last=True),
    
    "valid": data.DataLoader(KT_DevDataset(val_file_list, 0), 
                             batch_size=300, 
                             shuffle=False,
                             num_workers=10,
                             pin_memory=True,
                             drop_last=False)
}

# model

model = KTModel(drop_rate=0.2)

model.to(device)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

es = utils.EarlyStopping(patience=20, mode="max", delta=0.0005)
print(f"Training is Starting...")


max_acc_correct = -1
max_auc = -1
max_acc_choice = -1

for epoch in range(EPOCHS):
    train_fn(loaders["train"], model, optimizer, device, scheduler=scheduler, epoch=epoch)

    acc_correct, auc, acc_choice = eval_fn(loaders["valid"], model, device)

    if acc_correct > max_acc_correct:
        max_acc_correct = acc_correct
        max_auc = auc
        max_acc_choice = acc_choice

    print(f"Test Correct ACC Score = {acc_correct}")
    print(f"Test Correct AUC Score = {auc}")
    print(f"Test Choice ACC Score = {acc_choice}")
    

    es(acc_correct, model, model_path=os.path.join(OUTPUT_PATH, f"model.bin"))

    if es.early_stop:
        print("Early stopping")
        break


print("Best Correct Acc Score = {}".format(max_acc_correct))
print("Best AUC Score = {}".format(max_auc))
print("Best Choice Acc Score = {}".format(max_acc_choice))
