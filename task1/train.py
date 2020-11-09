import os
import time
import json, pickle
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import copy
from model import BertMLM, RnnMLM
import random
import argparse
import warnings
warnings.filterwarnings("ignore")


def load_pickle(data_path):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data


def load_json(path):
    with open(path, 'r') as handle:
        data = json.load(handle)
    return data

def save_json(data, path):
    with open(path, 'w') as handle:
        json.dump(data, handle)


def apply_masking(is_correct_seq, chance, mask_token):
    '''
    args:
    is_correct_seq: 2d list, 做题信息正误矩阵，大小为[batch_size, num_steps], value in [0, 1]

    returns:
    is_correct_seq_masked : 被部分masked后的正误矩阵
    masked_labels：被masked掉的位置的原始正误标签，一维向量
    '''

    is_correct_seq_masked = []
    masked_labels = []
    for i in range(len(is_correct_seq)):
        tmp = []
        for j in range(len(is_correct_seq[i])):
            p = np.random.uniform()
            if(p<=chance):
                masked_labels.append(is_correct_seq[i][j])
                tmp.append(mask_token)
            else:
                tmp.append(is_correct_seq[i][j])
        is_correct_seq_masked.append(tmp)
    return is_correct_seq_masked, masked_labels


def pad(x, max_len, padding=0):
    if(len(x)<max_len):
        return x + [padding for _ in range(max_len-len(x))]
    else:
        return x[:max_len]


def preprocess_data(data, max_num_steps, mask_token=2):
    q = [x['questions'] for x in data]
    c = [x['corrects'] for x in data]

    mask_proba_lst = np.load('/workspace/Guowei/project/nips2020/data/mask概率.npy')
    masked_prob = random.choice(mask_proba_lst)
    c_masked, masked_labels = apply_masking(c, chance=masked_prob, mask_token=mask_token)

    q = [pad(x, max_num_steps) for x in q]
    c = [pad(x, max_num_steps, padding=3) for x in c] # 3是zero padding
    c_masked = [pad(x, max_num_steps, padding=3) for x in c_masked]

    return (q, c, c_masked, masked_labels)


def get_batch_train(data, batch_size, max_num_steps, device, masked_prob, mask_token=2):
    N = len(data)
    selected_idx = np.random.randint(low=0, high=N, size=batch_size)

    # 此时每个question序列还没有padding
    q = [data[x]['questions'] for x in selected_idx]
    c = [data[x]['corrects'] for x in selected_idx]

    
    c_masked, masked_labels = apply_masking(c, chance=masked_prob, mask_token=mask_token)

    q = [pad(x, max_num_steps) for x in q]
    c = [pad(x, max_num_steps, padding=3) for x in c]
    c_masked = [pad(x, max_num_steps, padding=3) for x in c_masked]

    q_tensor = torch.tensor(q).to(device)
    c_tensor = torch.tensor(c).to(device)
    c_masked_tensor = torch.tensor(c_masked).to(device)
    masked_labels_tensor = torch.tensor(masked_labels).to(device)
    return q_tensor, c_tensor, c_masked_tensor, masked_labels_tensor

def get_batch_valid(usable_data, batch_size, batch_idx, device, mask_token=2):
    N = len(usable_data[0])
    q, c, c_masked, masked_labels = usable_data
    st = int(batch_idx*batch_size)
    ed = min(st+batch_size, N)

    q_tensor = torch.tensor(q[st:ed]).to(device)
    c_tensor = torch.tensor(c[st:ed]).to(device)
    c_masked_tensor = torch.tensor(c_masked[st:ed]).to(device)

    mlm_mask = (c_masked_tensor==mask_token).long()
    masked_labels_tensor = c_tensor[mlm_mask==1]

    #masked_labels_tensor = torch.tensor(masked_labels[st:ed]).to(device)
    return q_tensor, c_tensor, c_masked_tensor, masked_labels_tensor

def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)
    
def train_model(model, learning_rate, train_data, batch_size, max_num_steps, device):
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.to(device)
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    train_size = len(train_data)
    num_steps = int(train_size//batch_size)
    model.train()
    num_corrects = 0
    num_masked_token = 0

    mask_proba_lst = np.load('/workspace/Guowei/project/nips2020/data/mask概率.npy')
    for step in range(num_steps):
        optim.zero_grad()
        # 读取一个batch的数据，y_truth是该batch数据中被mask掉的题目正误的序列，一维张量
        masked_prob = random.choice(mask_proba_lst)
        question_seq_batch, is_correct_seq_batch, is_correct_seq_masked_batch, y_truth = \
                 get_batch_train(train_data, batch_size, max_num_steps, device, masked_prob)

        probability, loss = model(question_seq_batch, is_correct_seq_batch, is_correct_seq_masked_batch)
        num_corrects_batch = (torch.argmax(probability, dim=1) == y_truth).float().sum()
        num_corrects += num_corrects_batch.item()
        num_masked_token += y_truth.shape[0]

        loss.backward()
        clip_gradient(model, 1e-1)
        optim.step()

        total_epoch_loss += loss.item()
    if(num_masked_token>0):
        acc = num_corrects / num_masked_token
    else:
        acc = -1
    return total_epoch_loss/num_steps, acc

    
def eval_model(model, usable_data, batch_size, device):
    # usable_data is a tuple = (q, c, c_masked, masked_labels)
    total_epoch_loss = 0
    total_epoch_acc = 0
    num_corrects = 0
    num_masked_token = 0
    model.eval()

    valid_size = len(usable_data[0])
    num_steps = int(valid_size//batch_size)
    with torch.no_grad():
        for step in range(num_steps):
            question_seq_batch, is_correct_seq_batch, is_correct_seq_masked_batch, y_truth = get_batch_valid(usable_data, batch_size, step, device)
            probability, loss = model(question_seq_batch, is_correct_seq_batch, is_correct_seq_masked_batch)
            
            num_corrects_batch = (torch.argmax(probability, dim=1) == y_truth).float().sum()
            num_corrects += num_corrects_batch.item()
            num_masked_token += y_truth.shape[0]
            total_epoch_loss += loss.item()
    if(num_masked_token>0):
        acc = num_corrects / num_masked_token
    else:
        acc = -1
    return total_epoch_loss/num_steps, acc



def run(model, learning_rate, train_data, usable_valid_data, batch_size, max_num_steps, \
                    device, max_iters, best_model_save_path, model_prefix, max_early_stop_counts=10):
    best_loss = 1e9
    best_model = model
    early_stop_count = 0
    best_model_epoch = 0

    try:
        for epoch in range(max_iters):
            if(early_stop_count> max_early_stop_counts):
                print('Early stop at epoch {}, best model at epoch {}.'.format(epoch, best_model_epoch), flush=True)
                break
            if(best_loss>0.6 and epoch>10):
                print('Early stop at epoch {}, best model at epoch {}.'.format(epoch, best_model_epoch), flush=True)
                break
            train_loss, train_acc = train_model(model, learning_rate, train_data, batch_size, max_num_steps, device)
            val_loss, val_acc = eval_model(model, usable_valid_data, batch_size, device)
            print('Epoch {}, train loss {}, train acc {}, val loss {}, val acc {}. best loss {}'.format(epoch+1, round(train_loss, 5), round(train_acc, 5), round(val_loss, 5), round(val_acc, 5), best_loss), flush=True)
            if(val_loss<best_loss):
                early_stop_count = 0
                best_loss = val_loss
                best_model = model
                best_model_epoch = epoch

                if(not os.path.exists(best_model_save_path)):
                    os.makedirs(best_model_save_path)
                with open(os.path.join(best_model_save_path, '{}.pt'.format(model_prefix)), 'wb') as f:
                    torch.save(best_model, f)
            else:
                early_stop_count += 1
        print('Training completed...', flush=True)
        print('Best validation loss is {}'.format(best_loss), flush=True)
        print('-'*66, flush=True)
    except Exception as e:
        print(e, flush=True)
    return best_model


def train_rnn_mlm(args, train_data_path, valid_data_path, best_model_save_path):
    num_question = args.num_question
    num_labels = args.num_labels
    max_num_steps = args.max_num_steps
    max_iters = args.max_iters
    device_id = args.device_id

    print('Training RNN MLM', flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device_id)
    train_data = load_json(train_data_path)
    usable_valid_data = load_pickle(valid_data_path)

    hidden_size_lst = [200, 300, 400]
    num_rnn_layers_lst = [2, 4]
    learning_rate_lst = [1e-2, 1e-3]
    batch_size_lst = [256, 512, 1024]
    bidirectional = True

    for hidden_size in hidden_size_lst:
        for num_rnn_layers in num_rnn_layers_lst:
            for learning_rate in learning_rate_lst:
                for batch_size in batch_size_lst:
                    try:
                        embed_size = hidden_size
                        model_prefix = 'RNN_embdsz_{}_hidden_{}_lr_{}_bs_{}_nlayers_{}_bidirectional_{}'.format(embed_size, hidden_size, \
                                            learning_rate, batch_size, num_rnn_layers, bidirectional)
                        print('model parames:{}'.format(model_prefix), flush=True)
                        
                        model = RnnMLM(embed_size, num_question, hidden_size, num_labels, num_rnn_layers, bidirectional)

                        best_model = run(model, learning_rate, train_data, usable_valid_data, batch_size, max_num_steps, \
                                        device, max_iters, best_model_save_path, model_prefix, max_early_stop_counts=10)
                    except Exception as e:
                            print('Failed to train model {}'.format(model_prefix), flush=True)
                            print(e, flush=True)


def train_bert_mlm(args, train_data_path, valid_data_path, best_model_save_path):
    num_question = args.num_question
    num_labels = args.num_labels
    max_num_steps = args.max_num_steps
    max_iters = args.max_iters
    device_id = args.device_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device_id)

    train_data = load_json(train_data_path)
    usable_valid_data = load_pickle(valid_data_path)

    hidden_size_lst = [200, 400, 600]
    learning_rate_lst = [1e-2, 1e-3, 1e-4]
    batch_size_lst = [512, 1025]
    n_layers_lst = [2, 3, 4, 6]
    nhead_lst = [4, 6, 8]

    for hidden_size in hidden_size_lst:
        for learning_rate in learning_rate_lst:
            for batch_size in batch_size_lst:
                for n_layers in n_layers_lst:
                    for nhead in nhead_lst:
                        intermediate_size = 2*hidden_size
                        try:
                            model_prefix = 'BERT_hidden_{}_lr_{}_bs_{}_nlayers_{}_nhead_{}_intermediateSize_{}'.format(hidden_size, \
                                                learning_rate, batch_size, n_layers, nhead, intermediate_size)
                            print('model parames:{}'.format(model_prefix), flush=True)
                            
                            model = BertMLM(num_question, hidden_size, num_labels, num_hidden_layers=n_layers, num_attention_heads=nhead, intermediate_size=intermediate_size)

                            best_model = run(model, learning_rate, train_data, usable_valid_data, batch_size, max_num_steps, \
                                            device, max_iters, best_model_save_path, model_prefix, max_early_stop_counts=10)
                        except Exception as e:
                            print('Failed to train model {}'.format(model_prefix), flush=True)
                            print(e, flush=True)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root_path", type=str, default='/workspace/Guowei/project/nips2020/data/mlm_data', help="where to read training/validation data")
    parser.add_argument("--model_save_path", type=str, default='/workspace/Guowei/project/nips2020/model/MLM', help="where save model")
    parser.add_argument("--version", type=str, default='v05', help="training data version")
    parser.add_argument("--device_id", type=int, default=0, help="GPU device id to run")
    parser.add_argument("--model", type=str, default='bert', help="must be bert or rnn")

    parser.add_argument("--num_question", type=int, default=27613)
    parser.add_argument("--num_labels", type=int, default=2)
    parser.add_argument("--max_num_steps", type=int, default=100)
    parser.add_argument("--max_iters", type=int, default=100)

    args = parser.parse_args()
    data_root_path = args.data_root_path
    model_save_path = args.model_save_path
    version = args.version
    which_model = args.model


    train_data_path = '{}/{}/train.json'.format(data_root_path, version)
    valid_data_path = '{}/{}/static_usable_valid.pkl'.format(data_root_path, version)
    best_model_save_path = '{}/{}'.format(model_save_path, version)
    
    if(which_model=='bert'):
        train_bert_mlm(args, train_data_path, valid_data_path, best_model_save_path)
    elif(which_model=='rnn'):
        train_rnn_mlm(args, train_data_path, valid_data_path, best_model_save_path)
    else:
        print('Invalid input for model, program ends....')