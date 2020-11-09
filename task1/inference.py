import torch
import pickle
import jieba
import os
import numpy as np
from train import pad

def load_model(checkpoint, device, device_id):
    with open(checkpoint, 'rb') as f:
        torch.cuda.set_device(device_id)
        model = torch.load(f).to(device)
    model.eval()
    print('model loaded...')
    return model


def get_batch_test(q, c, c_masked, batch_size, batch_idx, device, mask_token=2, max_num_steps=100):
    N = len(q)
    st = int(batch_idx*batch_size)
    ed = min(st+batch_size, N)
    if(st>=N):
        print('Consumed all data, exit...')
        return None
    q_tensor = torch.tensor(q[st:ed]).to(device)
    c_tensor = torch.tensor(c[st:ed]).to(device)
    c_masked_tensor = torch.tensor(c_masked[st:ed]).to(device)
    return q_tensor, c_tensor, c_masked_tensor


def inference(model_save_path, usable_data_lst, batch_size, device_id, max_num_steps=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_save_path, device, device_id)

    data_size = len(usable_data_lst)
    q = [pad(x['questions'], max_num_steps) for x in usable_data_lst]
    c = [pad(x['corrects'], max_num_steps, padding=3) for x in usable_data_lst]
    c_masked = [pad(x['corrects_masked'], max_num_steps, padding=3) for x in usable_data_lst]

    prediction_lst, proba_lst = [], []
    
    num_steps = int(data_size//batch_size)
    with torch.no_grad():
        for step in range(num_steps+3):
            if(step%50==0):
                print('Doing {} steps out of {} steps'.format(step, num_steps))
            batch_data = get_batch_test(q, c, c_masked, batch_size, step, device)
            if(batch_data!=None):
                question_seq_batch, is_correct_seq_batch, is_correct_seq_masked_batch = batch_data
                probability, _ = model(question_seq_batch, is_correct_seq_batch, is_correct_seq_masked_batch)

                proba_lst += [float(x) for x in probability[:,1].cpu().numpy()]
                prediction_batch = torch.argmax(probability, dim=1).cpu().numpy()
                prediction_lst += [int(x) for x in prediction_batch]
            else:
                break

    masked_answer_seq = []
    for item in usable_data_lst:
        masked_answer_seq += item['masked_answerid']

    result = {}
    for answer_id, y_hat, y_hat_proba in zip(masked_answer_seq, prediction_lst, proba_lst):
        result[str(answer_id)] = (y_hat, y_hat_proba)
    return result



def make_prediction(submit, usable_test_data, model_save_path, result_save_root_path, batch_size, device_id):
    print('*'*88)
    if(not os.path.exists(result_save_root_path)):
        os.makedirs(result_save_root_path)

    print('reloading model from {}'.format(model_save_path))
    print('num test sequence:{}'.format(len(usable_test_data)))
    print('Start prediction task, please wait...')
    print('-'*88)

    result = inference(model_save_path, usable_test_data, batch_size, device_id)

    y_hat_lst, y_hat_proba_lst = [], []
    if(submit.shape[0]!=len(list(result.keys()))):
        print('*'*88)
        print('预测结果和实际输出结果数目对不上，请重视！！！')
        print('测试集数目:', submit.shape[0])
        print('实际输出数目:', len(list(result.keys())))
    else:
        print('待预测结果和实际输出结果数目相同，验证通过！！！')

    print('写提交文件中，请稍等....')
    for i in range(submit.shape[0]):
        answer_id = submit['AnswerId'].iloc[i]
        
        pred = result.get(str(answer_id))
        if(pred!=None):
            y_hat = pred[0]
            y_proba = float(pred[1])
        else:
            y_hat = 1
            y_proba = 0.5
        y_hat_lst.append(int(y_hat))
        y_hat_proba_lst.append(y_proba)
    submit['IsCorrect'] = y_hat_lst
    submit['probability'] = y_hat_proba_lst
    submit = submit[['QuestionId', 'UserId', 'AnswerId', 'IsCorrect', 'probability']]
    submit.to_csv(os.path.join(result_save_root_path, model_name.replace('.pt', '.csv')), index=False)
    print('完成！')


if __name__=='__main__':
    batch_size = 2000
    device_id = 0

    from train import load_json, save_json
    import pandas as pd
    submit = pd.read_csv('/workspace/Guowei/project/nips2020/data/submission/submission_task_1_2.csv')[['QuestionId', 'UserId', 'AnswerId']]

    usable_test_data_path = '/workspace/Guowei/project/nips2020/data/submission/submission_sequence.json'
    print('loading test sequence data, please wait...')
    usable_test_data = load_json(usable_test_data_path)
    model_root_path = '/workspace/Guowei/project/nips2020/model/MLM'
    
    version_lst = ['v03']
    model_lst = ['RNN_embdsz_200_hidden_200_lr_0.001_bs_512_nlayers_2_bidirectional_True.pt', \
                'RNN_embdsz_200_hidden_200_lr_0.01_bs_1024_nlayers_2_bidirectional_True.pt',\
                'RNN_embdsz_200_hidden_200_lr_0.001_bs_256_nlayers_4_bidirectional_True.pt',\
                'RNN_embdsz_200_hidden_200_lr_0.001_bs_1024_nlayers_4_bidirectional_True.pt',\
                'RNN_embdsz_200_hidden_200_lr_0.01_bs_1024_nlayers_4_bidirectional_True.pt',\
                'RNN_embdsz_200_hidden_200_lr_0.001_bs_1024_nlayers_2_bidirectional_True.pt']

    for version in version_lst:
        version_path = os.path.join(model_root_path, version)
        if(model_lst==None):
            model_lst = [x for x in os.listdir(version_path) if '.pt' in x]
    
        for model_name in model_lst:
            model_save_path = os.path.join(version_path, model_name)
            result_save_root_path = '/workspace/Guowei/project/nips2020/submission/{}'.format(version)

            make_prediction(submit, usable_test_data, model_save_path, result_save_root_path, batch_size, device_id)