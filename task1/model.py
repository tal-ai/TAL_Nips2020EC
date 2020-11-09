import torch
import torch.nn as nn
import numpy as np
import copy
from transformers import BertTokenizer, BertForMaskedLM, BertConfig, BertForTokenClassification


class BertMLM(nn.Module):
    def __init__(self, num_question, hidden_size, num_labels, \
                     num_hidden_layers=12, num_attention_heads=12, \
                     intermediate_size=3072, config = BertConfig()):
        super(BertMLM, self).__init__()
        config.vocab_size = num_question
        config.hidden_size = hidden_size
        config.num_labels = num_labels
        config.num_hidden_layers = num_hidden_layers
        config.num_attention_heads = num_attention_heads
        config.intermediate_size = intermediate_size
        
        self.model = BertForTokenClassification(config)
        #self.model = BertForMaskedLM(config)
        self.question_embd_layer = nn.Embedding(num_question, hidden_size)
        self.is_correct_embd_layer = nn.Embedding(4, hidden_size) # 0 => wrong, 1 => correct, 2 => masked, 3 => padding
        self.loss_fn = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, question_seq, is_correct_seq, is_correct_seq_masked, mask_token=2):
        question_embd = self.question_embd_layer(question_seq)
        is_correct_embd = self.is_correct_embd_layer(is_correct_seq_masked)
        inputs_embds = question_embd + is_correct_embd # shape = [batch_size, num_steps, hidden_size]

        # attention_mask为1的地方参与attention计算，0的地方不参与
        attention_mask = (is_correct_seq_masked<=2).float() # shape=(bs, num_steps). 1代表没有被mask的地方，0是被masked的地方。padding的地方要mask为0

        mlm_mask = (is_correct_seq_masked==mask_token).long()

        output = self.model(
                            inputs_embeds=inputs_embds, 
                           attention_mask=attention_mask, 
                           return_dict=True,
                           output_hidden_states=True)

        logits = output.logits # shape = [batch_size, num_steps, num_labels]
        
        # 只把被mask的地方的logits抽出来
        masked_logits = logits[mlm_mask==1] # shape=[num_masks_in_data, num_labels]
        masked_labels = is_correct_seq[mlm_mask==1] # shape=[num_masks_in_data]
        probability = self.softmax(masked_logits)
        loss = self.loss_fn(masked_logits, masked_labels)
        return probability, loss


class RnnMLM(nn.Module):
    def __init__(self, embed_size, num_question, hidden_size, num_labels, num_rnn_layers, bidirectional=True):
        super(RnnMLM, self).__init__()        
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers=num_rnn_layers, batch_first=True, bidirectional=bidirectional)
        self.question_embd_layer = nn.Embedding(num_question, hidden_size)
        self.is_correct_embd_layer = nn.Embedding(4, embed_size) # 0 => wrong, 1 => correct, 2 => masked, 3 => padding
        if(bidirectional):
            print('双向RNN开启')
            self.linear_layer = nn.Linear(2*hidden_size, num_labels)
        else:
            print('单向RNN开启')
            self.linear_layer = nn.Linear(hidden_size, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, question_seq, is_correct_seq, is_correct_seq_masked, mask_token=2, batch_size=None):
        question_embd = self.question_embd_layer(question_seq)
        is_correct_embd = self.is_correct_embd_layer(is_correct_seq_masked)
        inputs_embds = question_embd + is_correct_embd # shape = [batch_size, num_steps, hidden_size]

        #output: shape=[batch_size, num_steps, num_directions * hidden_size]
        # tensor containing the output features (h_t) from the last layer of the LSTM, for each t.
        output, _ = self.rnn(inputs_embds)

        # shape = [batch_size, num_steps, num_labels]
        logits = self.linear_layer(output)
        
        # 只把被mask的地方的logits抽出来
        mlm_mask = (is_correct_seq_masked==mask_token).long()
        masked_logits = logits[mlm_mask==1] # shape=[num_masks_in_data, num_labels]
        masked_labels = is_correct_seq[mlm_mask==1] # shape=[num_masks_in_data]
        probability = self.softmax(masked_logits)
        loss = self.loss_fn(masked_logits, masked_labels)
        return probability, loss