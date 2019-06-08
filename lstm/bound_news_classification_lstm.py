#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 21:42:28 2019

@author: root
"""

from lstm import My_lstm, cut_lstm
from get_max_eps import getUntargetedMaximumEps
import NewsTitleClassification.train_model as train_model

import torch
from torchtext import data
from torchtext import datasets
import torch.nn  as nn
import torch.optim as optim
import torch.nn.functional as F

import os
import argparse

def get_kth_step_bound(lstm,W,b, p, x, k, true_label, save_dir, 
                       a0=None, c0=None):
    # compute certified lower bound for the k-th time step in the input sequence x 
    # k could range from 0 to seq_len-1
    lstm.a0 = a0
    lstm.c0 = c0
    
    seq_len = x.shape[1]
    eps_idx = torch.zeros(seq_len, device=x.device)
    eps_idx[k] = 1
    lstm, x_new, eps_idx_new = cut_lstm(lstm, x, eps_idx) 
    #this function will reset lstm.a0 lstm.c0
    head_info = 'bounding %d-th frame in the input sequence \n' % (k+1)
    l_eps, u_eps = getUntargetedMaximumEps(lstm, W,b, x_new, p,true_label, save_dir, 
                  eps0=0.5, max_iter=10, acc=0.01, eps_idx=eps_idx_new, head_info = head_info)
    
    lstm.a0 = a0
    lstm.c0 = c0
    lstm.reset_seq_len(seq_len)
    return l_eps, u_eps

def get_each_step_bound(lstm,W,b, p, x, true_label, save_dir, 
                       a0=None, c0=None):
    # compute certified lower bound for every individual time step in the input sequence x
    # x is of shape (N, seq_len, in_features)
    eps = torch.zeros(x.shape[0], x.shape[1], device = x.device) #(N, seq_len)
    seq_len = x.shape[1]
    for k in range(seq_len-1, -1, -1):
        print('bounding %d-th frame in the input sequence' % (k+1))
    # for k in range(seq_len):
        #k from seq_len-1 to 0
        l_eps, u_eps = get_kth_step_bound(lstm,W,b, p, x, k, true_label, 
                                          save_dir, a0=a0, c0=c0)
        eps[:,k] = l_eps
        torch.save({'eps':eps, 'x':x, 'true_label':true_label, 'p':p, 'k':k},
                   save_dir + 'bound_eps')       
    return eps

category_list = ["sport", "world", "us", "business", "health", 
                 "entertainment", "sci_tech"]

def find_shortest_batch(model, iterator, TEXT, LABEL, minimum_length=10):
    model.eval()
    with torch.no_grad():
        short = torch.zeros(1, 1000)
        true_label = 0
        for batch_idx, batch in enumerate(iterator):
            text = torch.transpose(batch.Title, 0, 1) #(batch, seq_len)
            if text.shape[1]<short.shape[1] and text.shape[1]>minimum_length:
                short = text
                true_label = batch.Category.long()
                
        out = model(short)
        pred = out.argmax(dim=1)
        idx = (pred == true_label)
        
        short = short[idx,:]
        true_label = true_label[idx]
        print('The resulting text is of shape', short.shape)
        words = []
        seq_len = short.shape[1]
        for i in range(short.shape[0]):
            words.append([TEXT.vocab.itos[short[i,j]] for j in range(seq_len)])
        text_labels = [category_list[true_label[i]] for i in range(true_label.shape[0])]
        # text_labels = 1
    return short, true_label, words, text_labels

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description = 'Compute Certified Bound for LSTMs')

    parser.add_argument('--hidden-size', default = 32, type = int, metavar = 'HS',
                        help = 'hidden layer size (default: 32)')
    parser.add_argument('--work-dir', default = '../models/news_title_classifier/lstm_hidden_size_32/', type = str, metavar = 'WD',
                        help = 'the directory where the pretrained model is stored and the place to save the computed result')
    parser.add_argument('--model-name', default = 'lstm', type = str, metavar = 'MN',
                        help = 'the name of the pretrained model (default: lstm)')
    parser.add_argument('--use-constant', action='store_true',
                        help='whether to use constants to bound 2D nonlinear activations (default: False)')
    parser.add_argument('--use-1D-line', action='store_true',
                        help='whether to 1D lines to bound 2D nonlinear activations (default: False)')
    parser.add_argument('--N', default = 128, type = int,
                        help = 'number of samples to compute bounds for (default: 128)')
    parser.add_argument('--p', default = 2, type = int,
                        help = 'p norm, if p > 100, we will deem p = infinity (default: 2)')
    args = parser.parse_args()


    #load data
    TEXT1 = data.Field(tokenize='spacy', sequential=True)
    LABEL1 = data.Field(sequential=False, use_vocab=False, dtype=torch.float)
    TEXT2 = data.Field(tokenize='spacy', sequential=True)
    LABEL2 = data.Field(sequential=False, use_vocab=False, dtype=torch.float)
    train_dataset, test_dataset = data.TabularDataset.splits(path= 'NewsTitleClassification/', train='train_data_pytorch.csv',
                                               test='test_data_pytorch.csv',
                                               format='csv', skip_header=False,
                                        fields=[('Num', LABEL1), ('Title', TEXT1), 
                                        ('Description', TEXT2),('Category', LABEL2)])
    TEXT1.build_vocab(train_dataset, max_size=10000)#, vectors="glove.6B.100d")
    # LABEL1.build_vocab(dataset)
    TEXT2.build_vocab(train_dataset, max_size=10000)#, vectors="glove.6B.100d")

    device = torch.device('cpu')
    
    train_iterator, test_iterator = data.BucketIterator.splits(
            (train_dataset, test_dataset), 
    batch_size=args.N, shuffle=True, device=device, sort_key=lambda x:1)
    print('Finished Loading News Title Data')


    #load model
    INPUT_DIM = len(TEXT1.vocab)
    EMBEDDING_DIM = 100
    HIDDEN_DIM = args.hidden_size
    OUTPUT_DIM = 7#len(LABEL2.vocab)
    DROPOUT = 0
    
    model = train_model.RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, DROPOUT)
    
    model.load_state_dict(torch.load(args.work_dir + args.model_name))
    model = model.to(device)
    loss, acc = train_model.evaluate(model, test_iterator)
    print('model acc:', acc)
    
    
    #prepare input data
    text, true_label, words, text_labels = find_shortest_batch(model, 
                                                train_iterator, TEXT1, LABEL2,
                                                minimum_length=0)
    # inputs = torch.load('models/model5/certified_bound/input')
    # text = inputs['text']
    # true_label = inputs['true_label']
    # words = inputs['words']
    # text_labels = inputs['text_labels']
    
    
    x = model.embedding(text)
    seq_len = x.shape[1]
    W = model.fc.weight
    b = model.fc.bias
    
    
    p = args.p
    if p>100:
        p = float('inf')
    lstm = My_lstm(model.rnn, device, W,b,seq_len)

    # choose bounding techniques
    if args.use_constant:
        lstm.use_constant = True
        print('use constants to bound the 2D nonlinear activations')
    elif args.use_1D_line:
        lstm.use_1D_line = True
        print('use 1D lines to bound the 2D nonlinear activations')

    lstm.print_info = False
    save_dir = args.work_dir + '%s_norm/' % str(p)
    os.makedirs(save_dir, exist_ok=True)
    torch.save({'text':text, 'true_label':true_label, 'words':words,
                'text_labels':text_labels, 'x':x}, save_dir + 'input')
    print('Saved input data to ' + save_dir + 'input')

    eps = get_each_step_bound(lstm,W,b, p, x, true_label, save_dir, 
                            a0=None, c0=None)
    print('The lower bound we found is:')
    print(eps)
    
    
    
    
    
    
    
    
    
    
    
    