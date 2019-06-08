#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 16:38:58 2019

@author: root
"""

import torch
from torchtext import data
from torchtext import datasets
import torch.nn  as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F

import os
import argparse

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, bidirectional=False, 
                               dropout=0, batch_first=True)
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        #x is of size [batch, seq_len]
        
        embedded = self.dropout(self.embedding(x))
        #embedded is of size  [batch, seq_len, embedding_dim]
        
        output, (hidden, cell) = self.rnn(embedded)
        # output is of size [batch, seq_len, hidden_dim * num directions], 
        # tensor containing the output features (h_t) from the last layer of the LSTM, for each t.
        # hidden is of size [num layers * num directions, batch, hidden_dim]
        # tensor containing the hidden state for t = seq_len
        # cell is of size [num layers * num directions, batch, hidden_dim]
        # tensor containing the cell state for t = seq_len
        
        hidden = self.dropout(hidden[0,:,:])
        #hidden is of size [batch size, hidden_dim]
            
        return self.fc(hidden)
    
def train(model, iterator, optimizer, scheduler):
    
    epoch_loss = 0
    epoch_correct = 0
    epoch_total_num = 0

    scheduler.step()
    model.train()
    
    for batch_idx, batch in enumerate(iterator):
        
        text = batch.Title
        # text = batch.Description
        text = torch.transpose(text, 0, 1)
        label = batch.Category.long()
        
        output = model(text)
        
        pred = output.argmax(dim=1)
        acc = (pred == label).float().mean()
        
        loss = F.cross_entropy(output, label, reduction='mean')
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print('loss: %.4f acc %.4f' % (loss, acc))
        epoch_loss += loss.item() * label.shape[0]
        epoch_correct += (pred == label).sum()
        epoch_total_num += label.shape[0]
        
    return epoch_loss / epoch_total_num, (epoch_correct.float() / epoch_total_num)

def evaluate(model, iterator):
    
    epoch_loss = 0
    epoch_correct = 0
    epoch_total_num = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch_idx, batch in enumerate(iterator):
            label = batch.Category.long()
            # print(label)
            text = torch.transpose(batch.Title, 0, 1)
            # text = torch.transpose(batch.Description,0,1)
            output = model(text)
            
            pred = output.argmax(dim=1)
            acc = (pred == label).float().mean()
            loss = F.cross_entropy(output, label, reduction='mean')

            
            # print('loss: %.4f acc %.4f' % (loss, acc))
            epoch_loss += loss.item() * label.shape[0]
            epoch_correct += (pred == label).sum()
            epoch_total_num += label.shape[0]
        
    return epoch_loss / epoch_total_num, epoch_correct.float() / epoch_total_num

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Train LSTM News Title Classifier')

    parser.add_argument('--hidden-size', default = 32, type = int, metavar = 'HS',
                        help = 'hidden layer size (default: 32)')
    parser.add_argument('--save-dir', default = '../../models/news_title_classifier/', type = str, metavar = 'SD',
                        help = 'the directory to save the trained model')
    parser.add_argument('--cuda', action='store_true',
                        help='whether to allow gpu for training')
    args = parser.parse_args()
    
    #load data
    TEXT1 = data.Field(tokenize='spacy', sequential=True)
    LABEL1 = data.Field(sequential=False, use_vocab=False, dtype=torch.float)
    TEXT2 = data.Field(tokenize='spacy', sequential=True)
    LABEL2 = data.Field(sequential=False, use_vocab=False, dtype=torch.float)
    train_dataset, test_dataset = data.TabularDataset.splits(path= './', train='train_data_pytorch.csv',
                                               test='test_data_pytorch.csv',
                                               format='csv', skip_header=False,
                                        fields=[('Num', LABEL1), ('Title', TEXT1), 
                                                ('Description', TEXT2),('Category', LABEL2)])
    TEXT1.build_vocab(train_dataset, max_size=10000, vectors="glove.6B.100d")
    TEXT2.build_vocab(train_dataset, max_size=10000, vectors="glove.6B.100d")
    

    use_cuda = args.cuda and torch.cuda.is_available()
    print('use_cuda: ',use_cuda)
    device = torch.device("cuda" if use_cuda else "cpu")
    
    
    train_dataloader, test_dataloader = data.BucketIterator.splits(
            (train_dataset, test_dataset), 
            batch_size=128, shuffle=True, device=device, sort_key=lambda x:1)
    
    INPUT_DIM = len(TEXT2.vocab)
    EMBEDDING_DIM = 100
    HIDDEN_DIM = args.hidden_size
    OUTPUT_DIM = 7
    BIDIRECTIONAL = False
    DROPOUT = 0.5
    save_dir = args.save_dir + 'lstm_hidden_size_%s/' % str(HIDDEN_DIM)
    
    model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, DROPOUT)
    
    pretrained_embeddings = TEXT2.vocab.vectors
    model.embedding.weight.data.copy_(pretrained_embeddings)
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = lr_scheduler.StepLR(optimizer, step_size = 30, gamma = 0.1)


    for i in range(100):
        print('Epoch %d Train' % (i+1))
        train_loss, train_acc = train(model, train_dataloader, optimizer, scheduler)
        # print('Epoch %d evaluate' % (i+1))
        loss, acc = evaluate(model, test_dataloader)
        print('-----------------------------------------------')
        print('Epoch %d Testset loss: %.3f accuracy: %.2f' % (i+1, loss, acc*100))
        print('-----------------------------------------------')
    
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.cpu().state_dict(), save_dir + 'lstm')
    print('Have saved the trained model to ' + save_dir+'lstm')
    
    