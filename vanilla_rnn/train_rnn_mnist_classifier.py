#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 10:30:14 2018

@author: root
This files helps train a RNN
"""
import torch
import torch.nn as nn
import bound_vanilla_rnn as v_rnn
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn.functional as F
import argparse
import os

def train(log_interval, model, device, train_loader, optimizer, 
          epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        N = data.shape[0]
        data = data.view(N, model.time_step, -1)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.sampler),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            N = data.shape[0]
            data = data.view(N, model.time_step, -1)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.sampler),
        100. * correct / len(test_loader.sampler)))

def main(model,  savefile, cuda):
    
    batch_size = 64
    test_batch_size = 128
    epochs = 20
    log_interval = 10
    lr = 0.01
    momentum = 0.5
    shuffle_train = True
    shuffle_test = True
    
    
    use_cuda = cuda and torch.cuda.is_available()
    print('use_cuda: ',use_cuda)


    device = torch.device("cuda" if use_cuda else "cpu")
    model.to(device)
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ])),
        batch_size=batch_size, shuffle=shuffle_train, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data/mnist', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ])),
        batch_size=test_batch_size, shuffle=shuffle_test, **kwargs)


    
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    for epoch in range(1, epochs + 1):
        train(log_interval, model, device, train_loader, optimizer, 
              epoch)
        test(model, device, test_loader)
        
    torch.save(model.cpu().state_dict(), savefile)
    print("have saved the trained model to ",savefile)

    return 0



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Train Mnist RNN Classifier')

    parser.add_argument('--hidden-size', default = 64, type = int, metavar = 'HS',
                        help = 'hidden layer size (default: 64)')
    parser.add_argument('--time-step', default = 7, type = int, metavar = 'TS',
                        help = 'number of slices to cut the 28*28 image into, it should be a factor of 28 (default: 7)')
    parser.add_argument('--activation', default = 'tanh', type = str, metavar = 'a',
                        help = 'nonlinearity used in the RNN, can be either tanh or relu (default: tanh)')
    parser.add_argument('--save_dir', default = '../models/mnist_classifier/', type = str, metavar = 'SD',
                        help = 'the directory to save the trained model')
    parser.add_argument('--cuda', action='store_true',
                        help='whether to allow gpu for training')
    args = parser.parse_args()


    input_size = int(28*28 / args.time_step)
    hidden_size = args.hidden_size
    output_size = 10
    time_step = args.time_step 
    activation= args.activation
    
    rnn = v_rnn.RNN(input_size, hidden_size, output_size, time_step, activation)
    model_name = 'rnn_%s_%s_%s' % (str(time_step), str(hidden_size), activation)
    save_dir = args.save_dir + model_name
    os.makedirs(save_dir, exist_ok=True)
    main(rnn, save_dir+'/rnn', args.cuda)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    