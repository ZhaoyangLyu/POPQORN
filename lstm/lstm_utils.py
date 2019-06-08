#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 09:45:20 2019

@author: root
"""

import torch

def generateNoise(shape, p, eps, device=None, eps_idx=None):
    #eps idx could None or a tensor of size (seq_len)
    #it contains 0 or 1, which indicate whether to perturb the corresponding frame
    #X is of size (batch, seq_len, in_features)
    #the output noise will be the same size as X
    #eps cound be a real number or a tensor of size (batch, seq_len)
    #generateSamples X' such that ||x'-x||p <= e for each sample at each time step
    with torch.no_grad():
        N = shape[0]
        seq_len = shape[1]
        if eps_idx is None:
            eps_idx = torch.ones(seq_len, device=device)
        if type(eps) == torch.Tensor:
            #eps could be a tensor of shape N
            # eps = eps.to(device)
            if len(eps.shape) == 1:
                eps = eps.unsqueeze(1)#.expand(-1, seq_len) #(N, seq_len)
            if not (device is None):
                eps = eps.to(device)
            # eps = eps.view(N*seq_len)
        noise = torch.rand(shape, device = device) - 0.5 #from -0.5 to 0.5
        noise = noise.view(N*seq_len, -1) #(N*seq_len, in_features) 
        data_norm = torch.norm(noise, p=p, dim=1) #size N*seq_len
        #eps (N,1) or (N,seq_len)
        desire_norm = torch.rand([N,seq_len], device=device) * eps #(N,seq_len)
        desire_norm = desire_norm.view(N*seq_len)
        times = desire_norm/data_norm #N*seq_len
        # noise = torch.einsum('i...,i->i...',(noise, times))
        noise = noise * times.unsqueeze(1)
        noise = noise.view(N, seq_len, -1) * eps_idx.unsqueeze(0).unsqueeze(2)
    return noise


def verify_bound(lstm,m,p,eps,x, max_iter=100, verify_y=True, verify_c=True,
                 verify_a = True, thred=1e-3, eps_idx=None, a0 = None, c0 = None, print_info = True):
    #m range from 1 to seq_len
    # yi_min,yi_max,yf_min,yf_max,yg_min,yg_max,yo_min,yo_max = lstm.get_y(
    #         m=m,eps=eps,x=x,p=p)
    yi_min = lstm.yi_l[m-1]
    yi_max = lstm.yi_u[m-1]
    yf_min = lstm.yf_l[m-1]
    yf_max = lstm.yf_u[m-1]
    yg_min = lstm.yg_l[m-1]
    yg_max = lstm.yg_u[m-1]
    yo_min = lstm.yo_l[m-1]
    yo_max = lstm.yo_u[m-1]
    
    for i in range(max_iter):
        noise = generateNoise(x.shape, p, eps, device=x.device,
                              eps_idx = eps_idx)
        a,c,yi,yf,yg,yo = lstm(x+noise, a0=a0, c0=c0)
        
        if verify_y:
            yi_yi_min = (yi[:,m-1,:]-yi_min).min()
            yf_yf_min = (yf[:,m-1,:]-yf_min).min()
            yg_yg_min = (yg[:,m-1,:]-yg_min).min()
            yo_yo_min = (yo[:,m-1,:]-yo_min).min()
            if print_info:
                print('yi abs mean %.4f yi-yi_l mean %.4f yi-yi_l min %.4f ' % (
                        yi[:,m-1,:].abs().mean(), (yi[:,m-1,:]-yi_min).mean(), yi_yi_min))
                print('yf abs mean %.4f yf-yf_l mean %.4f yf-yf_l min %.4f ' % (
                        yf[:,m-1,:].abs().mean(), (yf[:,m-1,:]-yf_min).mean(), yf_yf_min))
                print('yg abs mean %.4f yg-yg_l mean %.4f yg-yg_l min %.4f ' % (
                        yg[:,m-1,:].abs().mean(), (yg[:,m-1,:]-yg_min).mean(), yg_yg_min))
                print('yo abs mean %.4f yo-yo_l mean %.4f yo-yo_l min %.4f ' % (
                        yo[:,m-1,:].abs().mean(), (yo[:,m-1,:]-yo_min).mean(), yo_yo_min))
            # print(yf_yf_min)
            # print(yg_yg_min)
            # print(yo_yo_min)
            if (yi_yi_min<-thred) or (yf_yf_min<-thred) or (yg_yg_min<-thred) or (yo_yo_min<-thred):
                raise Exception('fail')
            
            
            yi_max_yi = (yi_max - yi[:,m-1,:]).min()
            yf_max_yf = (yf_max - yf[:,m-1,:]).min()
            yg_max_yg = (yg_max - yg[:,m-1,:]).min()
            yo_max_yo = (yo_max - yo[:,m-1,:]).min()
            if print_info:
                print('yi abs mean %.4f yi_u-yi mean %.4f yi_u-yi min %.4f ' % (
                        yi[:,m-1,:].abs().mean(), (yi_max - yi[:,m-1,:]).mean(), yi_max_yi))
                print('yf abs mean %.4f yf_u-yf mean %.4f yf_u-yf min %.4f ' % (
                        yf[:,m-1,:].abs().mean(), (yf_max - yf[:,m-1,:]).mean(), yf_max_yf))
                print('yg abs mean %.4f yg_u-yg mean %.4f yg_u-yg min %.4f ' % (
                        yg[:,m-1,:].abs().mean(), (yg_max - yg[:,m-1,:]).mean(), yg_max_yg))
                print('yo abs mean %.4f yo_u-yo mean %.4f yo_u-yo min %.4f ' % (
                        yo[:,m-1,:].abs().mean(), (yo_max - yo[:,m-1,:]).mean(), yo_max_yo))
            # print(yi_max_yi)
            # print(yf_max_yf)
            # print(yg_max_yg)
            # print(yo_max_yo)
            
            if (yi_max_yi<-thred) or (yf_max_yf<-thred) or (yg_max_yg<-thred) or (yo_max_yo<-thred):
                raise Exception('fail')
        if verify_c:    
            cl_min = (c[:,m-1,:] - lstm.c_l[m-1]).min()
            cu_min = (lstm.c_u[m-1] - c[:,m-1,:]).min()
            if print_info:
                print('c abs mean %.4f c-c_l mean %.4f c-c_l min %.4f ' % (
                        c[:,m-1,:].abs().mean(), (c[:,m-1,:]-lstm.c_l[m-1]).mean(), cl_min))
                print('c abs mean %.4f c_u-c mean %.4f c_u-c min %.4f ' % (
                        c[:,m-1,:].abs().mean(), (lstm.c_u[m-1]- c[:,m-1,:]).mean(), cu_min))
            # print('cl',cl_min)
            # print('cu',cu_min)
            if cl_min<-thred or cu_min<-thred:
                raise Exception('fail')
        
        if verify_a:
            al_min = (a[:,m-1,:] - lstm.a_l[m-1]).min() 
            au_min = (lstm.a_u[m-1] - a[:,m-1,:]).min()
            if print_info:
                print('a abs mean %.4f a-a_l mean %.4f a-a_l min %.4f ' % (
                        a[:,m-1,:].abs().mean(), (a[:,m-1,:]-lstm.a_l[m-1]).mean(), al_min))
                print('a abs mean %.4f a_u-a mean %.4f a_u-a min %.4f ' % (
                        a[:,m-1,:].abs().mean(), (lstm.a_u[m-1]- a[:,m-1,:]).mean(), au_min))
            # print('al',al_min)
            # print('au',au_min)
            if al_min<-thred or au_min<-thred:
                raise Exception('fail')
        
    # print('success!')
    print('layer %d bound verification succeed' % m)
    return 0

def verify_final_output(lstm_classifier, x, minimum, maximum,p,eps,
                        max_iter = 100, thred=1e-4, eps_idx=None, 
                        a0 = None, c0 = None, print_info = True):
    for i in range(max_iter):
        noise = generateNoise(x.shape, p, eps, device=x.device,
                              eps_idx = eps_idx)
        out = lstm_classifier(x+noise,(a0,c0))
        
        zl_min = (out - minimum).min() 
        zu_min = (maximum - out).min()
        if print_info:
            print('f abs mean %.4f f-f_l mean %.4f f-f_l min %.4f ' % (
                        out.abs().mean(), (out-minimum).mean(), zl_min))
            print('f abs mean %.4f f_u-f mean %.4f f_u-f min %.4f ' % (
                        out.abs().mean(), (maximum-out).mean(), zu_min))
        # print('zl',zl_min)
        # print('zu',zu_min)
        if zl_min<-thred or zu_min<-thred:
            raise Exception('fail')
    print('final output bound verification succeed!')
    return 0

def verify_final_output2(lstm, W, b, x, minimum, maximum,p,eps,
                        max_iter = 100, thred=1e-4, eps_idx = None,
                        a0 = None, c0 = None, print_info = True):
    for i in range(max_iter):
        noise = generateNoise(x.shape, p, eps, device=x.device,
                              eps_idx = eps_idx)
        a,_,_,_,_,_ = lstm(x+noise, a0=a0, c0=c0)
        a = (a[:,-1,:]).unsqueeze(2) #batch hidden 1
        #W out hidden
        out = torch.matmul(W,a).squeeze(2) #batch out
        #b (out)
        out = out + b.unsqueeze(0)
        zl_min = (out - minimum).min() 
        zu_min = (maximum - out).min()
        if print_info:
            print('f abs mean %.4f f-f_l mean %.4f f-f_l min %.4f ' % (
                        out.abs().mean(), (out-minimum).mean(), zl_min))
            print('f abs mean %.4f f_u-f mean %.4f f_u-f min %.4f ' % (
                        out.abs().mean(), (maximum-out).mean(), zu_min))
        # print('zl',zl_min)
        # print('zu',zu_min)
        if zl_min<-thred or zu_min<-thred:
            raise Exception('fail')
    print('final output bound verification succeed!')
    return 0

def verifyGetMaximumEps(lstm, x0, eps, p,true_label, target_label, 
                        eps_idx = None, a0 = None, c0 = None, untargeted=False,
                        max_iter = 1000):
    N = x0.shape[0]
    idx = torch.arange(N)
    
    y0 = lstm.get_final_output(x0, a0=a0, c0=c0)
    # pred0 = y0.argmax(dim=1)
    out0 = y0[idx, true_label]
    
    
    for i in range(max_iter):
        noise = generateNoise(x0.shape, p, eps, eps_idx = eps_idx, 
                              device=x0.device)
        y = lstm.get_final_output(x0 + noise, a0=a0, c0=c0)
        # pred = y.argmax(dim=1)
        if not untargeted:
            out = y[idx, target_label]
            out = out-(1e-3)
        else:
            y[idx, true_label] = y[idx, true_label] - 1e8
            out = torch.max(y, dim=1)[0]
            out = out-(1e-3)
        # valid = (pred0 == pred)
        valid = (out0 >= out)
        print('iter %d true-target min %.4f' % (i, (out0-out).min()))
        if valid.min() < 1:
            print('failed')
            break
    return 0
    
    
    
    
    
    
    
    
    
    
    
    
    
    