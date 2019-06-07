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
    #shape is of size (N, seq_len, in_features)
    #the output noise will be of size shape
    #eps cound be a real number or a tensor of size (N)
    #generateSamples X' such that ||x'-x||p <= e for each sample at each time step
    with torch.no_grad():
        N = shape[0]
        seq_len = shape[1]
        if eps_idx is None:
            eps_idx = torch.ones(seq_len, device=device)
        if type(eps) == torch.Tensor:
            #eps could be a tensor of shape N or a single number
            if len(eps.shape) == 1:
                eps = eps.unsqueeze(1)#now is of shape (N, 1)
            if not (device is None):
                eps = eps.to(device)
        
        noise = torch.rand(shape, device = device) - 0.5 #from -0.5 to 0.5
        noise = noise.view(N*seq_len, -1) #(N*seq_len, in_features) 
        data_norm = torch.norm(noise, p=p, dim=1) #size N*seq_len
        #eps is a tensor of shape (N,1) or a single number
        desire_norm = torch.rand([N,seq_len], device=device) * eps # (N,seq_len) * (N,1)
        desire_norm = desire_norm.view(N*seq_len) # (N*seq_len)
        times = desire_norm/data_norm # N*seq_len
        
        noise = noise * times.unsqueeze(1) # (N*seq_len, in_features) * (N*seq_len, 1) 
        noise = noise.view(N, seq_len, -1) * eps_idx.unsqueeze(0).unsqueeze(2) # (N, seq_len, in_features) * (1,seq_len,1)
    return noise


def verify_final_output(classifier, x, minimum, maximum,p,eps,
                        max_iter = 100, thred=1e-4, eps_idx=None):
    for i in range(max_iter):
        noise = generateNoise(x.shape, p, eps, device=x.device,
                              eps_idx = eps_idx)
        out = classifier(x+noise)
        
        zl_min = (out - minimum).min() 
        zu_min = (maximum - out).min()
        print('f abs mean %.4f f-f_l mean %.4f f-f_l min %.4f ' % (
                    out.abs().mean(), (out-minimum).mean(), zl_min))
        print('f abs mean %.4f f_u-f mean %.4f f_u-f min %.4f ' % (
                    out.abs().mean(), (maximum-out).mean(), zu_min))
        
        if zl_min<-thred or zu_min<-thred:
            raise Exception('fail')
    print('final output verification succeed!')
    return 0



def verifyMaximumEps(classifier, x0, eps, p,true_label, target_label, 
                        eps_idx = None, untargeted=False, thred=1e-4):
    N = x0.shape[0]
    idx = torch.arange(N)
    
    y0 = classifier(x0)
    out0 = y0[idx, true_label]
    max_iter = 1000
    
    for i in range(max_iter):
        noise = generateNoise(x0.shape, p, eps, eps_idx = eps_idx, device=x0.device)
        y = classifier(x0 + noise)
        if not untargeted:
            out = y[idx, target_label]
        else:
            y[idx, true_label] = y[idx, true_label] - 1e8
            out = torch.max(y, dim=1)[0]
            
        valid = (out0 + thred >= out)
        print('iter %d true-target min %.4f' % (i, (out0-out).min()))
        if valid.min() < 1:
            print('failed')
            break
    return 0
    
if __name__ == '__main__':
    print(0)
    
    
    
    
    
    
    
    
    
    
    
    
    
    