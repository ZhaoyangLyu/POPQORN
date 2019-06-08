#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 19:27:25 2019

@author: root
"""
import sys
sys.path.append('../')
from lstm import get_last_layer_bound, My_lstm, cut_lstm
import torch 


def getUntargetedMaximumEps(lstm, W,b, x, p,true_label, save_dir, 
                  eps0=1, max_iter=10, acc=0.001, eps_idx=None, head_info = None):
    #this function finds bound for untargeted attack
    a0 = lstm.a0.data.clone()
    c0 = lstm.c0.data.clone()
    #when u_eps-l_eps < acc, we stop searching
    log_file = save_dir + 'logs.txt'
    f = open(log_file, 'a')
    
    N = x.shape[0]
    idx=torch.arange(N)
    l_eps = torch.zeros(N, device = x.device) #lower bound of eps
    u_eps = torch.ones(N, device = x.device) * eps0 #upper bound of eps
    
    yL,yU = get_last_layer_bound(W,b,lstm,x,u_eps,p, verify_bound=True, 
                                 reset=True, eps_idx=eps_idx)
    #yL and yU is of shape (N, num_labels)
    if not head_info is None:
        f.write(head_info)
    f.write('-' * 80 + '\n')
    f.write('initial u_eps \n' + str(u_eps) + '\n')

    
    true_lower = yL[idx,true_label]
    yU[idx, true_label] = yU[idx, true_label] - 1e8
    max_upper = torch.max(yU, dim=1)[0]
    increase_u_eps = true_lower > max_upper 
    #indicate whether to further increase the upper bound of eps 
    f.write('initial true_lower - max_upper \n' +str(true_lower - max_upper)+'\n')
    
    f.close()
    while (increase_u_eps.sum()>0):
        f = open(log_file, 'a')
        #find true and nontrivial lower bound and upper bound of eps
        num = increase_u_eps.sum()
        l_eps[increase_u_eps] = u_eps[increase_u_eps]
        #for those increase_u_eps is 1, we can increase their l_eps to u_eps
        #for those increase_u_eps is 0, u_eps is big enough, we keep their 
        #l_eps and u_eps
        
        u_eps[increase_u_eps ] = u_eps[increase_u_eps ] * 2
        #for those increase_u_eps is 1, increase their increase_u_eps by
        #a factor of 2, try to find a big enough u_eps
        f.write('u_eps \n' + str(u_eps) + '\n')
        f.write('l_eps \n' + str(l_eps) + '\n')
        
        # yL, yU = self.getLastLayerBound(u_eps[increase_u_eps], p, 
        #             x=x[increase_u_eps,:],clearIntermediateVariables=True)
        lstm.a0 = a0[increase_u_eps, :]
        lstm.c0 = c0[increase_u_eps, :]
        yL, yU = get_last_layer_bound(W,b,lstm,x[increase_u_eps,:,:],
                                      u_eps[increase_u_eps],p, verify_bound=True, 
                                      reset=True, eps_idx=eps_idx)
        #yL and yU only for those equal to 1 in increase_u_eps
        #they are of size (num,_)
        
        true_lower = yL[torch.arange(num),true_label[increase_u_eps]]
        yU[torch.arange(num),true_label[increase_u_eps]] = yU[torch.arange(num),true_label[increase_u_eps]] - 1e8
        max_upper = torch.max(yU, dim=1)[0]
        # target_upper = yU[torch.arange(num),target_label[increase_u_eps]]
        temp = true_lower > max_upper #size num
        f.write('true_lower - max_upper \n' + str(true_lower- max_upper) + '\n')
    
        increase_u_eps[increase_u_eps ] = temp    
        f.close()
        
    f = open(log_file, 'a')
    f.write('Finished finding upper and lower bound \n')
    f.write('The upper bound we found is \n'+str(u_eps)+'\n')
    f.write('The lower bound we found is \n'+str(l_eps)+'\n')
    f.close()
    
    search = ((u_eps-l_eps) / ((u_eps+l_eps) / 2 + 1e-8)) > acc
    #indicate whether to further perform binary search
    
    iteration = 0 
    while(search.sum()>0):
        f = open(log_file, 'a')
        #perform binary search
        if iteration > max_iter:
            print('Have reached the maximum number of iterations')
            break
        #print(search)
        num = search.sum()
        eps = (l_eps[search]+u_eps[search])/2
        # yL, yU = self.getLastLayerBound(eps, p, x=x[search,:],
        #                 clearIntermediateVariables=True)
        f.write('binary search step %d \n' % (iteration+1) )
        f.write('eps \n' + str(eps) + '\n')
        lstm.a0 = a0[search, :]
        lstm.c0 = c0[search, :]
        yL, yU = get_last_layer_bound(W,b,lstm,x[search,:,:],
                        eps,p, verify_bound=True, reset=True,
                        eps_idx=eps_idx)
        
        true_lower = yL[torch.arange(num),true_label[search]]
        yU[torch.arange(num),true_label[search]] = yU[torch.arange(num),true_label[search]] - 1e8
        max_upper = torch.max(yU, dim=1)[0]
        temp = true_lower>max_upper
        f.write('true_lower - max_upper \n' + str(true_lower- max_upper) + '\n')
        search_copy = search.data.clone()

        search[search] = temp 
        #set all active units in search to temp
        #original inactive units in search are still inactive
        
        l_eps[search] = eps[temp]
        #increase active and true_lower>target_upper units in l_eps 
        
        u_eps[search_copy-search] = eps[temp==0]
        #decrease active and true_lower<target_upper units in u_eps
        
        search = ((u_eps-l_eps) / ((u_eps+l_eps) / 2 + 1e-8)) > acc #reset active units in search
        # print('u_eps - l_eps \n', u_eps - l_eps)
        
        # print('true_lower - max_upper \n', true_lower-max_upper)
        
        iteration = iteration + 1
        
        f.close()
        
        lstm.a0 = a0
        lstm.c0 = c0

    f = open(log_file, 'a')
    f.write('Finished Binary Search \n')
    f.write('The upper bound we found is \n'+str(u_eps)+'\n')
    f.write('The lower bound we found is \n'+str(l_eps)+'\n')
    f.write('-' * 80 + '\n')
    f.close()

    return l_eps, u_eps   

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    