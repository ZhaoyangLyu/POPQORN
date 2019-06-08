#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 09:11:15 2018

@author: root
"""

import torch
import matplotlib.pyplot as plt

def d_tanh(x):
    #the derivative of tanh
    return 1- (torch.tanh(x))**2

def d_atan(x):
    return 1/(1+x**2)

def d_sigmoid(x):
    sx = torch.sigmoid(x)
    return sx*(1-sx)

Activation = {'tanh':[torch.tanh, d_tanh],
              'atan':[torch.atan, d_atan],
              'sigmoid':[torch.sigmoid, d_sigmoid]}

def getConvenientGeneralActivationBound(l,u, activation):
    if (l>u).sum()>0:
        print(l-u, (l-u).max())
        raise Exception('l must be strictly larger than u')
    func = Activation[activation][0]
    dfunc = Activation[activation][1]
    kl, bl, ku, bu = getGeneralActivationBound(l,u, func, dfunc)
    return kl, bl, ku, bu

def getGeneralActivationBound(l,u, func, dfunc):
    #l and u are tensors of any shape. l and u must have the same shape
    #the first dimension of l and u is the batch dimension
    #users must make sure that u > l
    
    #initialize the desired variables
    device = l.device
    ku = torch.zeros(u.shape, device = device)
    bu = torch.zeros(u.shape, device = device)
    kl = torch.zeros(l.shape, device = device)
    bl = torch.zeros(l.shape, device = device)
    
    yl = func(l)
    yu = func(u)
    k = (yu - yl) / (u-l)
    b = yl - k * l
    d = (u+l) / 2
    
    func_d = func(d)
    d_func_d = dfunc(d) #derivative of tanh at x=d
    
    #l and u both <=0
    minus = (u <= 0) * (l<=0)
    ku[minus] = k[minus]
    bu[minus] = b[minus]
    kl[minus] = d_func_d[minus]
    bl[minus] = func_d[minus] - kl[minus] * d[minus]
    
    #l and u both >=0
    plus = (l >= 0)
    kl[plus] = k[plus]
    bl[plus] = b[plus]
    ku[plus] = d_func_d[plus]
    bu[plus] = func_d[plus] - ku[plus] * d[plus]
    
    #l < 0 and u>0
    pn = (l < 0) * (u > 0)
    kl[pn], bl[pn] = general_lb_pn(l[pn], u[pn], func, dfunc)
    ku[pn], bu[pn] = general_ub_pn(l[pn], u[pn], func, dfunc)
    
    return kl, bl, ku, bu

def getTanhBound(l,u):
    #l and u are tensors of any shape. l and u must have the same shape
    #the first dimension of l and u is the batch dimension
    #users must make sure that u > l
    
    #initialize the desired variables
    device = l.device
    ku = torch.zeros(u.shape, device = device)
    bu = torch.zeros(u.shape, device = device)
    kl = torch.zeros(l.shape, device = device)
    bl = torch.zeros(l.shape, device = device)
    
    yl = torch.tanh(l)
    yu = torch.tanh(u)
    k = (yu - yl) / (u-l)
    b = yl - k * l
    d = (u+l) / 2
    tanh_d = torch.tanh(d)
    d_tanh_d = 1 - tanh_d**2 #derivative of tanh at x=d
    
    #l and u both <=0
    minus = (u <= 0) * (l<=0)
    ku[minus] = k[minus]
    bu[minus] = b[minus]
    kl[minus] = d_tanh_d[minus]
    bl[minus] = tanh_d[minus] - kl[minus] * d[minus]
    
    #l and u both >=0
    plus = (l >= 0)
    kl[plus] = k[plus]
    bl[plus] = b[plus]
    ku[plus] = d_tanh_d[plus]
    bu[plus] = tanh_d[plus] - ku[plus] * d[plus]
    
    #l < 0 and u>0
    pn = (l < 0) * (u > 0)
    kl[pn], bl[pn] = general_lb_pn(l[pn], u[pn], torch.tanh, d_tanh)
    ku[pn], bu[pn] = general_ub_pn(l[pn], u[pn], torch.tanh, d_tanh)
    
    return kl, bl, ku, bu

def testGetGeneralActivationBound():
    u = torch.ones(1) * (3)
    l = torch.ones(1) * (-3)
#    kl, bl, ku, bu = getTanhBound(l,u)
    activation = 'tanh'
    
    func = Activation[activation][0]#torch.atan#torch.tanh##
#    dfunc = Activation[activation][1]
#    kl, bl, ku, bu = getGeneralActivationBound(l,u, func, dfunc)
    kl, bl, ku, bu = getConvenientGeneralActivationBound(l,u, activation)
#    kl, bl, ku, bu = getGeneralActivationBound(l,u, torch.atan, d_atan)
    # x = torch.rand(1000) * (u-l) + l
    x = torch.linspace(l.item(),u.item(),1000) #* (u-l) + l
    
    func_x = func(x)
    l_func_x = kl * x + bl
    u_func_x = ku * x + bu


    linewidth=10
    fontsize = 30
    plt.figure(figsize = (15,12))
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    
    plt.xlabel('v', fontsize = fontsize)
    plt.ylabel('z', fontsize = fontsize)
    plt.plot(x.numpy(), func_x.numpy(), '-', linewidth=linewidth)
    plt.plot(x.numpy(), l_func_x.numpy(), '-', linewidth=linewidth)
    plt.plot(x.numpy(), u_func_x.numpy(), '-', linewidth=linewidth)
    

    print((l_func_x <= func_x).min())
    print(x[l_func_x > func_x], l_func_x[l_func_x > func_x], func_x[l_func_x > func_x])
    print((u_func_x >= func_x).min())
    print(x[u_func_x < func_x], u_func_x[u_func_x < func_x], func_x[u_func_x < func_x])

    plt.show()
    
    

def get_d_UB(l,u,func,dfunc):
    #l and u are tensor of any shape. Their shape should be the same
    #the first dimension of l and u is batch_dimension
    diff = lambda d,l: (func(d)-func(l))/(d-l) - dfunc(d)
    max_iter = 1000;
#    d = u/2
#    d = u
    ub = -l
    d = ub/2
    #use -l as the upper bound as d, it requires f(x) = f(-x)
    #and f to be convex when x<0 and concave when x>0
    
    #originally they use u as the upper bound, it may not always work  
    device = l.device
    lb = torch.zeros(l.shape, device=device);
    keep_search = torch.ones(l.shape, device=device).byte()
    for i in range(max_iter):
        t = diff(d[keep_search], l[keep_search])
        idx = (t<0) + (t.abs() > 0.01)
        keep_search[keep_search] = (idx > 0)
        if keep_search.sum() == 0:
            break
        t = t[idx>0]
       
        idx = t>0
        keep_search_copy = keep_search.data.clone()
        keep_search_copy[keep_search_copy] = idx
        ub[keep_search_copy] = d[keep_search_copy]
        d[keep_search_copy] = (d[keep_search_copy] + lb[keep_search_copy]) / 2
      
        idx = t<0
        keep_search_copy = keep_search.data.clone()
        keep_search_copy[keep_search_copy] = idx
        lb[keep_search_copy] = d[keep_search_copy]
        d[keep_search_copy] = (d[keep_search_copy] + ub[keep_search_copy]) / 2
        
#    print('Use %d iterations' % i) 
#    print(diff(d,l))
#    print('d:', d)
    return d

def general_ub_pn(l, u, func, dfunc):
    d_UB = get_d_UB(l,u,func,dfunc)
#    print(d_UB)
    k = (func(d_UB)-func(l))/(d_UB-l)
    b  = func(l) - (l - 0.01) * k
    return k, b

def get_d_LB(l,u,func,dfunc):
    #l and u are tensor of any shape. Their shape should be the same
    #the first dimension of l and u is batch_dimension
    diff = lambda d,u: (func(d)-func(u))/(d-u) - dfunc(d)
    max_iter = 1000;
#    d = u/2
#    d = u
    device = l.device
    ub = torch.zeros(l.shape, device=device)
    lb = -u
    d = lb/2
    #use -l as the upper bound as d, it requires f(x) = f(-x)
    #and f to be convex when x<0 and concave when x>0
    
    #originally they use u as the upper bound, it may not always work  

    keep_search = torch.ones(l.shape, device=device).byte()
    for i in range(max_iter):
        t = diff(d[keep_search], u[keep_search])
        idx = (t<0) + (t.abs() > 0.01)
        keep_search[keep_search] = (idx > 0)
        if keep_search.sum() == 0:
            break
        t = t[idx>0]
       
        idx = t>0
        keep_search_copy = keep_search.data.clone()
        keep_search_copy[keep_search_copy] = idx
        lb[keep_search_copy] = d[keep_search_copy]
        d[keep_search_copy] = (d[keep_search_copy] + ub[keep_search_copy]) / 2
      
        idx = t<0
        keep_search_copy = keep_search.data.clone()
        keep_search_copy[keep_search_copy] = idx
        ub[keep_search_copy] = d[keep_search_copy]
        d[keep_search_copy] = (d[keep_search_copy] + lb[keep_search_copy]) / 2
        
#    print('Use %d iterations' % i) 
#    print(diff(d,l))
#    print('d:', d)
    return d

def general_lb_pn(l, u, func, dfunc):
    d_LB = get_d_LB(l,u,func,dfunc)
#    print(d_LB)
    k = (func(d_LB)-func(u))/(d_LB-u)
    b  = func(u) - (u + 0.01) * k
    return k, b

def test_general_b_pn():
    u = torch.ones(1) * 1
    l = torch.ones(1) * (-1)
    
#    d = get_d_LB(l,u,torch.tanh,d_tanh)
    ku,bu = general_ub_pn(l, u, torch.tanh,d_tanh)
    kl,bl = general_lb_pn(l, u, torch.tanh,d_tanh)
    
    x = torch.rand(1000) * (u-l) * 1 + l
    y0 = torch.tanh(x)
    yu = ku * x + bu
    yl = kl * x + bl
    print((yu>y0).min())
    print((yl<y0).min())
    plt.plot(x.numpy(),y0.numpy(),'.')
    plt.plot(x.numpy(),yl.numpy(),'.')
    plt.plot(x.numpy(),yu.numpy(),'.')
    
if __name__ == '__main__':
   testGetGeneralActivationBound()
    
    
    
    
    