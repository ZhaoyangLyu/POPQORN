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
              'sigmoid':[torch.sigmoid, d_sigmoid],
              'ba':[torch.sign, 0],
              'relu':[torch.relu, 0],
              'relu_adaptive':[torch.relu, 0]}

def get_bound_for_relu(l, u, adaptive=False):
    device = l.device
    ku = torch.zeros(u.shape, device = device)
    bu = torch.zeros(u.shape, device = device)
    kl = torch.zeros(l.shape, device = device)
    bl = torch.zeros(l.shape, device = device)

    idx = l>=0
    kl[idx] = 1
    ku[idx] = 1

    idx = (l<0) * (u>0)

    k = (u / (u-l))[idx]
    # k u + b = u -> b = (1-k) * u
    b = (1-k) * u[idx]

    
    ku[idx] = k
    bu[idx] = b

    if not adaptive:
        kl[idx] = k
    else:
        idx = (l<0) * (u>0) * (u.abs()>=l.abs())
        kl[idx] = 1
        idx = (l<0) * (u>0) * (u.abs()<l.abs())
        kl[idx] = 0
    return kl, bl, ku, bu

def getConvenientGeneralActivationBound(l,u, activation, use_constant=False):
    if (l>u).sum()>0:
        raise Exception('l must be less or equal to u')
        # print(l-u, (l-u).max())
        # if (l-u).max()>1e-4:
        #     raise Exception('l must be less or equal to u')
        # temp = l>u
        # l[temp] = l[temp] - 1e-4
        # u[temp] = u[temp] + 1e-4
    device = l.device
    
    ku = torch.zeros(u.shape, device = device)
    bu = torch.zeros(u.shape, device = device)
    kl = torch.zeros(l.shape, device = device)
    bl = torch.zeros(l.shape, device = device)
    if use_constant:
        #we have assume that the activaiton is monotomic
        function = Activation[activation][0]
        bu = function(u)
        bl = function(l)
        return kl, bl, ku, bu
    if activation == 'relu':
        kl, bl, ku, bu = get_bound_for_relu(l, u, adaptive=False)
        return kl, bl, ku, bu
    if activation == 'relu_adaptive':
        kl, bl, ku, bu = get_bound_for_relu(l, u, adaptive=True)
        return kl, bl, ku, bu
    if activation == 'ba':
        # print(u)
        print('binary activation')
        bu = torch.sign(u)
        bl = torch.sign(l)
        idx = (l<0) * (u>0) * (u.abs() > l.abs())
        kl[idx] = 2/u[idx]

        idx = (l<0) * (u>0) * (u.abs() < l.abs())
        ku[idx] = -2/l[idx]

        # idx = (l>0) * (l>0.8*u)
        # ku[idx] = 1/l[idx]
        # #ku l + bu = 1
        # bu[idx] = 1-ku[idx] * l[idx]
        print('uncertain neurons', ((l<0) * (u>0)).float().mean())
        return kl, bl, ku, bu
    
    idx = (l==u)
    if idx.sum()>0:
        bu[idx] = l[idx]
        bl[idx] = l[idx]
        
        ku[idx] = 1e-4
        kl[idx] = 1e-4
    
    valid = (1-idx)
    
    if valid.sum()>0:
        func = Activation[activation][0]
        dfunc = Activation[activation][1]
        kl_temp, bl_temp, ku_temp, bu_temp = getGeneralActivationBound(
                l[valid],u[valid], func, dfunc)
        kl[valid] = kl_temp
        ku[valid] = ku_temp
        bl[valid] = bl_temp
        bu[valid] = bu_temp
    # if (kl==0).sum()>0 or (ku==0).sum()>0:
    #     print(kl,ku)
    #     raise Exception('some elements of kl or ku are 0')
    idx2 = (kl==0)
    if idx2.sum()>0:
        kl[idx2] = 1e-8
    idx3 = (ku==0)
    if idx3.sum()>0:
        ku[idx3] = 1e-8
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
    u = torch.ones(1) * (5)
    l = torch.ones(1) * (-4)
#    kl, bl, ku, bu = getTanhBound(l,u)
    activation = 'relu'
    
    func = Activation[activation][0]#torch.atan#torch.tanh##
#    dfunc = Activation[activation][1]
#    kl, bl, ku, bu = getGeneralActivationBound(l,u, func, dfunc)
    kl, bl, ku, bu = getConvenientGeneralActivationBound(l,u, activation,
                    use_constant=True)
#    kl, bl, ku, bu = getGeneralActivationBound(l,u, torch.atan, d_atan)
    x = torch.rand(1000) * (u-l) + l
    
    func_x = func(x)
    l_func_x = kl * x + bl
    u_func_x = ku * x + bu
    
    plt.plot(x.numpy(), func_x.numpy(), '.')
    plt.plot(x.numpy(), l_func_x.numpy(), '.')
    plt.plot(x.numpy(), u_func_x.numpy(), '.')
    
    print((l_func_x <= func_x).min())
    print(x[l_func_x > func_x], l_func_x[l_func_x > func_x], func_x[l_func_x > func_x])
    print((u_func_x >= func_x).min())
    print(x[u_func_x < func_x], u_func_x[u_func_x < func_x], func_x[u_func_x < func_x])
    
    

def get_d_UB(l,u,func,dfunc):
    #l and u are tensor of any shape. Their shape should be the same
    #the first dimension of l and u is batch_dimension
    diff = lambda d,l: (func(d)-func(l))/(d-l) - dfunc(d)
    max_iter = 1000;
    # d = u/2
    # d = u
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
        
    # print('Use %d iterations' % i) 
    # print(diff(d,l))
    # print('d:', d)
    return d

def general_ub_pn(l, u, func, dfunc):
    d_UB = get_d_UB(l,u,func,dfunc)
    # print(d_UB)
    k = (func(d_UB)-func(l))/(d_UB-l)
    b  = func(l) - (l - 0.01) * k
    return k, b

def get_d_LB(l,u,func,dfunc):
    #l and u are tensor of any shape. Their shape should be the same
    #the first dimension of l and u is batch_dimension
    diff = lambda d,u: (func(d)-func(u))/(d-u) - dfunc(d)
    max_iter = 1000;
    # d = u/2
    # d = u
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
        
    # print('Use %d iterations' % i) 
    # print(diff(d,l))
    # print('d:', d)
    return d

def general_lb_pn(l, u, func, dfunc):
    d_LB = get_d_LB(l,u,func,dfunc)
    # print(d_LB)
    k = (func(d_LB)-func(u))/(d_LB-u)
    b  = func(u) - (u + 0.01) * k
    return k, b

def test_general_b_pn():
    u = torch.ones(1) * 1
    l = torch.ones(1) * (-1)
    
    # d = get_d_LB(l,u,torch.tanh,d_tanh)
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
    
    
    
    
    