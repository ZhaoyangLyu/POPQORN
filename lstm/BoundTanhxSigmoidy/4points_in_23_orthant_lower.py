#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 21:25:52 2019

@author: root
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from utils import plane, get_volume, plot_surface, plot_2_surface
# from 4points_in_2nd_orthant import lower_plane
second = __import__('4points_in_2nd_orthant')
import tanh_sigmoid as ts

#big idea: the plane should tangent on the surface in the area A-x2-x3-C
#and be lower than the surface on x1-x4 and AC 

def qualification_loss(x_minus, x_plus, y_minus, y_plus, a, b ,c, confidence=-0.1):
    #this function determines wheather z = ax + by + c is lower than 
    #the surface at x1-x4 and AC
    alpha1 = torch.sigmoid(y_minus)
    loss1 = ts.tanh_lower(alpha1, a,b*y_minus+c,x_minus,x_plus, plot=False, num=0)
    valid = (loss1 <= 0)
    loss1 = torch.clamp(loss1, min=confidence)
    #when loss1<confidence, we stop minimizing it
    
    alpha2 = torch.sigmoid(y_minus*0)
    loss2 = ts.tanh_lower(alpha2, a,c,x_minus,x_plus, plot=False, num=0)
    valid = valid * (loss2<=0)
    loss2 = torch.clamp(loss2, min=confidence)
    
    loss = loss1+loss2
    return loss, valid

def train_lower(u0, v0, ka0, kb0, x_minus, x_plus, y_minus, y_plus, 
          lr_x = 1e-3, lr_k=1e-2, max_iter = 100, print_info = True):
    device = x_minus.device
    x_best = torch.zeros(x_minus.shape).to(device)
    y_best = torch.zeros(x_minus.shape).to(device)
    a_best = torch.zeros(x_minus.shape).to(device)
    b_best = torch.zeros(x_minus.shape).to(device)
    c_best = torch.zeros(x_minus.shape).to(device)
    
    ka_best = torch.zeros(x_minus.shape).to(device)
    kb_best = torch.zeros(x_minus.shape).to(device)
    
    cubic = -(x_plus-x_minus) * (y_plus-y_minus) * torch.tanh(x_minus) * torch.sigmoid(y_plus)
    cubic = torch.clamp(cubic, min=1e-3)
    v_best = -cubic/cubic * 10000
    
    # eps = 0.1
    u = u0.data.clone()#torch.clamp(x_plus.data.clone(), max=3)#torch.rand(x_plus.shape)#torch.Tensor([3])
    v = v0.data.clone()#torch.clamp(y_plus.data.clone(), max=3)#torch.rand(y_plus.shape)#torch.Tensor([3])
    
    ka = ka0.data.clone()#torch.Tensor([1])
    kb = kb0.data.clone()#torch.Tensor([1])
    
    u.requires_grad = True
    v.requires_grad = True
    
    ka.requires_grad = True
    kb.requires_grad = True

    # optimizer = optim.SGD([u,v,ka,kb], lr=lr, momentum=momentum)
    optimizer_x = optim.Adam([u,v], lr=lr_x)
    optimizer_k = optim.Adam([ka,kb], lr=lr_k)
    
    max_iter = max_iter
    
    # tanh_l_min = tanh_lmin(x_minus, y_minus)
    # sigmoid_l_min = sigmoid_lmin(x_plus, y_minus)
    for i in range(max_iter):
        #x: 0 to x_minus <=0
        #y: 0 to y_plus
        slop = 0.01
        
        idx_x = (u<=x_plus).float()
        u_minus = u * idx_x + (1-idx_x)*(slop*(u-x_plus)+x_plus)
        #this makes u_minus grows much slower when u>=x_plus
        
        # u_minus = -F.leaky_relu(-u, negative_slope=0.01)
        #this makes u_minus to be non positive
        v_plus = F.leaky_relu(v, negative_slope=slop)
        #this makes v_plus to be non negative
        
        idx_x = (u>=x_minus).float()
        x = u_minus * idx_x + (1-idx_x)*(slop*(u_minus-x_minus)+x_minus)
        idx_y = (v<=y_plus).float()
        y = v_plus * idx_y + (1-idx_y)*(slop*(v_plus-y_plus)+y_plus)
       
        a,b,c = plane(x,y)
            
        idx = (x<=x_minus).float()
        a = a - F.leaky_relu(ka, negative_slope=slop) * idx
        c = c + F.leaky_relu(ka, negative_slope=slop) * x * idx
        
        #ka = ka * idx 
        #if x<=x_minus, we keep its original value
        #if x>x_minus, we reset it to 0
        
        idx = (y>=y_plus).float()
        b = b + F.leaky_relu(kb, negative_slope=slop) * idx
        c = c - F.leaky_relu(kb, negative_slope=slop) * y * idx
        
        # q_loss, valid = qualification_loss_lower(a,b,c,x_minus, x_plus, y_minus, y_plus, 
        #                      tanh_l_min,
        #                sigmoid_l_min, confidence=-0)
        q_loss, valid = qualification_loss(x_minus, x_plus, y_minus, y_plus, 
                                           a, b ,c, confidence=-0)
        # print('q_loss:',q_loss)
        v_loss = get_volume(a,b,c,x_minus, x_plus, y_minus, y_plus)
        v_loss = v_loss/cubic
        # print('volume', v_loss)
        if print_info:
            print('23l q loss: %.4f volume: %.4f' % (q_loss.mean().item(), v_loss.mean().item()))
        loss = (q_loss - v_loss*(valid.float()+0.01)).mean() #we want to maximize the volume
        
        best = (v_loss > v_best) * (valid)
        v_best[best] = v_loss[best]
        x_best[best] = x[best]
        y_best[best] = y[best]
        ka_best[best] = ka[best]
        kb_best[best] = kb[best]
        a_best[best] = a[best]
        b_best[best] = b[best]
        c_best[best] = c[best]
        
        
        optimizer_x.zero_grad()
        optimizer_k.zero_grad()
        loss.backward()
        
        idx = (y>y_plus) * (kb>0)
        #if y>y_plus and kb>0, we don't move v
        v.grad = v.grad * (1-idx.float())
        
        idx = (x<x_minus) * (ka>0)
        #if x<x_minus and ka>0, we don't move u 
        u.grad = u.grad * (1-idx.float())
        # if y>=y_plus and kb >=0:
        #     v.grad = v.grad * 0
        # if x<=x_plus and ka >=0:
        #     u.grad = u.grad * 0
        optimizer_x.step()
        optimizer_k.step()
        # print('u,v:',u,v)
    
    return x_best,y_best,ka_best,kb_best,a_best,b_best,c_best,v_best 

def find_initial_feasible_solution(x_minus, x_plus, y_minus, y_plus):
    
    with torch.no_grad():
        device = x_minus.device
        x_best = torch.zeros(x_minus.shape).to(device)
        y_best = torch.zeros(x_minus.shape).to(device)
        # c_best = torch.zeros(x_minus.shape)
        ka_best = torch.zeros(x_minus.shape).to(device)
        kb_best = torch.zeros(x_minus.shape).to(device)
        
       
       
        x = x_minus
        y = y_plus
        a,b,c = plane(x,y)
        q_loss, valid = qualification_loss(x_minus, x_plus, y_minus, y_plus, 
                                           a, b ,c, confidence=-0)
    
        if valid.sum()>0:
            x_temp, y_temp = binary_search_x(x_minus[valid], x_plus[valid],
                                             y_minus[valid], y_plus[valid])
            x_best[valid] = x_temp
            y_best[valid] = y_temp
            ka_best[valid] = 0
            kb_best[valid] = 0
        
        valid = 1-valid
        if valid.sum()>0:
            ka_temp, kb_temp = binary_search_k(x_minus[valid], x_plus[valid], 
                                               y_minus[valid], y_plus[valid])
            x_best[valid] = x_minus[valid]
            y_best[valid] = y_plus[valid]
            ka_best[valid] = ka_temp
            kb_best[valid] = kb_temp
    return x_best, y_best, ka_best, kb_best

def binary_search_x(x_minus, x_plus, y_minus, y_plus):
    #users must make sure there exits a feasible solution in the rectangular area
    
    #the search range is x_minus <= x <= x_plus, 0 <= y <= y_plus
    with torch.no_grad():
        a,b,c = plane(x_minus,y_plus)
        q_loss, valid = qualification_loss(x_minus, x_plus, y_minus, y_plus,
                                           a, b ,c, confidence=-0)

        if valid.min()<1:
            raise Exception('(x_minus, y_plus) is not always feasible')
        
        
        alpha_u = torch.ones(x_minus.shape, device=x_minus.device)
        #alpha_u is always feasible
        alpha_l = alpha_u * 0
        for i in range(10):
            alpha = (alpha_u + alpha_l)/2
            x = x_plus + (x_minus-x_plus) * alpha
            y = y_plus * alpha
            a,b,c = plane(x,y)
            q_loss, valid = qualification_loss(x_minus, x_plus, y_minus, y_plus,
                                           a, b ,c, confidence=-0)
            valid = valid.float()
            alpha_l = (1-valid)*alpha + valid*alpha_l
            alpha_u = valid*alpha + (1-valid)*alpha_u
      
    return x_plus + (x_minus-x_plus) * alpha_u, alpha_u*y_plus

def binary_search_k(x_minus, x_plus, y_minus, y_plus):
    with torch.no_grad():
       
        # ka = a #a = a - ka, ka from 0 to a
        # kb = b #b = b - kb, kb from 0 to b
        a0,b0,c0 = plane(x_minus,y_plus)
        
        alpha_u = torch.ones(x_minus.shape, device=x_minus.device)
        #alpha_u is always feasible
        alpha_l = alpha_u * 0
        #ka = alpha * a, a = a - ka = (1-alpha) * a
        #kb = alpha * b, b = b - kb = (1-alpha) * b
        
        for i in range(10):
            alpha = (alpha_u + alpha_l)/2
            ka = a0 * alpha
            kb = -b0 * alpha
            
            a = a0 - ka
            # c = c0 - ka * x_minus
            b = b0 + kb
            c = c0 - kb * y_plus + ka * x_minus
            q_loss, valid = qualification_loss(x_minus, x_plus, y_minus, y_plus,
                                           a, b ,c, confidence=-0)    
            valid = valid.float()
            alpha_l = (1-valid)*alpha + valid*alpha_l
            alpha_u = valid*alpha + (1-valid)*alpha_u
    return a0*alpha_u, -b0*alpha_u        

def main_lower(x_minus, x_plus, y_minus, y_plus, print_info = True):
    u0,v0,ka0,kb0 = find_initial_feasible_solution(x_minus, x_plus, y_minus, y_plus)
    x,y,ka,kb,a,b,c,v = train_lower(u0, v0, ka0, kb0, x_minus, x_plus, y_minus, y_plus, 
          lr_x = 1e-3, lr_k=1e-2, max_iter = 100, print_info = print_info)
    # increase = second.lower_plane(a,b,c,x,y,x_minus, x_plus, y_minus*0, y_plus)
    increase = second.lower_plane(a,b,c,x,y,x_minus, x_plus, y_minus*0, y_plus, print_info = print_info)
    # print(a,b,c,x,y,x_minus, x_plus, y_minus*0, y_plus)
    # print('increase', increase)
    c = c+increase
    return a.detach(),b.detach(),c.detach()

if __name__ == '__main__':
   
    x_minus = torch.Tensor([-1])
    x_plus = torch.Tensor([0])
    y_minus = torch.Tensor([-5])
    y_plus = torch.Tensor([5])
    # 
    num=0
    
    a,b,c = main_lower(x_minus, x_plus, y_minus, y_plus, print_info = False)
    plot_surface(x_minus[num], x_plus[num],y_minus[num], y_plus[num],
                                a[num],b[num],c[num])
    
    
    
    
    