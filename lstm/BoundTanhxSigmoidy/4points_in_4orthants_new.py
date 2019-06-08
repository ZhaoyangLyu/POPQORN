#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 19:30:04 2019

@author: root
"""

import torch
import matplotlib.pyplot as plt
import torch.optim as optim
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch.nn.functional as F
from utils import plane,get_volume, plot_surface
from utils import plot_2_surface
second = __import__('4points_in_2nd_orthant')
import tanh_sigmoid as ts
import time

def train_lower(u0, v0, ka0, kb0, x_minus, x_plus, y_minus, y_plus, 
          lr_x = 1e-3, lr_k=1e-2, max_iter = 100, print_info=True):
    device = x_minus.device
    x_best = torch.zeros(x_minus.shape).to(device)
    y_best = torch.zeros(x_minus.shape).to(device)
    a_best = torch.zeros(x_minus.shape).to(device)
    b_best = torch.zeros(x_minus.shape).to(device)
    c_best = torch.zeros(x_minus.shape).to(device)
    
    ka_best = torch.zeros(x_minus.shape).to(device)
    kb_best = torch.zeros(x_minus.shape).to(device)
    
    cubic = (x_plus-x_minus) * (y_plus-y_minus)
    cubic = torch.clamp(cubic, min=1e-4)
    

    v_best = -torch.ones(x_minus.shape).to(device)
    
    # eps = 0.1
    u = u0#torch.clamp(x_plus.data.clone(), max=3)#torch.rand(x_plus.shape)#torch.Tensor([3])
    v = v0#torch.clamp(y_plus.data.clone(), max=3)#torch.rand(y_plus.shape)#torch.Tensor([3])
    
    ka = ka0#torch.Tensor([1])
    kb = kb0#torch.Tensor([1])
    
    u.requires_grad = True
    v.requires_grad = True
    
    ka.requires_grad = True
    kb.requires_grad = True

    # optimizer = optim.SGD([u,v,ka,kb], lr=lr, momentum=momentum)
    optimizer_x = optim.Adam([u,v], lr=lr_x)
    optimizer_k = optim.Adam([ka,kb], lr=lr_k)
    
    max_iter = max_iter
    
    for i in range(max_iter):
        #x: 0 to x_minus <=0
        #y: 0 to y_plus
        slop = 0.01
        u_minus = -F.leaky_relu(-u, negative_slope=0.01)
        v_plus = F.leaky_relu(v, negative_slope=0.01)
        
        idx_x = (u>=x_minus).float()
        x = u_minus * idx_x + (1-idx_x)*(slop*(u_minus-x_minus)+x_minus)
        idx_y = (v<=y_plus).float()
        y = v_plus * idx_y + (1-idx_y)*(slop*(v_plus-y_plus)+y_plus)
       
        a,b,c = plane(x,y)
            
        idx = (x<=x_minus).float()
        a = a - F.leaky_relu(ka, negative_slope=0.01) * idx
        c = c + F.leaky_relu(ka, negative_slope=0.01) * x * idx
        
        #ka = ka * idx 
        #if x<=x_minus, we keep its original value
        #if x>x_minus, we reset it to 0
        
        idx = (y>=y_plus).float()
        b = b + F.leaky_relu(kb, negative_slope=0.01) * idx
        c = c - F.leaky_relu(kb, negative_slope=0.01) * y * idx
        
        q_loss, valid = qualification_loss_lower(a,b,c,x_minus, x_plus, 
                                                 y_minus, y_plus, confidence=-0)
        # print('q_loss:',q_loss)
        v_loss = get_volume(a,b,c,x_minus, x_plus, y_minus, y_plus)
        v_loss = v_loss/cubic
        # print('cubic', cubic.min())
        # print('u,v', u.max(), u.min(), v.max(), v.min())
        # print('ka,kb', ka.max(), ka.min(), ka.max(), ka.min())
        # # print('volume', v_loss)
        # print('a,b,c',a.max(),b.max(),c.max(), a.min,b.min,c.min)
        if print_info:
            print('all 4l q_loss %.4f, volume %.4f' % (q_loss.mean().item(), v_loss.mean().item()))
        loss = q_loss - (v_loss*(valid.float()+0.1)).mean()
        
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
        v.grad = v.grad * (1-idx.float())
        
        idx = (x<x_minus) * (ka>0)
        u.grad = u.grad * (1-idx.float())
        # print('u grad', u.grad)
        # print('v grad', v.grad)
        # nan = (u.grad != u.grad)
        # print('not a number', u0[nan], v0[nan], ka0[nan], kb0[nan], 
        #       x_minus[nan], x_plus[nan], y_minus[nan], y_plus[nan])
        
        # print('u', u)
        # print('v', v)
        # if y>=y_plus and kb >=0:
        #     v.grad = v.grad * 0
        # if x<=x_plus and ka >=0:
        #     u.grad = u.grad * 0
        u.grad[u.grad != u.grad] = 0
        v.grad[v.grad != v.grad] = 0
        ka.grad[ka.grad != ka.grad] = 0
        kb.grad[kb.grad != kb.grad] = 0
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
        q_loss, valid = qualification_loss_lower(a,b,c,x_minus, x_plus, 
                                                     y_minus, y_plus, confidence=-0)
    
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
    #users must make sure it is feasible
    with torch.no_grad():
        a,b,c = plane(x_minus,y_plus)
        q_loss, valid = qualification_loss_lower(a,b,c,x_minus, x_plus, 
                                                 y_minus, y_plus, confidence=-0)
        if valid.min()<1:
            raise Exception('(x_minus, y_plus) is not always feasible')
        
        
        alpha_u = torch.ones(x_minus.shape, device = x_minus.device)
        #alpha_u is always feasible
        alpha_l = alpha_u * 0
        for i in range(10):
            alpha = (alpha_u + alpha_l)/2
           
            x = x_minus * alpha
            y = y_plus * alpha
            a,b,c = plane(x,y)
            q_loss, valid = qualification_loss_lower(a,b,c,x_minus, x_plus, 
                                                     y_minus, y_plus, confidence=-0)
            valid = valid.float()
            alpha_l = (1-valid)*alpha + valid*alpha_l
            alpha_u = valid*alpha + (1-valid)*alpha_u
    return alpha_u*x_minus, alpha_u*y_plus

def binary_search_k(x_minus, x_plus, y_minus, y_plus):
    with torch.no_grad():
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

            q_loss, valid = qualification_loss_lower(a,b,c,x_minus, x_plus, 
                                                     y_minus, y_plus, confidence=-0)
                
            valid = valid.float()
            alpha_l = (1-valid)*alpha + valid*alpha_l
            alpha_u = valid*alpha + (1-valid)*alpha_u
    return a0*alpha_u, -b0*alpha_u        
    

def qualification_loss_lower(a,b,c,x_minus, x_plus, y_minus, y_plus, confidence=-0.01):
    #check whether a*x + b*y + c is below z(x) in the rectangular area
    #we require loss1-loss7 <=0
    
    #h(x,y) = a*x + b*y + c
    #z(x,y) = tanh(x) * sigmoid(y)
    #x1 (x_minus, y_minus)
    #x2 (x_minus, y_plus)
    #x3 (x_plus, y_plus)
    #x4 (x_plus, y_minus)
    #A (x_minus, 0) z = 0.5 * tanh(x_minus), h = a*x_minus + c
    #B (0, y_plus) z = 0, h = b*y_plus + c
    #C (x_plus, 0) z = 0.5 * tanh(x_plus), h = a*x_plus + c
    #D (0, y_minus) z = 0, h = b*y_minus + c
    #O (0,0) z=0, h =c
    # h-z <=0, minimize h-z
    
    
    loss1 = b*y_plus + c #B
    valid = (loss1<=0)
    # print('loss1', loss1)
    loss1 = torch.clamp(loss1, min = confidence).mean()
    
    loss2 = ((a*x_plus + b*y_plus + c)-
             torch.tanh(x_plus) * torch.sigmoid(y_plus)) #x3
    valid = valid * (loss2<=0)
    # print('loss2', loss2)
    loss2 = torch.clamp(loss2, min = confidence).mean()
    
    loss3 = (a*x_plus + c) - 0.5 * torch.tanh(x_plus)#C
    valid = valid * (loss3<=0)
    # print('loss3', loss3)
    loss3 = torch.clamp(loss3, min = confidence).mean()
    
    loss4 = c - 0 #O
    valid = valid * (loss4<=0)
    # print('loss4', loss4)
    loss4 = torch.clamp(loss4, min = confidence).mean()
    
    loss5 = (b*y_minus + c) - 0 #D
    valid = valid * (loss5<=0)
    # print('loss5', loss5)
    loss5 = torch.clamp(loss5, min = confidence).mean()
    
    #C-x_4
    #z = by + a*x_plus +  c
    #f = torch.tanh(x_plus) * torch.sigmoid(y)
    loss6 = ts.sigmoid_lower(torch.tanh(x_plus), b,a*x_plus +  c,
                          y_minus,y_plus*0, plot=False, num=0)
    # print('loss6', loss6.mean())
    valid = valid * (loss6<=0)
    # print('loss6', loss6)
    loss6 = torch.clamp(loss6, min=confidence)
    loss6 = loss6.mean()
    
    #A-o
    #z = ax + c
    #f = sigmoid(0) * tanh(x)
    loss7 = ts.tanh_lower(0.5*torch.ones(x_minus.shape, device = x_minus.device), 
                          a,c,x_minus,x_plus*0, plot=False, num=0)
    # print('loss7', loss7.mean())
    valid = valid * (loss7<=0)
    # print('loss7', loss7)
    loss7 = torch.clamp(loss7, min=confidence)
    loss7 = loss7.mean()
    
    #D-x1
    #z = ax + b*y_minus + c
    #f = sigmoid(y_minus) * tanh(x)
    loss8 = ts.tanh_lower(torch.sigmoid(y_minus), a,b*y_minus + c,x_minus,x_plus*0, plot=False, num=0)
    # print('loss8', loss8.mean())
    valid = valid * (loss8<=0)
    # print('loss7', loss7)
    loss8 = torch.clamp(loss8, min=confidence)
    loss8 = loss8.mean()
    
    loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8
    # print('loss1', loss1.mean())
    # print('loss2', loss2.mean())
    # print('loss3', loss3.mean())
    # print('loss4', loss4.mean())
    # print('loss5', loss5.mean())
    return loss, valid


def main_lower(x_minus, x_plus, y_minus, y_plus, print_info = True):
    u0, v0, ka0, kb0 = find_initial_feasible_solution(x_minus, x_plus, y_minus, y_plus)
    x,y,ka,kb,a,b,c,v = train_lower(u0, v0, ka0, kb0, x_minus, x_plus, y_minus, y_plus, 
          lr_x = 1e-2, lr_k=1e-2, max_iter = 200, print_info=print_info)
    increase = second.lower_plane(a,b,c,x,y,x_minus, x_plus*0, y_minus*0, y_plus, print_info=print_info)
    c = c+increase
    return a.detach(),b.detach(),c.detach()

def main_upper(x_minus, x_plus, y_minus, y_plus, print_info = True):
    if print_info:
        print('4 orthants upper: using 4 orthants lower function')
    a,b,c = main_lower(-x_plus, -x_minus, y_minus, y_plus, print_info = print_info)
    b = -b
    c = -c
    return a.detach(),b.detach(),c.detach()

if __name__ == '__main__':
    
    length = 5
    x_minus = torch.Tensor([-length])
    x_plus = torch.Tensor([length])
    y_minus = torch.Tensor([-length])
    y_plus = torch.Tensor([length])
    
    num = 0
    print_info = False
    
    a_u,b_u,c_u = main_upper(x_minus, x_plus, y_minus, y_plus, print_info = print_info)
    a_l,b_l,c_l = main_lower(x_minus, x_plus, y_minus, y_plus, print_info = print_info)
    
    v1, v2 = plot_2_surface(x_minus[num], x_plus[num],y_minus[num], y_plus[num], 
                 a_l[num],b_l[num],c_l[num],a_u[num], b_u[num], c_u[num])
    
    
    