#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 16:13:07 2019

@author: root
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from utils import plane, get_volume, plot_surface, plot_2_surface
import tanh_sigmoid as ts

#big idea
#z=ax+by+c is lower than the surface at A-x1, B-x2, x3-x4, A,B
#we search over a,b,c using gradient descent


def qualification_loss(x_minus, x_plus, y_minus, y_plus,a, b ,c, confidence=-0.1):
    #the function determines wheather z=ax+by+c is lower than the surface
    #at A-x1, B-x2, x3-x4, A,B
    
    #at A-x1: h = ax + b*y_minus + c
    #         f = sigmoid(y_minus) * tanh(x)
    loss1 = ts.tanh_lower(torch.sigmoid(y_minus), a,b*y_minus + c,
                       x_minus,x_plus*0, plot=False, num=0)
    valid = (loss1 <= 0)
    loss1 = torch.clamp(loss1, min=confidence)
    
    #at B-x2: h = ax + b*y_plus + c
    #         f = sigmoid(y_plus) * tanh(x)
    loss2 = ts.tanh_lower(torch.sigmoid(y_plus), a,b*y_plus + c,
                       x_minus,x_plus*0, plot=False, num=0)
    valid = valid * (loss2 <= 0)
    loss2 = torch.clamp(loss2, min=confidence)
    
    #at x3-x4: h = b*y + a*x_plus + c
    #          f = tanh(x_plus) sigmoid(y)
    loss3 = ts.sigmoid_lower(torch.tanh(x_plus), b,a*x_plus + c,
                          y_minus,y_plus, plot=False, num=0)
    valid = valid * (loss3 <= 0)
    loss3 = torch.clamp(loss3, min=confidence)
    
    #at A(0,y_minus)
    loss4 = b*y_minus + c - 0
    valid = valid * (loss4 <= 0)
    loss4 = torch.clamp(loss4, min=confidence)
    
    #at B(0, y_plus)
    loss5 = b*y_plus + c - 0
    valid = valid * (loss5 <= 0)
    loss5 = torch.clamp(loss5, min=confidence)
    
    loss = loss1 + loss2 + loss3 + loss4 + loss5
    return loss, valid


import train_activation_plane
def main_lower(x_minus, x_plus, y_minus, y_plus, print_info = True):
    # a0 = torch.zeros(x_minus.shape, device=x_minus.device)
    # b0 = torch.zeros(x_minus.shape, device=x_minus.device)
    # c0 = torch.tanh(x_minus) * torch.sigmoid(y_plus)
    # a,b,c = train_lower(a0,b0,c0,x_minus, x_plus, y_minus, y_plus, lr=1e-3,
    #             max_iter = 500)
    z10 = torch.tanh(x_minus) * torch.sigmoid(y_plus)
    z20 = torch.tanh(x_minus) * torch.sigmoid(y_plus)
    z30 = torch.tanh(x_minus) * torch.sigmoid(y_plus)
    a,b,c = train_activation_plane.train_lower(z10,z20,z30,
                x_minus, x_plus, y_minus, y_plus, qualification_loss, 
                '34l', lr=1e-2,
                max_iter = 500, print_info = print_info)
    return a.detach(),b.detach(),c.detach()

def main_upper(x_minus, x_plus, y_minus, y_plus, print_info = True):
    if print_info:
        print('34 orthant upper: using 34 orthant lower function')
    a,b,c = main_lower(-x_plus, -x_minus, y_minus, y_plus, print_info = print_info)
    b = -b
    c = -c
    return a.detach(),b.detach(),c.detach()

if __name__ == '__main__':
    
    x_minus = torch.Tensor([-0.1])
    x_plus = torch.Tensor([0.1])
    y_minus = torch.Tensor([-5])
    y_plus = torch.Tensor([-3])
    
    num = 0
    print_info = False
    a_l,b_l,c_l = main_lower(x_minus, x_plus, y_minus, y_plus, print_info = print_info)
    
    a_u, b_u, c_u = main_upper(x_minus, x_plus, y_minus, y_plus, print_info = print_info)

    v1, v2 = plot_2_surface(x_minus[num], x_plus[num],y_minus[num], y_plus[num], 
                 a_l[num],b_l[num],c_l[num],a_u[num], b_u[num], c_u[num])
    
    
    
    
    