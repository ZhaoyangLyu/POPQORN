#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 09:15:19 2019

@author: root
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from utils import plane, get_volume, plot_surface, plot_2_surface
import tanh_sigmoid as ts
import train_activation_plane
third =  __import__('4points_in_3rd_orthant')

#big idea, we search over a,b,c using gradient descent
#we can determine wheather a plane is entirely above or below the surface
#only through weather it is above or below the surface on the boundary
#x1-x2, x3-x4; x1-x4, x2-x3  


def sigmoid(x):
    return 1/(1+np.exp(-x))



def qualification_loss_upper(x_minus, x_plus, y_minus, y_plus,a, b ,c, confidence=-0.1):
    #the function determines wheather z=ax+by+c is bigger than the surface
    #at x1-x2, x3-x4
    
    #at x1-x1: h = by + a*x_minus + c
    #         f = sigmoid(y) * tanh(x_minus)
    loss1 = ts.sigmoid_upper(torch.tanh(x_minus), b,
                          a*x_minus + c,y_minus,y_plus)
    valid = (loss1 <= 0)
    loss1 = torch.clamp(loss1, min=confidence)
    
    #at x3-x4: h = by + a*x_plus + c
    #         f = sigmoid(y) * tanh(x_plus)
    loss2 = ts.sigmoid_upper(torch.tanh(x_plus), b,
                          a*x_plus + c,y_minus,y_plus)
    valid = valid * (loss2 <= 0)
    loss2 = torch.clamp(loss2, min=confidence)
    
    
    loss = loss1 + loss2
    return loss, valid

def main_upper(x_minus, x_plus, y_minus, y_plus, a0_is_valid=True, print_info=True):
    a0,b0,c0,v, y1, y2 = third.main_upper(x_minus, x_plus, y_minus, y_plus)
    # a0 = torch.zeros(x_minus.shape)
    # b0 = torch.zeros(x_minus.shape)
    # c0 = torch.tanh(x_plus) * torch.sigmoid(y_minus)
    c0 = c0 + c0.abs() * 0.0001
    
    a=a0
    b=b0
    c=c0
    
    eps = (1e-3) * (y_plus-y_minus)
    idx = (y1>(y_minus+eps)) * (y1<(y_plus-eps)) * (y2>(y_minus+eps)) * (y2<(y_plus-eps))
    
    idx= 1-idx
    if idx.sum()>0:
        if print_info:
            print('3u fine tuning the upper plane in the third orthant')
        # a[idx],b[idx],c[idx] = train_upper(a0[idx],b0[idx],c0[idx],
        #  x_minus[idx], x_plus[idx], y_minus[idx], y_plus[idx], lr=1e-2,
        #             max_iter = 500, a0_is_valid=a0_is_valid)
        z10 = a*x_minus +b*y_minus+c
        z20 = a*x_minus +b*y_plus+c
        z30 = a*x_plus +b*y_plus+c
        a[idx],b[idx],c[idx] = train_activation_plane.train_upper(z10[idx],z20[idx],z30[idx],
                            x_minus[idx], x_plus[idx], y_minus[idx], y_plus[idx], 
                            qualification_loss_upper, '3u', lr=1e-2,
                            max_iter = 500, print_info = print_info)
    return a.detach(),b.detach(),c.detach()



def qualification_loss_lower(x_minus, x_plus, y_minus, y_plus,a, b ,c, confidence=-0.1):
    #the function determines wheather z=ax+by+c is lower than the surface
    #at x1-x4, x2-x3
    
    #at x1-x4: h = ax + b*y_minus + c
    #         f = sigmoid(y_minus) * tanh(x)
    loss1 = ts.tanh_lower(torch.sigmoid(y_minus), a,b*y_minus + c,
                       x_minus,x_plus, plot=False, num=0)
    valid = (loss1 <= 0)
    loss1 = torch.clamp(loss1, min=confidence)
    
    #at x2-x3: h = ax + b*y_plus + c
    #         f = sigmoid(y_plus) * tanh(x)
    loss2 = ts.tanh_lower(torch.sigmoid(y_plus), a,b*y_plus + c,
                       x_minus,x_plus, plot=False, num=0)
    valid = valid * (loss2 <= 0)
    loss2 = torch.clamp(loss2, min=confidence)
    
    
    loss = loss1 + loss2
    return loss, valid


def main_lower(x_minus, x_plus, y_minus, y_plus, a0_is_valid=True, print_info = True):
    a0,b0,c0,v, x1, x2 = third.main_lower(x_minus, x_plus, y_minus, y_plus)
    c0 = c0 - c0.abs() * 0.0001
    # a0 = torch.zeros(x_minus.shape)
    # b0 = torch.zeros(x_minus.shape)
    # c0 = torch.tanh(x_minus) * torch.sigmoid(y_plus)
    a=a0
    b=b0
    c=c0
    eps = (x_plus-x_minus) * 1e-3
    idx = (x1>(x_minus+eps)) * (x1<(x_plus-eps)) * (x2>(x_minus+eps)) * (x2<(x_plus-eps))
    
    idx = (1-idx)
    if idx.sum()>0:
        if print_info:
            print('fine tuning the lower plane in the third orthant')
        # a[idx],b[idx],c[idx] = train_lower(a0[idx],b0[idx],c0[idx],
        #  x_minus[idx], x_plus[idx], y_minus[idx], y_plus[idx], lr=1e-2,
        #             max_iter = 500,a0_is_valid = a0_is_valid)
        z10 = a*x_minus +b*y_minus+c
        z20 = a*x_minus +b*y_plus+c
        z30 = a*x_plus +b*y_plus+c
        a[idx],b[idx],c[idx] = train_activation_plane.train_lower(z10[idx],z20[idx],z30[idx],
                            x_minus[idx], x_plus[idx], y_minus[idx], y_plus[idx], 
                            qualification_loss_lower, '3l', lr=1e-2,
                            max_iter = 500, print_info = print_info)
    return a.detach(),b.detach(),c.detach()


if __name__ == '__main__':
    
    x_minus = torch.Tensor([-2])
    x_plus = torch.Tensor([-0])
    y_minus = torch.Tensor([-2])
    y_plus = torch.Tensor([-0])
    
    num = 0
    print_info = False

    a_u, b_u, c_u = main_upper(x_minus, x_plus, y_minus, y_plus, print_info = print_info)

    a_l, b_l, c_l = main_lower(x_minus, x_plus, y_minus, y_plus, print_info = print_info)
    
    v1, v2 = plot_2_surface(x_minus[num], x_plus[num],y_minus[num], y_plus[num], 
                 a_l[num],b_l[num],c_l[num],a_u[num], b_u[num], c_u[num])