#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 22:03:58 2019

@author: root
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from utils import plane, get_volume, plot_surface, plot_2_surface
import tanh_sigmoid as ts
#big idea
#z=ax+by+c should be bigger than the surface
#at A-x1, C-x4, x2, x3, they we search over a,b,c using geadient descent


def qualification_loss(x_minus, x_plus, y_minus, y_plus,a, b ,c, confidence=-0.1):
    #the function determines wheather z=ax+by+c is bigger than the surface
    #at A-x1, C-x4, x2, x3
    
    #at A-x1: h = by + a*x_minus + c
    #         f = sigmoid(y) * tanh(x_minus)
    loss1 = ts.sigmoid_upper(torch.tanh(x_minus), b,
                          a*x_minus + c,y_minus,y_plus*0)
    valid = (loss1 <= 0)
    loss1 = torch.clamp(loss1, min=confidence)
    
    #at C-x4: h = by + a*x_plus + c
    #         f = sigmoid(y) * tanh(x_plus)
    loss2 = ts.sigmoid_upper(torch.tanh(x_plus), b,
                          a*x_plus + c,y_minus,y_plus*0)
    valid = valid * (loss2 <= 0)
    loss2 = torch.clamp(loss2, min=confidence)
    
    #at x2(x_minus,y_plus)
    loss4 = torch.tanh(x_minus)*torch.sigmoid(y_plus) - (a*x_minus + b*y_plus + c)
    valid = valid * (loss4 <= 0)
    loss4 = torch.clamp(loss4, min=confidence)
    
    #at x3(x_plus, y_plus)
    loss5 = torch.tanh(x_plus)*torch.sigmoid(y_plus) - (a*x_plus + b*y_plus + c)
    valid = valid * (loss5 <= 0)
    loss5 = torch.clamp(loss5, min=confidence)
    
    loss = loss1 + loss2 + loss4 + loss5
    return loss, valid


import train_activation_plane
def main_upper(x_minus, x_plus, y_minus, y_plus, print_info = True):
    z10 = torch.tanh(x_plus) * torch.sigmoid(y_minus)
    z20 = torch.tanh(x_plus) * torch.sigmoid(y_minus)
    z30 = torch.tanh(x_plus) * torch.sigmoid(y_minus)
    a,b,c = train_activation_plane.train_upper(z10,z20,z30,
                x_minus, x_plus, y_minus, y_plus, qualification_loss, 
                '23u', lr=1e-2,
                max_iter = 500, print_info = print_info)
    return a.detach(),b.detach(),c.detach()



if __name__ == '__main__':
    
    x_minus = torch.Tensor([-2])
    x_plus = torch.Tensor([-0.1])
    y_minus = torch.Tensor([-2])
    y_plus = torch.Tensor([2])
    
    num = 0
    a_u, b_u, c_u = main_upper(x_minus, x_plus, y_minus, y_plus, print_info = False)
    
    plot_surface(x_minus[num], x_plus[num],y_minus[num], y_plus[num],
                                a_u[num],b_u[num],c_u[num])