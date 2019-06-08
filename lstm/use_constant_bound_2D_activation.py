#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:48:45 2019

@author: zhaoyang
"""

import torch
from utils import plane, get_volume, plot_surface, plot_2_surface
from get_bound_for_general_activation_function import getConvenientGeneralActivationBound


def tanh_sigmoid(x,y):
    z = torch.tanh(x)*torch.sigmoid(y) 
    return z

def x_sigmoid(x,y):
    z = (x)*torch.sigmoid(y) 
    return z

def constant_bounding_2D_activation(x_minus, x_plus,y_minus, y_plus, tanh=True):
    #if tanh is True, bound tanh(x) sigmoid(y)
    #else, bound x sigmoid(y)
    
    if tanh:
        func = tanh_sigmoid
    else:
        func = x_sigmoid
        
    z1 = func(x_minus, y_minus)
    z2 = func(x_minus, y_plus)
    z3 = func(x_plus, y_minus)
    z4 = func(x_plus, y_plus)
    
    gamma_l = torch.min(torch.min(torch.min(z1, z2), z3), z4)
    gamma_u = torch.max(torch.max(torch.max(z1, z2), z3), z4)
    
    alpha_l = torch.zeros(x_minus.shape, device=x_minus.device)
    beta_l = torch.zeros(x_minus.shape, device=x_minus.device)
    # gamma_l = I_l * X_l * bl + (1-I_l) * X_l * bu
    
    #tanh(x)sigmoid(y) <= X_u*k_u y + X_u*b_u, when X_u>=0
    #tanh(x)sigmoid(y) <= X_u*k_l y + X_u*b_l, when X_u<0
    
    alpha_u = torch.zeros(x_plus.shape, device=x_minus.device)
    beta_u = torch.zeros(x_plus.shape, device=x_minus.device)
    # gamma_u = I_u * X_u * bu + (1-I_u) * X_u * bl
    return alpha_l,beta_l,gamma_l, alpha_u,beta_u,gamma_u

if __name__ == '__main__':
    #bound tanh(x) sigmoid(y)
    x_minus = torch.rand(2,3) - 0.5
    x_plus = x_minus + 0.5 
    
    y_minus = torch.rand(2,3) - 0.5
    y_plus = y_minus + 0.5 
    # x_minus = torch.Tensor([0.5])
    # x_plus = torch.Tensor([1])
    
    # y_minus = torch.Tensor([-1])
    # y_plus = torch.Tensor([1])
    
    kl, bl, ku, bu = getConvenientGeneralActivationBound(y_minus,
                                                            y_plus, 'sigmoid')
    X_l = torch.tanh(x_minus)
    X_u = torch.tanh(x_plus)
    
    I_l = (X_l>=0).float()
    I_u = (X_u>=0).float()
    
    #k_l y + b_l <= sigmoid(y) <= k_u y + b_u
    #X_l*k_l y + X_l*b_l <= tanh(x)sigmoid(y), when X_l>=0
    #X_l*k_u y + X_l*b_u <= tanh(x)sigmoid(y), when X_l<0
    
    alpha_l = torch.zeros(x_minus.shape, device=x_minus.device)
    beta_l = I_l * X_l * kl + (1-I_l) * X_l * ku
    gamma_l = I_l * X_l * bl + (1-I_l) * X_l * bu
    
    #tanh(x)sigmoid(y) <= X_u*k_u y + X_u*b_u, when X_u>=0
    #tanh(x)sigmoid(y) <= X_u*k_l y + X_u*b_l, when X_u<0
    
    alpha_u = torch.zeros(x_plus.shape, device=x_minus.device)
    beta_u = I_u * X_u * ku + (1-I_u) * X_u * kl
    gamma_u = I_u * X_u * bu + (1-I_u) * X_u * bl
    
    idx= (0,0)
    
    # plot_surface(x_minus, x_plus,y_minus, y_plus, alpha_l,beta_l,gamma_l)
    plot_2_surface(x_minus[idx], x_plus[idx],y_minus[idx], y_plus[idx], 
                   alpha_l[idx],beta_l[idx],gamma_l[idx],
                 alpha_u[idx],beta_u[idx],gamma_u[idx], plot=True, num_points=20)
    
    
    
    
    
    
    