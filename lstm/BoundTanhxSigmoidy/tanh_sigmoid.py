#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 21:33:27 2019

@author: root
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

def inverse_sigmoid(y):
    #y = 1/(1+exp(-x))
    #exp(-x) = 1/y - 1
    #x = -ln(1/y-1)
    x = -torch.log(1/y-1)
    return x

def sigmoid(x):
    return 1/(1+np.exp(-x))
    
def find_sigmoid(beta,k, eps=1e-4):
    #we find 2 points on function z = beta*sigmoid(y)
    #we also require the two points to be negative
    #that their gradient bound k
    #beta *v* (1-v) = k
    #v = sigmoid(y)
    v = (1-torch.sqrt(1-4*k/beta))/2
    #when k is close to 0, v is close to zeros
    extreme = (v<=1e-2)
    y1 = inverse_sigmoid(v-eps)
    y1[extreme] = -100
    y2 = inverse_sigmoid(v+eps)
    k1 = beta * torch.sigmoid(y1) * (1-torch.sigmoid(y1))
    k1[extreme] = 0
    k2 = beta * torch.sigmoid(y2) * (1-torch.sigmoid(y2))
    return y1,y2,k1,k2 

def get_cross_point(x1,y1,k1,x2,y2,k2):
    #find the cross point (x,y) of 2 lines
    #y = k1*(x-x1) + y1 = k2*(x-x2) + y2
    #(k1-k2) x = k1*x1 - k2*x2 + y2 - y1
     #y1<=y2 and k1<=k2
    if (x1>x2).sum()>0 or (k1>k2).sum()>0:
        # print(beta[between], k[between])
        print(x1-x2, (x1-x2).max())
        print(k1-k2, (k1-k2).max())
        raise Exception('x1 should be lower than x2')
    temp = (k2-k1) < 1e-4
    k2[temp] = k2[temp] + 1e-4
    k1[temp] = k1[temp] - 1e-4
    x = (k1*x1 - k2*x2 + y2 - y1) / (k1-k2)
    y = k1*(x-x1) + y1
    return x,y

def sigmoid_lower(beta, k,b,y_minus,y_plus, plot=False, num=0):
    #in this case y_minus <= y_plus <= 0
    #the function judge whether the line y = kx+b is
    #entirely below the curve y = beta*sigmoid(y), beta>0
    #in the interval y_minus, y_plus
    if (beta<0).sum()>0:
        print(beta)
        raise Exception('beta must be non-negative')
    if (y_plus>0).sum()>0:
        raise Exception('y_plus and y_minus must be non-positive')
    
    loss = torch.zeros(k.shape, device = y_minus.device)
    
    zero = (beta==0)
    
    v_plus = torch.sigmoid(y_plus)
    v_minus = torch.sigmoid(y_minus)
    k_plus = beta * v_plus * (1-v_plus)
    k_minus = beta * v_minus * (1-v_minus)
    
    
    touch_minus = (k<=k_minus) * (1-zero)
    #if loss>0, kx+b is bigger than alpha*tanh(x), we need to lower the loss
    if touch_minus.sum()>0:
        loss[touch_minus] = (k*y_minus + b -  beta*v_minus)[touch_minus]
    
    touch_plus = (k>=k_plus)* (1-zero)
    #if loss>0, kx+b is bigger than alpha*tanh(x), we need to lower the loss
    if touch_plus.sum()>0:
        loss[touch_plus] = (k*y_plus + b -  beta*v_plus)[touch_plus]
    
    between = (k>k_minus) * (k<k_plus)* (1-zero)
    if between.sum()>0:
        y1, y2, k1, k2 = find_sigmoid(beta[between],k[between], eps=1e-3)
        
        v1 = torch.sigmoid(y1) * beta[between]
        v2 = torch.sigmoid(y2) * beta[between]
        
        x0, y0 = get_cross_point(y1,v1,k1,y2,v2,k2)
        loss[between] = k[between]*x0 + b[between] - y0
    
    if zero.sum()>0:
        #there may be a problem here, consider the confidence problem
        loss[zero] = torch.relu(k*y_minus + b) + torch.relu(k*y_plus+b)   
    if plot:
        x = np.linspace(y_minus[num].item(), y_plus[num].item())
        y = beta[num].item() * sigmoid(x)
        h = k[num].item() * x + b[num].item()
        plt.plot(x,y)
        plt.plot(x,h)
    return loss

def sigmoid_upper(beta, k,b,y_minus,y_plus):
    #in this case y_minus <= y_plus <= 0
    #the function judge whether the line y = kx+b is
    #entirely above the curve y = beta*sigmoid(y), where beta<0
    #in the interval y_minus, y_plus
    
    #kx+b>=beta*sigmoid(y) <=> -kx-b <= -beta*sigmoid(y)
    loss = sigmoid_lower(-beta, -k,-b,y_minus,y_plus, plot=False, num=0)
    return loss


def inverse_tanh(x):
    y = torch.log((1+x)/(1-x)) / 2
    return y

def find_tanh(alpha,k, eps=1e-4, positive=True):
    #we find 2 points on function z = alpha*tanh(x)
    #that their gradient bound k
    #alpha (1-u**2) = k
    #u = tanh(x)
    u = torch.sqrt(1-k/alpha)    
    #when k is close to 0, u is close to 1
    extreme = ((1-u)<=1e-2)
    x1 = inverse_tanh(u-eps)
    x2 = inverse_tanh(u+eps)
    x2[extreme] = 100
    k1 = (1-torch.tanh(x1)**2) * alpha
    k2 = (1-torch.tanh(x2)**2) * alpha
    k2[extreme] = 0
    if positive:
        return x1,x2,k1,k2
    else:
        return -x2, -x1, k2, k1
    
def tanh_lower(alpha, k,b,x_minus,x_plus, plot=False, num=0):
    #in this case x_minus <= x_plus <= 0
    #the function judge whether the line y = kx+b is
    #entirely below the curve y = alpha*tanh(x), alpha>=0
    #in the interval x_minus, x_plus
    if (alpha<0).sum()>0:
        raise Exception('alpha must be non-negative')
    if (x_plus>0).sum()>0:
        raise Exception('x_plus and x_minus must be non-positive')
    
    
    loss = torch.zeros(k.shape, device = x_minus.device)
    
    zero = (alpha==0)
    
    u_plus = torch.tanh(x_plus)
    u_minus = torch.tanh(x_minus)
    k_plus = alpha * (1-u_plus**2)
    k_minus = alpha * (1-u_minus**2)
    
    
    touch_minus = (k<=k_minus) * (1-zero)
    #if loss>0, kx+b is bigger than alpha*tanh(x), we need to lower the loss
    if touch_minus.sum()>0:
        loss[touch_minus] = (k*x_minus + b -  alpha*u_minus)[touch_minus]
    
    touch_plus = (k>=k_plus) * (1-zero)
    #if loss>0, kx+b is bigger than alpha*tanh(x), we need to lower the loss
    if touch_plus.sum()>0:
        loss[touch_plus] = (k*x_plus + b -  alpha*u_plus)[touch_plus]
    
    between = (k>k_minus) * (k<k_plus) * (1-zero)
    if between.sum()>0:
        x1, x2, k1, k2 = find_tanh(alpha[between],k[between], eps=1e-3, positive=False)
        u1 = torch.tanh(x1) * alpha[between]
        u2 = torch.tanh(x2) * alpha[between]
        x0, y0 = get_cross_point(x1,u1,k1,x2,u2,k2)
        loss[between] = k[between]*x0 + b[between] - y0
    
    if zero.sum()>0:
        #there may be a problem here, consider the confidence problem
        loss[zero] = torch.relu(k*x_minus + b) + torch.relu(k*x_plus+b)       
    if plot:
        x = np.linspace(x_minus[num].item(), x_plus[num].item())
        y = alpha[num].item() * np.tanh(x)
        h = k[num].item() * x + b[num].item()
        plt.plot(x,y)
        plt.plot(x,h)
    return loss