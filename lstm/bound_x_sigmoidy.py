#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 13:02:22 2019

@author: root
"""
import sys
sys.path.append('BoundTanhxSigmoidy')

import torch
import matplotlib.pyplot as plt
import torch.optim as optim
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch.nn.functional as F
import time

def sigmoid(x):
    #numpy operation
    return 1/(1+np.exp(-x))

def get_volume(a,b,c,x_minus, x_plus, y_minus, y_plus):
    #a,b,c,x_minus, x_plus, y_minus, y_plus could be any shape
    #they should be of the same shape
    #v is the same shape of the input
    v = 0.5 * (x_plus**2 - x_minus**2)*(y_plus - y_minus) * a
    v = v+0.5*(y_plus**2 - y_minus**2)*(x_plus - x_minus) * b
    v = v + (x_plus-x_minus)*(y_plus-y_minus) * c
    return v   


def plot_surface(x_minus, x_plus,y_minus, y_plus, a,b,c):
    x_minus, x_plus,y_minus, y_plus = x_minus.item(), x_plus.item(), y_minus.item(), y_plus.item()
    
    n = 20
    x = np.linspace(x_minus,x_plus,n)
    y = np.linspace(y_minus,y_plus,n)
    X, Y = np.meshgrid(x, y)
    Z = X*sigmoid(Y)
    
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(X, Y, Z)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    a,b,c = a.item(), b.item(), c.item()
    H = a*X + b*Y + c
    ax.plot_surface(X, Y, H)
    plt.show()
    return H-Z

def plot_2_surface(x_minus, x_plus,y_minus, y_plus, a1,b1,c1,
                 a2,b2,c2, plot=True, num_points=20):
    #a1,b1,c1 should be the upper plane
    #a2,b2,c2 should be the lower plane
    x_minus, x_plus,y_minus, y_plus = x_minus.item(), x_plus.item(), y_minus.item(), y_plus.item()
    
    n = num_points
    x = np.linspace(x_minus,x_plus,n)
    y = np.linspace(y_minus,y_plus,n)
    X, Y = np.meshgrid(x, y)
    Z = X*sigmoid(Y)
    
    if plot:
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_surface(X, Y, Z)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    
    a1,b1,c1 = a1.item(), b1.item(), c1.item()
    H1 = a1*X + b1*Y + c1
    
    if plot:
        ax.plot_surface(X, Y, H1)
    
    a2,b2,c2 = a2.item(), b2.item(), c2.item()
    H2 = a2*X + b2*Y + c2
    if plot:
        ax.plot_surface(X, Y, H2)
        plt.show()
    return H1-Z, H2-Z

def inverse_sigmoid(y):
    #y = 1/(1+exp(-x))
    #exp(-x) = 1/y - 1
    #x = -ln(1/y-1)
    x = -torch.log(1/y-1)
    return x
    
def find_sigmoid(beta,k, eps=1e-4, upper=False):
    #we find 2 points on function z = beta*sigmoid(y), beta>0
    #we also require the two points to be negative if upper=False
    #we require the two points to be positive if upper=True
    #that their gradient bound k
    #beta *v* (1-v) = k
    #v = sigmoid(y)
    if not upper:
        v = (1-torch.sqrt(1-4*k/beta))/2
        #when k is close to 0, v is close to zeros
        extreme = (v<=1e-2)
        y1 = inverse_sigmoid(v-eps)
        y1[extreme] = -100
        y2 = inverse_sigmoid(v+eps)
        k1 = beta * torch.sigmoid(y1) * (1-torch.sigmoid(y1))
        k1[extreme] = 0
        k2 = beta * torch.sigmoid(y2) * (1-torch.sigmoid(y2))
        
    else:
        v = (1+torch.sqrt(1-4*k/beta))/2
        #when k is close to 0, v is close to 1
        extreme = (v>=(1-1e-2))
        y1 = inverse_sigmoid(v-eps)
        y2 = inverse_sigmoid(v+eps)
        y2[extreme] = 100
        k1 = beta * torch.sigmoid(y1) * (1-torch.sigmoid(y1))
        k2 = beta * torch.sigmoid(y2) * (1-torch.sigmoid(y2))
        k2[extreme] = 0
    return y1,y2,k1,k2 
        

def get_cross_point(x1,y1,k1,x2,y2,k2, upper=False):
    #find the cross point (x,y) of 2 lines
    #y = k1*(x-x1) + y1 = k2*(x-x2) + y2
    #(k1-k2) x = k1*x1 - k2*x2 + y2 - y1
    if not upper: #x1<=x2, k1<=k2
        if (x1>x2).sum()>0 or (k1>k2).sum()>0:
            # print(beta[between], k[between])
            print(x1-x2, (x1-x2).max())
            print(k1-k2, (k1-k2).max())
            raise Exception('x1 should be lower than x2')
        temp = (k2-k1) < 1e-4
        k2[temp] = k2[temp] + 1e-4
        k1[temp] = k1[temp] - 1e-4
    else: #x1<x2, k1>k2
        if (x1>x2).sum()>0 or (k1<k2).sum()>0:
            # print(beta[between], k[between])
            print(x1-x2, (x1-x2).max())
            print(k2-k1, (k2-k1).max())
            raise Exception('x1 should be lower than x2')
        temp = (k1-k2) < 1e-4
        k2[temp] = k2[temp] - 1e-4
        k1[temp] = k1[temp] + 1e-4
        
    x = (k1*x1 - k2*x2 + y2 - y1) / (k1-k2)
    y = k1*(x-x1) + y1
    return x,y

def sigmoid_lower(beta, k,b,y_minus,y_plus, plot=False, num=0):
    #in this case y_minus <= y_plus <= 0
    #the function judge whether the line y = kx+b is
    #entirely below the curve y = beta*sigmoid(y)
    #in the interval y_minus, y_plus
    loss = torch.zeros(k.shape, device=y_minus.device)
    v_plus = torch.sigmoid(y_plus)
    v_minus = torch.sigmoid(y_minus)
    k_plus = beta * v_plus * (1-v_plus)
    k_minus = beta * v_minus * (1-v_minus)
    
    
    touch_minus = (k<=k_minus)
    #if loss>0, kx+b is bigger than alpha*tanh(x), we need to lower the loss
    if touch_minus.sum()>0:
        loss[touch_minus] = (k*y_minus + b -  beta*v_minus)[touch_minus]
    
    touch_plus = (k>=k_plus)
    #if loss>0, kx+b is bigger than alpha*tanh(x), we need to lower the loss
    if touch_plus.sum()>0:
        loss[touch_plus] = (k*y_plus + b -  beta*v_plus)[touch_plus]
    
    between = (k>k_minus) * (k<k_plus)
    if between.sum()>0:
        y1, y2, k1, k2 = find_sigmoid(beta[between],k[between], eps=1e-3, upper=False)
        v1 = torch.sigmoid(y1) * beta[between]
        v2 = torch.sigmoid(y2) * beta[between]
        x0, y0 = get_cross_point(y1,v1,k1,y2,v2,k2, upper=False)
        loss[between] = k[between]*x0 + b[between] - y0
        
    if plot:
        x = np.linspace(y_minus[num].item(), y_plus[num].item())
        y = beta[num].item() * sigmoid(x)
        h = k[num].item() * x + b[num].item()
        plt.plot(x,y)
        plt.plot(x,h)
    return loss

def sigmoid_upper(beta, k,b,y_minus,y_plus, plot=False, num=0):
    #in this case 0<=y_minus <= y_plus
    #the function judge whether the line y = kx+b is
    #entirely above the curve y = beta*sigmoid(y), beta>0
    #in the interval y_minus, y_plus
    loss = torch.zeros(k.shape, device=y_minus.device)
    v_plus = torch.sigmoid(y_plus)
    v_minus = torch.sigmoid(y_minus)
    k_plus = beta * v_minus * (1-v_minus)
    k_minus = beta * v_plus * (1-v_plus)
    
    
    touch_minus = (k>=k_plus)
    #if loss>0, kx+b is bigger than alpha*tanh(x), we need to lower the loss
    if touch_minus.sum()>0:
        loss[touch_minus] = (beta*v_minus - (k*y_minus + b))[touch_minus]
    
    touch_plus = (k<=k_minus)
    #if loss>0, kx+b is bigger than alpha*tanh(x), we need to lower the loss
    if touch_plus.sum()>0:
        loss[touch_plus] = (beta*v_plus - (k*y_plus + b))[touch_plus]
    
    between = (k>k_minus) * (k<k_plus)
    if between.sum()>0:
        y1, y2, k1, k2 = find_sigmoid(beta[between],k[between], eps=1e-3, upper=True)
        v1 = torch.sigmoid(y1) * beta[between]
        v2 = torch.sigmoid(y2) * beta[between]
        x0, y0 = get_cross_point(y1,v1,k1,y2,v2,k2, upper=True)
        loss[between] = y0 - (k[between]*x0 + b[between])
        
    if plot:
        x = np.linspace(y_minus[num].item(), y_plus[num].item())
        y = beta[num].item() * sigmoid(x)
        h = k[num].item() * x + b[num].item()
        plt.plot(x,y)
        plt.plot(x,h)
    return loss

def sigmoid_lower_positive(beta, k,b,y_minus,y_plus, plot=False, num=0,
                          confidence = -0.1):
    #in this case y_minus and y_plus can be any value
    #the function judge whether the line y = ky+b is
    #entirely below the curve y = beta*sigmoid(y), beta >0
    #in the interval y_minus, y_plus
    loss = torch.zeros(y_minus.shape, device=y_minus.device)
    valid = (loss>0)
    case1 = (y_plus <=0)
    if case1.sum()>0:
        loss_temp = sigmoid_lower(beta[case1], k[case1],b[case1],
                             y_minus[case1],y_plus[case1], plot=False, num=0)
        valid[case1] = (loss_temp<=0)
        loss_temp = torch.clamp(loss_temp, min=confidence)
        loss[case1] = loss_temp
        
        
    case2 = (y_minus>=0)
    if case2.sum()>0:
        loss1 = (k*y_minus + b - beta*torch.sigmoid(y_minus))[case2]
        valid[case2] = (loss1<=0)
        loss1 = torch.clamp(loss1, min=confidence)
        loss2 = (k*y_plus + b - beta*torch.sigmoid(y_plus))[case2]
        valid[case2] = valid[case2] * (loss2<=0)
        loss2 = torch.clamp(loss2, min=confidence)
        loss[case2] = loss1 + loss2
        
    case3 = (y_plus>0) * (y_minus<0)
    if case3.sum()>0:
        loss1 = sigmoid_lower(beta[case3], k[case3],b[case3],
                             y_minus[case3],y_plus[case3]*0, plot=False, num=0)
        valid[case3] = (loss1<=0)
        loss1 = torch.clamp(loss1, min=confidence)
        loss2 = (k*y_plus + b - beta*torch.sigmoid(y_plus))[case3]
        valid[case3] = valid[case3] * (loss2<=0)
        loss2 = torch.clamp(loss2, min=confidence)
        loss[case3] = loss1 + loss2
    
    if plot:
        x = np.linspace(y_minus[num].item(), y_plus[num].item())
        y = beta[num].item() * sigmoid(x)
        h = k[num].item() * x + b[num].item()
        plt.plot(x,y)
        plt.plot(x,h)
    return loss, valid

def sigmoid_lower_negative(beta, k,b,y_minus,y_plus, plot=False, num=0,
                          confidence = -0.1):
    #in this case y_minus and y_plus can be any value
    #the function judge whether the line y = ky+b is
    #entirely below the curve y = beta*sigmoid(y), beta <0
    #in the interval y_minus, y_plus
    loss = torch.zeros(y_minus.shape, device=y_minus.device)
    valid = (loss>0)
    
    case1 = (y_minus >=0)
    if case1.sum()>0:
        loss_temp = sigmoid_upper(-beta[case1], -k[case1],-b[case1],
                             y_minus[case1],y_plus[case1], plot=False, num=0)
        valid[case1] = (loss_temp<=0)
        loss_temp = torch.clamp(loss_temp, min=confidence)
        loss[case1] = loss_temp
        
        
    case2 = (y_plus <= 0)
    if case2.sum()>0:
        loss1 = (k*y_minus + b - beta*torch.sigmoid(y_minus))[case2]
        valid[case2] = (loss1<=0)
        loss1 = torch.clamp(loss1, min=confidence)
        loss2 = (k*y_plus + b - beta*torch.sigmoid(y_plus))[case2]
        valid[case2] = valid[case2] * (loss2<=0)
        loss2 = torch.clamp(loss2, min=confidence)
        loss[case2] = loss1 + loss2
        
    case3 = (y_plus>0) * (y_minus<0)
    if case3.sum()>0:
        loss1 = sigmoid_upper(-beta[case3], -k[case3],-b[case3],
                             y_minus[case3]*0,y_plus[case3], plot=False, num=0)
        valid[case3] = (loss1<=0)
        loss1 = torch.clamp(loss1, min=confidence)
        loss2 = (k*y_minus + b - beta*torch.sigmoid(y_minus))[case3]
        valid[case3] = valid[case3] * (loss2<=0)
        loss2 = torch.clamp(loss2, min=confidence)
        loss[case3] = loss1 + loss2
    
    if plot:
        x = np.linspace(y_minus[num].item(), y_plus[num].item())
        y = beta[num].item() * sigmoid(x)
        h = k[num].item() * x + b[num].item()
        plt.plot(x,y)
        plt.plot(x,h)
    return loss, valid

def sigmoid_lower_general(beta, k,b,y_minus,y_plus, plot=False, num=0,
                          confidence = -0.1):
    #in this case beta, y_minus and y_plus can be any value
    #the function judge whether the line y = ky+b is
    #entirely below the curve y = beta*sigmoid(y),
    #in the interval y_minus, y_plus
    loss = torch.zeros(y_minus.shape, device=y_minus.device)
    valid = (loss>0)
    
    positive = (beta>0)
    if positive.sum()>0:
        loss_temp, valid_temp = sigmoid_lower_positive(beta[positive], k[positive],
                            b[positive],y_minus[positive],y_plus[positive], plot=plot, num=num,
                              confidence = confidence)
        loss[positive] = loss_temp
        valid[positive] = valid_temp
    
    negative = (beta<0)
    if negative.sum()>0:
        loss_temp, valid_temp = sigmoid_lower_negative(beta[negative], k[negative],
                            b[negative],y_minus[negative],y_plus[negative], plot=plot, num=num,
                              confidence = confidence)
        loss[negative] = loss_temp
        valid[negative] = valid_temp
    
    zero = (beta==0)
    if zero.sum()>0:
        loss1 = (k*y_minus + b)[zero]
        valid[zero] = (loss1<=0)
        loss2 = (k*y_plus + b)[zero]
        valid[zero] = valid[zero] * (loss2<=0)
        loss[zero] = torch.clamp(loss1, min=confidence) + torch.clamp(loss2, min=confidence)
    
    return loss, valid

def qualification_loss(x_minus, x_plus, y_minus, y_plus, a,b,c,confidence = -0.1):
    #we need the plane z = ax + by + c is lower than the surface at
    #x1-x2 and x3-x4
    
    #on x1-x2, z = b*y + a*x_minus + c
    #          f = x_minus * sigmoid(y)
    loss1, valid1 = sigmoid_lower_general(x_minus, b,a*x_minus + c,
                        y_minus,y_plus, plot=False, num=0, confidence = confidence)
    
    #on x3-x4, z = b*y + a*x_plus + c
    #          f = x_plus * sigmoid(y)
    loss2, valid2 = sigmoid_lower_general(x_plus, b,a*x_plus + c,
                        y_minus,y_plus, plot=False, num=0, confidence = confidence)
    loss = loss1 + loss2
    valid = valid1 * valid2
    return loss, valid


def find_extreme(x_minus, x_plus, y_minus, y_plus):
    z1 = x_minus * torch.sigmoid(y_minus)
    z2 = x_minus * torch.sigmoid(y_plus)
    z3 = x_plus * torch.sigmoid(y_minus)
    z4 = x_plus * torch.sigmoid(y_plus)
    
    z_min = z1.data.clone()
    
    idx = (z2<z_min)
    z_min[idx] = z2[idx]
    
    idx = (z3<z_min)
    z_min[idx] = z3[idx]
    
    idx = (z3<z_min)
    z_min[idx] = z4[idx]
    
    
    z_max = z1.data.clone()
    
    idx = (z2>z_max)
    z_max[idx] = z2[idx]
    
    idx = (z3>z_max)
    z_max[idx] = z3[idx]
    
    idx = (z3>z_max)
     
    z_max[idx] = z4[idx]
    return z_min, z_max

import train_activation_plane
def main_lower(x_minus, x_plus, y_minus, y_plus, print_info = True):
    # a0 = torch.zeros(x_minus.shape, device=x_minus.device)
    # b0 = torch.zeros(x_minus.shape, device=x_minus.device)
    c0,_ = find_extreme(x_minus, x_plus, y_minus, y_plus)
    z10 = c0.data.clone()
    z20 = c0.data.clone()
    z30 = c0.data.clone()
    # a,b,c = train_lower(a0,b0,c0,x_minus, x_plus, y_minus, y_plus, lr=1e-2,
    #             max_iter = 500)
    a,b,c = train_activation_plane.train_lower(z10,z20,z30,
                x_minus, x_plus, y_minus, y_plus, qualification_loss, 
                'x sigmoid', lr=1e-2,
                max_iter = 500, print_info = print_info)
    return a.detach(),b.detach(),c.detach()

def main_upper(x_minus, x_plus, y_minus, y_plus, print_info = True):
    if print_info:
        print('x sigmoid upper: using x sigmoid lower function')
    a,b,c = main_lower(-x_plus, -x_minus, y_minus, y_plus, print_info = print_info)
    b = -b
    c = -c
    return a.detach(),b.detach(),c.detach()

from use_1D_line_bound_2D_activation import line_bounding_2D_activation
from use_constant_bound_2D_activation import constant_bounding_2D_activation

def main(x_minus, x_plus, y_minus, y_plus, fine_tune_c=True,
         use_1D_line=False, use_constant=False, print_info = True):
    if (x_minus>x_plus).sum()>0 or (y_minus>y_plus).sum()>0:
        print(x_minus-x_plus, (x_minus-x_plus).max())
        print(y_minus-y_plus, (y_minus-y_plus).max())
        raise Exception('x_plus must be strictly larger than x_minus and y_plus must be strictly larger than y_minus')
    
    if use_1D_line:
        a_l,b_l,c_l,a_u,b_u,c_u = line_bounding_2D_activation(
                x_minus, x_plus, y_minus, y_plus, tanh=False)
       
        return a_l,b_l,c_l,a_u,b_u,c_u
    if use_constant:
        a_l,b_l,c_l,a_u,b_u,c_u = constant_bounding_2D_activation(
                x_minus, x_plus, y_minus, y_plus, tanh=False)
        
        return a_l,b_l,c_l,a_u,b_u,c_u
    
    temp = (x_plus-x_minus)<1e-3
    x_plus[temp] = x_plus[temp] + 1e-3
    x_minus[temp] = x_minus[temp] - 1e-3
    
    temp = (y_plus-y_minus)<1e-3
    y_plus[temp] = y_plus[temp] + 1e-3
    y_minus[temp] = y_minus[temp] - 1e-3
    
    
    a_l,b_l,c_l = main_lower(x_minus, x_plus, y_minus, y_plus, print_info = print_info)
    a_u,b_u,c_u = main_upper(x_minus, x_plus, y_minus, y_plus, print_info = print_info)
    if fine_tune_c:
        c_l, c_u = validate(a_l,b_l,c_l,a_u,b_u,c_u,x_minus, x_plus, y_minus, y_plus, verify_and_modify_all = True,
                 max_iter=100, plot=False, eps=1e8, print_info = print_info)
    return a_l.detach(),b_l.detach(),c_l.detach(),a_u.detach(),b_u.detach(),c_u.detach()



def validate(a_l,b_l,c_l,a_u,b_u,c_u,x_minus, x_plus, y_minus, y_plus, verify_and_modify_all = False,
             max_iter=100, plot=False, eps=1e-5, print_info = True):
    # eps =1e-5
    original_shape = c_l.shape
    
    a_l_new = a_l.view(-1)#.data.clone()
    b_l_new = b_l.view(-1)#.data.clone()
    c_l_new = c_l.view(-1)#.data.clone()
    
    a_u_new = a_u.view(-1)#.data.clone()
    b_u_new = b_u.view(-1)#.data.clone()
    c_u_new = c_u.view(-1)#.data.clone()
    
    x_minus_new = x_minus.view(-1)#.data.clone()
    x_plus_new = x_plus.view(-1)#.data.clone()
    y_minus_new = y_minus.view(-1)#.data.clone()
    y_plus_new = y_plus.view(-1)#.data.clone()
    
    N = a_l_new.size(0)
    
    if verify_and_modify_all:
        max_iter = N

    for i in range(max_iter):
        
        if verify_and_modify_all:
            n = i
        else:
            n = torch.randint(0,N,[1])
            n = n.long()
        
        hl_fl, hu_fu =  plot_2_surface(x_minus_new[n], x_plus_new[n],y_minus_new[n], y_plus_new[n], 
                                       a_l_new[n],b_l_new[n],c_l_new[n],
                                       a_u_new[n],b_u_new[n],c_u_new[n], plot=plot)
        
        # print('hl-fl max', hl_fl.max())
        # print('hu-fu min', hu_fu.min())
        if print_info:
            print('x sigmoid iter: %d num: %d hl-fl max %.6f mean %.6f hu-fu min %.6f mean %.6f' % 
                (i,n, hl_fl.max(), hl_fl.mean(), hu_fu.min(), hu_fu.mean()))
        if hl_fl.max() > eps: #we want hl_fl.max() < 0
            print(x_minus_new[n], x_plus_new[n],y_minus_new[n], y_plus_new[n], 
                                       a_l_new[n],b_l_new[n],c_l_new[n],
                                       a_u_new[n],b_u_new[n],c_u_new[n])
            plot_surface(x_minus_new[n], x_plus_new[n],y_minus_new[n], y_plus_new[n], 
                                       a_l_new[n],b_l_new[n],c_l_new[n])
            print('hl-fl max',hl_fl.max())
            raise Exception('lower plane fail')
            break
        
        if hl_fl.max()>0 and verify_and_modify_all:
            c_l_new[n] = c_l_new[n] - hl_fl.max() * 2
            
        if hu_fu.min() < -eps: # we want hu_fu.min()>0
            print(x_minus_new[n], x_plus_new[n],y_minus_new[n], y_plus_new[n], 
                                       a_l_new[n],b_l_new[n],c_l_new[n],
                                       a_u_new[n],b_u_new[n],c_u_new[n])
            plot_surface(x_minus_new[n], x_plus_new[n],y_minus_new[n], y_plus_new[n], 
                                       a_u_new[n],b_u_new[n],c_u_new[n])
            print('hu-fu min',hu_fu.min())
            raise Exception('upper plane fail')
            break
        if hu_fu.min()<0 and verify_and_modify_all:
            c_u_new[n] = c_u_new[n] - hu_fu.min() * 2

    c_l_new = c_l_new.view(original_shape)
    c_u_new = c_u_new.view(original_shape)
    return c_l_new, c_u_new


if __name__ == '__main__':
    
    x_minus = torch.Tensor([-2])
    x_plus = torch.Tensor([2])
    y_minus = torch.Tensor([-5])
    y_plus = torch.Tensor([5])
    
    # num = [1000]
    # device = torch.device('cuda:0')
    # device = torch.device('cpu')
    # x_minus = (torch.rand(num, device=device) - 0.5) * 10
    # x_plus = torch.rand(num, device=device)*5 + x_minus
    # y_minus = (torch.rand(num, device=device)-0.5) * 10
    # y_plus = torch.rand(num, device=device)*5 + y_minus
    print_info = False
    start = time.time()
    a_l,b_l,c_l,a_u,b_u,c_u = main(x_minus, x_plus, y_minus, y_plus,
                                   fine_tune_c=True,
                                   use_1D_line=False, use_constant=False, print_info = print_info)
    plot_2_surface(x_minus, x_plus,y_minus, y_plus, a_l,b_l,c_l,
                  a_u,b_u,c_u, plot=True, num_points=20)
    end = time.time()
    print('time used: %.4f ' % (end-start))
    validate(a_l,b_l,c_l,a_u,b_u,c_u,x_minus, x_plus, y_minus, y_plus, 
              max_iter=1000, eps=1e-5, print_info = print_info)
    print('time used: %.4f ' % (end-start))
    # a0 = torch.Tensor([0])
    # b0 = torch.Tensor([0])
    # c0 = find_minimum(x_minus, x_plus, y_minus, y_plus)
    
    # a,b,c = train_lower(a0,b0,c0,x_minus, x_plus, y_minus, y_plus, lr=1e-1,
    #             max_iter = 500)
    # plot_surface(x_minus, x_plus,y_minus, y_plus, a,b,c)
    
    # loss,valid = qualification_loss(x_minus, x_plus, y_minus, y_plus, 
    #                                 a,b,c,confidence = -0)
    
    # a_u,b_u,c_u = main_upper(x_minus, x_plus, y_minus, y_plus)
    # plot_surface(x_minus, x_plus,y_minus, y_plus, a_u,b_u,c_u)
    # loss1, valid1 = sigmoid_lower_general(torch.tanh(x_minus), b,a*x_minus + c,
                        # y_minus,y_plus, plot=True, num=0, confidence = 0)
    
    
    
    