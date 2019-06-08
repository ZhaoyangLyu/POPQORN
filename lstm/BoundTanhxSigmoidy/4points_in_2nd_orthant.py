#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 14:07:28 2019

@author: root
"""

import torch
import matplotlib.pyplot as plt
import torch.optim as optim
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch.nn.functional as F
from utils import plane, get_volume, plot_surface, plot_2_surface



def qualification_loss_upper_standard(x_minus, x_plus, y_minus, y_plus,
                             a,b,c, confidence=-0.01):
    #abc and fx's are of the same shape, they cound be tensor
    #upper plane need to be larger than the surface at the 4 points
    fx1 = torch.tanh(x_minus) * torch.sigmoid(y_minus)
    fx2 = torch.tanh(x_minus) * torch.sigmoid(y_plus)
    fx3 = torch.tanh(x_plus) * torch.sigmoid(y_plus)
    fx4 = torch.tanh(x_plus) * torch.sigmoid(y_minus)
    
    loss1 = fx1 - (a*x_minus + b*y_minus + c) 
    #if loss1>0, we need to lower it
    #if loss1<0, it satisfies the requirement, we do nothig
    valid = (loss1 <= 0)
    loss1 = torch.clamp(loss1, min = confidence)
    
    loss2 = fx2 - (a*x_minus + b*y_plus + c)
    valid = valid * (loss2<=0)
    loss2 = torch.clamp(loss2, min = confidence)
    
    loss3 = fx3 - (a*x_plus + b*y_plus + c)
    valid = valid * (loss3<=0)
    loss3 = torch.clamp(loss3, min = confidence)
    
    loss4 = fx4 - (a*x_plus + b*y_minus + c)
    valid = valid * (loss4<=0)
    loss4 = torch.clamp(loss4, min = confidence)
    
    loss = loss1 + loss2 + loss3 + loss4
    return loss, valid

def qualification_loss_lower(x, y ,x_minus, x_plus, y_minus, y_plus):
    #this loss restrict x and y be in the range (x_minus, x_plus) (y_minus, y_plus)
    loss1 = torch.relu(x_minus - x)
    valid = (loss1 <= 0)
    loss2 = torch.relu(x - x_plus)
    valid = valid * (loss2<=0)
    loss3 = torch.relu(y_minus - y)
    valid = valid * (loss3<=0)
    loss4 = torch.relu(y - y_plus)
    valid = valid * (loss4<=0)
    loss = loss1 + loss2 + loss3 + loss4
    return loss, valid

def train_lower(x0, y0, x_minus, x_plus, y_minus, y_plus, lr=1e-3, 
                max_iter=100, print_info = True):
    # search over (x,y) \in (x_minus, x_plus) * (y_minus, y_plus) 
    # the plane z = ax + by + c is the tangent plane of the surface tanh(x) sigmoid(y) at point (x,y)
    x = x0.data.clone()
    x.requires_grad = True
    
    y = y0.data.clone()
    y.requires_grad = True
    
    a_best = torch.zeros(x_minus.shape, device=x_minus.device)
    b_best = torch.zeros(x_minus.shape, device=x_minus.device)
    c_best = torch.zeros(x_minus.shape, device=x_minus.device)
    x_best = torch.zeros(x_minus.shape, device=x_minus.device)
    y_best = torch.zeros(x_minus.shape, device=x_minus.device)
    
    optimizer = optim.Adam([x,y], lr=lr)
    cubic = torch.abs((x_plus-x_minus) * (y_plus-y_minus) * 
             torch.tanh(x_minus) * torch.sigmoid(y_plus))
    cubic = torch.clamp(cubic, min=1e-3)
    v_best = -torch.ones(x_minus.shape, device=x_minus.device) * 1000
    #we want to maximize the volume
    for i in range(max_iter):
        q_loss, valid = qualification_loss_lower(x, y ,
                            x_minus, x_plus, y_minus, y_plus)
        a,b,c = plane(x,y)
        v_loss = get_volume(a,b,c,x_minus, x_plus, y_minus, y_plus)/cubic
        
        best = (v_loss > v_best) * valid
        a_best[best] = a[best]
        b_best[best] = b[best]
        c_best[best] = c[best]
        x_best[best] = x[best]
        y_best[best] = y[best]
        v_best[best] = v_loss[best]
        # print('volume', v_loss)
        if print_info:
            print('2l q loss: %.4f volume: %.4f' % (q_loss.mean().item(), v_loss.mean().item()))
        
        loss = q_loss - (valid.float()+0.1) * v_loss
        loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print('x,y',x,y)
    return a_best, b_best, c_best, x_best, y_best

def adjust_lower_plane(a,b,c,x0, y0, x_minus, x_plus, y_minus, y_plus, lr=1e-2, 
                       max_iter=100, print_info = True):
    
    #this function finds the minimum value of tanh(x)*sigmoid(y) - (a*x + b*y + c)
    #in the rectangular area
    # if it is less than zero, we will need to lower the plane z = ax + by + c a little bit
    device = x_minus.device
    
    x0 = x0.detach()
    y0 = y0.detach()
    x1 = ((x0 + x_minus)/2).data.clone()
    y1 = ((y0 + y_minus)/2).data.clone()
    
    x2 = ((x0 + x_minus)/2).data.clone()
    y2 = ((y0 + y_plus)/2).data.clone()
    
    x3 = ((x0 + x_plus)/2).data.clone()
    y3 = ((y0 + y_plus)/2).data.clone()
    
    x4 = ((x0 + x_plus)/2).data.clone()
    y4 = ((y0 + y_minus)/2).data.clone()
    
    x1.requires_grad = True
    y1.requires_grad = True
    x2.requires_grad = True
    y2.requires_grad = True
    x3.requires_grad = True
    y3.requires_grad = True
    x4.requires_grad = True
    y4.requires_grad = True
    
    a = a.detach()
    b = b.detach()
    c = c.detach()
    # a,b,c = plane(x0,y0)
    optimizer = optim.Adam([x1,y1,x2,y2,x3,y3,x4,y4], lr=lr)
    
    x1_best = torch.zeros(x_minus.shape, device=device)
    y1_best = torch.zeros(x_minus.shape, device=device)
    loss1_best = torch.ones(x_minus.shape, device=device) * 1000
    x2_best = torch.zeros(x_minus.shape, device=device)
    y2_best = torch.zeros(x_minus.shape, device=device)
    loss2_best = torch.ones(x_minus.shape, device=device) * 1000
    x3_best = torch.zeros(x_minus.shape, device=device)
    y3_best = torch.zeros(x_minus.shape, device=device)
    loss3_best = torch.ones(x_minus.shape, device=device) * 1000
    x4_best = torch.zeros(x_minus.shape, device=device)
    y4_best = torch.zeros(x_minus.shape, device=device)
    loss4_best = torch.ones(x_minus.shape, device=device) * 1000
    for i in range(max_iter):
        loss1 = torch.tanh(x1)*torch.sigmoid(y1) - (a*x1 + b*y1 + c)
        #we want to find where the lower plane is larger than the surface
        loss2 = torch.tanh(x2)*torch.sigmoid(y2) - (a*x2 + b*y2 + c) 
        loss3 = torch.tanh(x3)*torch.sigmoid(y3) - (a*x3 + b*y3 + c) 
        loss4 = torch.tanh(x4)*torch.sigmoid(y4) - (a*x4 + b*y4 + c) 
        
        qloss1, valid1 = qualification_loss_lower(x1, y1 ,
                                    x_minus, x_plus, y_minus, y_plus)
        best1 = (loss1 < loss1_best) * valid1
        x1_best[best1] = x1[best1]
        y1_best[best1] = y1[best1]
        loss1_best[best1] = loss1[best1]
        
        qloss2, valid2 = qualification_loss_lower(x2, y2 ,
                                    x_minus, x_plus, y_minus, y_plus)
        best2 = (loss2 < loss2_best) * valid2
        x2_best[best2] = x2[best2]
        y2_best[best2] = y2[best2]
        loss2_best[best2] = loss2[best2]
        
        qloss3, valid3 = qualification_loss_lower(x3, y3 ,
                                    x_minus, x_plus, y_minus, y_plus)
        best3 = (loss3 < loss3_best) * valid3
        x3_best[best3] = x3[best3]
        y3_best[best3] = y3[best3]
        loss3_best[best3] = loss3[best3]
        
        qloss4, valid4 = qualification_loss_lower(x4, y4 ,
                                    x_minus, x_plus, y_minus, y_plus)
        best4 = (loss4 < loss4_best) * valid4
        x4_best[best4] = x4[best4]
        y4_best[best4] = y4[best4]
        loss4_best[best4] = loss4[best4]
        
        loss = loss1*(valid1.float()+0.1) + qloss1
        loss = loss + loss2*(valid2.float()+0.1) + qloss2
        loss = loss + loss3*(valid3.float()+0.1) + qloss3
        loss = loss + loss4*(valid4.float()+0.1) + qloss4
        loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if print_info:
            print('2 adjust lower plane loss: %.4f' % loss.item())
    return x1_best,y1_best,x2_best,y2_best,x3_best,y3_best,x4_best,y4_best,loss1_best,loss2_best,loss3_best,loss4_best

def lower_lower_plane(loss1,loss2,loss3,loss4,a,b,c,x_minus, x_plus, y_minus, y_plus):
    # get the minimum values in loss1, loss2, loss3, loss4
    # minimum = 0 if all the lossses are greater than 0
    minimum = torch.zeros(x_minus.shape, device = x_minus.device)
    valid1 = (loss1 < minimum).float()
    minimum = valid1 * loss1 + (1-valid1)*minimum
    
    valid2 = (loss2 < minimum).float()
    minimum = valid2 * loss2 + (1-valid2)*minimum
    
    valid3 = (loss3 < minimum).float()
    minimum = valid3 * loss3 + (1-valid3)*minimum
    
    valid4 = (loss4 < minimum).float()
    minimum = valid4 * loss4 + (1-valid4)*minimum
    
    loss1 = torch.tanh(x_minus)*torch.sigmoid(y_minus) - (a*x_minus + b*y_minus + c) 
    loss2 = torch.tanh(x_minus)*torch.sigmoid(y_plus) - (a*x_minus + b*y_plus + c) 
    loss3 = torch.tanh(x_plus)*torch.sigmoid(y_plus) - (a*x_plus + b*y_plus + c) 
    loss4 = torch.tanh(x_plus)*torch.sigmoid(y_minus) - (a*x_plus + b*y_minus + c) 
    
    valid1 = (loss1 < minimum).float()
    minimum = valid1 * loss1 + (1-valid1)*minimum
    
    valid2 = (loss2 < minimum).float()
    minimum = valid2 * loss2 + (1-valid2)*minimum
    
    valid3 = (loss3 < minimum).float()
    minimum = valid3 * loss3 + (1-valid3)*minimum
    
    valid4 = (loss4 < minimum).float()
    minimum = valid4 * loss4 + (1-valid4)*minimum

    # minimum<=0
    return minimum

def lower_plane(a,b,c,x0,y0,x_minus, x_plus, y_minus, y_plus, print_info = True):
    x1,y1,x2,y2,x3,y3,x4,y4,loss1,loss2,loss3,loss4 = adjust_lower_plane(
            a,b,c,x0, y0, x_minus, x_plus, y_minus, y_plus, lr=1e-2, max_iter=500, print_info=print_info)
    increase = lower_lower_plane(loss1,loss2,loss3,loss4,
                                 a,b,c,
                                 x_minus, x_plus, y_minus, y_plus)
    return increase * 1.01

def main_lower(x_minus, x_plus, y_minus, y_plus, print_info = True):
    x0 = (x_minus + x_plus)/2
    y0 = (y_minus + y_plus)/2
    a_lower, b_lower, c_lower, x_lower, y_lower = train_lower(x0, y0, 
                x_minus, x_plus, y_minus, y_plus, lr=1e-2, 
                max_iter=500, print_info = print_info)
    
    x1,y1,x2,y2,x3,y3,x4,y4,loss1,loss2,loss3,loss4 = adjust_lower_plane(a_lower,b_lower,c_lower,
            x_lower, y_lower, x_minus, x_plus, y_minus, y_plus, lr=1e-2, max_iter=200, print_info=print_info)
    increase = lower_lower_plane(loss1,loss2,loss3,loss4,
                                 a_lower,b_lower,c_lower,
                                 x_minus, x_plus, y_minus, y_plus)
    #increase is actually negative
    c_lower = c_lower + increase
    return a_lower.detach(), b_lower.detach(), c_lower.detach()

import train_activation_plane
def main_upper(x_minus, x_plus, y_minus, y_plus, print_info = True):
    z10 = torch.tanh(x_plus) * torch.sigmoid(y_minus)
    z20 = torch.tanh(x_plus) * torch.sigmoid(y_minus)
    z30 = torch.tanh(x_plus) * torch.sigmoid(y_minus)
    a_upper,b_upper,c_upper = train_activation_plane.train_upper(z10,z20,z30,
                x_minus, x_plus, y_minus, y_plus, 
                qualification_loss_upper_standard, 
                '2u', lr=1e-2,
                max_iter = 500, print_info = print_info)
    return a_upper.detach(),b_upper.detach(),c_upper.detach()

if __name__ ==  '__main__':
    x_minus = torch.Tensor([-5.2])
    x_plus = torch.Tensor([-0.1])
    y_minus = torch.Tensor([0.1])
    y_plus = torch.Tensor([5.2])
    num = 0
    
    print_info = False
    
    a_lower, b_lower, c_lower = main_lower(x_minus, x_plus, y_minus, y_plus, print_info = print_info)

    a_upper, b_upper, c_upper = main_upper(x_minus, x_plus, y_minus, y_plus, print_info = print_info)

    v1, v2 = plot_2_surface(x_minus[num], x_plus[num],y_minus[num], y_plus[num], 
                 a_lower[num],b_lower[num],c_lower[num],a_upper[num], b_upper[num], c_upper[num])