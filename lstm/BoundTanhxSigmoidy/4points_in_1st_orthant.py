#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 09:46:06 2018

@author: root
"""

import torch
import matplotlib.pyplot as plt
import torch.optim as optim
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch.nn.functional as F
from utils import plane, get_volume, plot_surface, plot_2_surface



def qualification_loss_lower_standard(x_minus, x_plus, y_minus, y_plus,
                                      a,b,c,confidence=-0.01):
    #abc and fx's are of the same shape, they cound be tensor
    fx1 = torch.tanh(x_minus) * torch.sigmoid(y_minus)
    fx2 = torch.tanh(x_minus) * torch.sigmoid(y_plus)
    fx3 = torch.tanh(x_plus) * torch.sigmoid(y_plus)
    fx4 = torch.tanh(x_plus) * torch.sigmoid(y_minus)
    
    loss1 = a*x_minus + b*y_minus + c - fx1
    valid = (loss1 <= 0)
    loss1 = torch.clamp(loss1, min = confidence)
    
    loss2 = a*x_minus + b*y_plus + c - fx2
    valid = valid * (loss2<=0)
    loss2 = torch.clamp(loss2, min = confidence)
    
    loss3 = a*x_plus + b*y_plus + c - fx3
    valid = valid * (loss3<=0)
    loss3 = torch.clamp(loss3, min = confidence)
    
    loss4 = a*x_plus + b*y_minus + c - fx4
    valid = valid * (loss4<=0)
    loss4 = torch.clamp(loss4, min = confidence)
    
    loss = loss1 + loss2 + loss3 + loss4
    return loss, valid




def qualification_loss_upper(x, y ,x_minus, x_plus, y_minus, y_plus):
    # indicate whether (x,y) \in (x_minus, x_plus) * (y_minus, y_plus)
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

def train_upper(x0, y0, x_minus, x_plus, y_minus, y_plus, lr=1e-3, 
                max_iter=100, print_info = True):
    # search over (x,y) \in (x_minus, x_plus) * (y_minus, y_plus) 
    # the plane z = ax + by + c is the tangent plane of the surface tanh(x) sigmoid(y) at point (x,y)
    x = x0.data.clone()
    x.requires_grad = True
    
    y = y0.data.clone()
    y.requires_grad = True
    
    a_best = torch.zeros(x_minus.shape, device = x_minus.device)
    b_best = torch.zeros(x_minus.shape, device = x_minus.device)
    c_best = torch.zeros(x_minus.shape, device = x_minus.device)
    x_best = torch.zeros(x_minus.shape, device = x_minus.device)
    y_best = torch.zeros(x_minus.shape, device = x_minus.device)
    
    optimizer = optim.Adam([x,y], lr=lr)
    cubic = ((x_plus-x_minus) * (y_plus-y_minus) * 
             torch.tanh(x_plus) * torch.sigmoid(y_plus))
    cubic = torch.clamp(cubic, min=1e-3)
    v_best = torch.ones(x_minus.shape, device = x_minus.device) * 1e8
    for i in range(max_iter):
        q_loss, valid = qualification_loss_upper(x, y ,
                            x_minus, x_plus, y_minus, y_plus)
        a,b,c = plane(x,y)
        v_loss = get_volume(a,b,c,x_minus, x_plus, y_minus, y_plus)/cubic
        
        best = (v_loss < v_best) * valid
        a_best[best] = a[best]
        b_best[best] = b[best]
        c_best[best] = c[best]
        x_best[best] = x[best]
        y_best[best] = y[best]
        v_best[best] = v_loss[best]
        # print('volume', v_loss)
        if print_info:
            print('1u q loss: %.4f volume: %.4f' % (q_loss.mean().item(), v_loss.mean().item()))
        
        loss = q_loss + (valid.float()+0.1) * v_loss
        loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print('x,y',x,y)
    return a_best, b_best, c_best, x_best, y_best

def adjust_upper_plane(x0, y0, x_minus, x_plus, y_minus, y_plus, lr=1e-2, 
                       max_iter=100, print_info = True):
    # this function finds the minimum value of a*x + b*y + c - torch.tanh(x)*torch.sigmoid(y)
    # if it is less than zero, we will need to raise the plane z = ax + by + c a little bit
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
    
    a,b,c = plane(x0,y0)
    optimizer = optim.Adam([x1,y1,x2,y2,x3,y3,x4,y4], lr=lr)
    
    x1_best = torch.zeros(x_minus.shape, device = x_minus.device)
    y1_best = torch.zeros(x_minus.shape, device = x_minus.device)
    loss1_best = torch.ones(x_minus.shape, device = x_minus.device) * 1000
    x2_best = torch.zeros(x_minus.shape, device = x_minus.device)
    y2_best = torch.zeros(x_minus.shape, device = x_minus.device)
    loss2_best = torch.ones(x_minus.shape, device = x_minus.device) * 1000
    x3_best = torch.zeros(x_minus.shape, device = x_minus.device)
    y3_best = torch.zeros(x_minus.shape, device = x_minus.device)
    loss3_best = torch.ones(x_minus.shape, device = x_minus.device) * 1000
    x4_best = torch.zeros(x_minus.shape, device = x_minus.device)
    y4_best = torch.zeros(x_minus.shape, device = x_minus.device)
    loss4_best = torch.ones(x_minus.shape, device = x_minus.device) * 1000
    for i in range(max_iter):
        loss1 = a*x1 + b*y1 + c - torch.tanh(x1)*torch.sigmoid(y1)
        loss2 = a*x2 + b*y2 + c - torch.tanh(x2)*torch.sigmoid(y2)
        loss3 = a*x3 + b*y3 + c - torch.tanh(x3)*torch.sigmoid(y3)
        loss4 = a*x4 + b*y4 + c - torch.tanh(x4)*torch.sigmoid(y4)
        
        qloss1, valid1 = qualification_loss_upper(x1, y1 ,
                                    x_minus, x_plus, y_minus, y_plus)
        best1 = (loss1 < loss1_best) * valid1
        x1_best[best1] = x1[best1]
        y1_best[best1] = y1[best1]
        loss1_best[best1] = loss1[best1]
        
        qloss2, valid2 = qualification_loss_upper(x2, y2 ,
                                    x_minus, x_plus, y_minus, y_plus)
        best2 = (loss2 < loss2_best) * valid2
        x2_best[best2] = x2[best2]
        y2_best[best2] = y2[best2]
        loss2_best[best2] = loss2[best2]
        
        qloss3, valid3 = qualification_loss_upper(x3, y3 ,
                                    x_minus, x_plus, y_minus, y_plus)
        best3 = (loss3 < loss3_best) * valid3
        x3_best[best3] = x3[best3]
        y3_best[best3] = y3[best3]
        loss3_best[best3] = loss3[best3]
        
        qloss4, valid4 = qualification_loss_upper(x4, y4 ,
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
            print('1 adjust upper plane loss: %.4f' % loss.item())
    return x1_best,y1_best,x2_best,y2_best,x3_best,y3_best,x4_best,y4_best,loss1_best,loss2_best,loss3_best,loss4_best

def raise_upper_plane(loss1,loss2,loss3,loss4,a,b,c,x_minus, x_plus, y_minus, y_plus):

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
    
    loss1 = a*x_minus + b*y_minus + c - torch.tanh(x_minus)*torch.sigmoid(y_minus)
    loss2 = a*x_minus + b*y_plus + c - torch.tanh(x_minus)*torch.sigmoid(y_plus)
    loss3 = a*x_plus + b*y_plus + c - torch.tanh(x_plus)*torch.sigmoid(y_plus)
    loss4 = a*x_plus + b*y_minus + c - torch.tanh(x_plus)*torch.sigmoid(y_minus)
    
    valid1 = (loss1 < minimum).float()
    minimum = valid1 * loss1 + (1-valid1)*minimum
    
    valid2 = (loss2 < minimum).float()
    minimum = valid2 * loss2 + (1-valid2)*minimum
    
    valid3 = (loss3 < minimum).float()
    minimum = valid3 * loss3 + (1-valid3)*minimum
    
    valid4 = (loss4 < minimum).float()
    minimum = valid4 * loss4 + (1-valid4)*minimum

    # minimum <= 0
    return -minimum * 1.01

import train_activation_plane
def main_lower(x_minus, x_plus, y_minus, y_plus, print_info = True):
    
    z10 = torch.tanh(x_minus) * torch.sigmoid(y_minus)
    z20 = torch.tanh(x_minus) * torch.sigmoid(y_minus)
    z30 = torch.tanh(x_minus) * torch.sigmoid(y_minus)
    a,b,c = train_activation_plane.train_lower(z10,z20,z30,x_minus, x_plus, y_minus, y_plus, 
                        qualification_loss_lower_standard, '1l', lr=1e-2,
                        max_iter = 500, print_info = print_info)
    return a.detach(),b.detach(),c.detach()

def main_upper(x_minus, x_plus, y_minus, y_plus, print_info = True):
    x0 = (x_minus + x_plus)/2
    y0 = (y_minus + y_plus)/2
    a_best, b_best, c_best, x_best, y_best = train_upper(x0, y0, 
                x_minus, x_plus, y_minus, y_plus, lr=1e-2, 
                max_iter=500, print_info=print_info)
    
    x1,y1,x2,y2,x3,y3,x4,y4,loss1,loss2,loss3,loss4 = adjust_upper_plane(x_best, y_best, x_minus, 
                    x_plus, y_minus, y_plus, lr=1e-2, max_iter=200, print_info=print_info)
    increase = raise_upper_plane(loss1,loss2,loss3,loss4,a_best,b_best,c_best,x_minus, x_plus, y_minus, y_plus)
    c_best = c_best + increase
    return a_best.detach(), b_best.detach(), c_best.detach()   

if __name__ ==  '__main__':
    
    x_minus = torch.Tensor([0.062])
    x_plus = torch.Tensor([5])
    y_minus = torch.Tensor([0.1032])
    y_plus = torch.Tensor([5.3253])
    
    print_info = False

    a,b,c = main_lower(x_minus, x_plus, y_minus, y_plus, print_info = print_info)
    a_best, b_best, c_best  = main_upper(x_minus, x_plus, y_minus, y_plus, print_info=print_info)
    
    num=0
    v1, v2 = plot_2_surface(x_minus[num], x_plus[num],y_minus[num], y_plus[num], 
                 a[num],b[num],c[num],a_best[num], b_best[num], c_best[num])