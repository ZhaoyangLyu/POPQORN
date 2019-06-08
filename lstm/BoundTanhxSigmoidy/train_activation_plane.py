#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 09:06:31 2019

@author: root
"""
import torch 
# import torch.nn.functional as F
import torch.optim as optim
from utils import plane, get_volume, plot_surface, plot_2_surface
first = __import__('4points_in_1st_orthant')

def train_lower(z10,z20,z30,x_minus, x_plus, y_minus, y_plus, qualification_loss_lower, 
                message, lr=1e-3, max_iter = 100, print_info = True):
    # the initial position of the plane z = ax + by + c is determined by the three points
    # (x_minus, y_minus, z10), (x_minus, y_plus, z20), (x_plus, y_plus, z30)
    # a,b,c can be determined by (x_minus, y_minus, z1), (x_minus, y_plus, z2), (x_plus, y_plus, z3)
    # z10,z20,z30,x_minus, x_plus, y_minus, y_plus must be tensors of the same shape

    # qualification_loss_lower(x_minus, x_plus, y_minus, y_plus, a,b,c, confidence=-0.0)
    # this function indicates whether the plane is valid, namely, plane z = ax + by + c is below the surface z = tanh(x) sigmoid(y)
    # in the rectangular area [x_minus, x_plus] * [y_minus， y_plus]
    # the plane is valid if and only if qualification_loss_lower <= 0

    # This function search over z1,z2,z3 to maximmize the volume between the plane z = ax + by + c and the z = 0 plane
    # with the constraint qualification_loss_lower <= 0

    z1 = z10.data.clone() #(x_minus, y_minus) 
    z1.requires_grad = True
    
    z2 = z20.data.clone() #(x_minus, y_plus)
    z2.requires_grad = True
    
    z3 = z30.data.clone() #(x_plus, y_plus)
    z3.requires_grad = True
    
    ones = torch.ones(x_minus.shape, device = x_minus.device)
    
    z1_best = torch.zeros(x_minus.shape, device = x_minus.device)
    z2_best = torch.zeros(x_minus.shape, device = x_minus.device)
    z3_best = torch.zeros(x_minus.shape, device = x_minus.device)
    
    optimizer = optim.Adam([z1,z2,z3], lr=lr)
    cubic = (x_plus-x_minus) * (y_plus-y_minus)
    cubic = torch.clamp(cubic, min=1e-3)
    
    v_best = -100000*torch.ones(x_minus.shape, device = x_minus.device)
    for i in range(max_iter):
        #x * a + y * b + c = z 
        a,b,c = get_abc(x_minus, y_minus, ones, z1,
                        x_minus, y_plus, ones, z2,
                        x_plus, y_plus, ones, z3)
        
        q_loss, valid = qualification_loss_lower(x_minus, x_plus, y_minus, y_plus,
                                 a,b,c, confidence=-0.0)
       
        v_loss = get_volume(a,b,c,x_minus, x_plus, y_minus, y_plus)/cubic
        
        best = (v_loss > v_best) * valid
        z1_best[best] = z1[best]
        z2_best[best] = z2[best]
        z3_best[best] = z3[best]
        v_best[best] = v_loss[best]
        # print('volume', v_loss)
        if print_info:
            print(message + ' q loss: %.4f volume: %.4f' % (q_loss.mean().item(), v_loss.mean().item()))
        
        loss = (q_loss - (valid.float()+0.1) * v_loss) 
        loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        z1.grad[z1.grad != z1.grad] = 0
        z2.grad[z2.grad != z2.grad] = 0
        z3.grad[z3.grad != z3.grad] = 0
        optimizer.step()
    a_best,b_best,c_best =  get_abc(x_minus, y_minus, ones, z1_best,
                                    x_minus, y_plus, ones, z2_best,
                                    x_plus, y_plus, ones, z3_best)   
    return a_best,b_best,c_best


def train_upper(z10,z20,z30,x_minus, x_plus, y_minus, y_plus, qualification_loss_upper, 
                message, lr=1e-3, max_iter = 100, print_info = True):
    # the initial position of the plane z = ax + by + c is determined by the three points
    # (x_minus, y_minus, z10), (x_minus, y_plus, z20), (x_plus, y_plus, z30)
    # a,b,c can be determined by (x_minus, y_minus, z1), (x_minus, y_plus, z2), (x_plus, y_plus, z3)
    # z10,z20,z30,x_minus, x_plus, y_minus, y_plus must be tensors of the same shape

    # qualification_loss_upper(x_minus, x_plus, y_minus, y_plus, a,b,c, confidence=-0.0)
    # this function indicates whether the plane is valid, namely, plane z = ax + by + c is above the surface z = tanh(x) sigmoid(y)
    # in the rectangular area [x_minus, x_plus] * [y_minus， y_plus]
    # the plane is valid if and only if qualification_loss_upper <= 0

    # This function search over z1,z2,z3 to minimize the volume between the plane z = ax + by + c and the z = 0 plane
    # with the constraint qualification_loss_upper <= 0

    z1 = z10.data.clone() #(x_minus, y_minus) 
    z1.requires_grad = True
    
    z2 = z20.data.clone() #(x_minus, y_plus)
    z2.requires_grad = True
    
    z3 = z30.data.clone() #(x_plus, y_plus)
    z3.requires_grad = True
    
    ones = torch.ones(x_minus.shape, device = x_minus.device)
    
    z1_best = torch.zeros(x_minus.shape, device = x_minus.device)
    z2_best = torch.zeros(x_minus.shape, device = x_minus.device)
    z3_best = torch.zeros(x_minus.shape, device = x_minus.device)
    
    optimizer = optim.Adam([z1,z2,z3], lr=lr)
    cubic = (x_plus-x_minus) * (y_plus-y_minus)
    cubic = torch.clamp(cubic, min=1e-3)
    
    v_best = 100000*torch.ones(x_minus.shape, device = x_minus.device)
    for i in range(max_iter):
        #x * a + y * b + c = z 
        a,b,c = get_abc(x_minus, y_minus, ones, z1,
                        x_minus, y_plus, ones, z2,
                        x_plus, y_plus, ones, z3)
        
        q_loss, valid = qualification_loss_upper(x_minus, x_plus, y_minus, y_plus,
                                 a,b,c, confidence=-0.0)
       
        v_loss = get_volume(a,b,c,x_minus, x_plus, y_minus, y_plus)/cubic
        
        best = (v_loss < v_best) * valid
        z1_best[best] = z1[best]
        z2_best[best] = z2[best]
        z3_best[best] = z3[best]
        v_best[best] = v_loss[best]
        # print('volume', v_loss)
        if print_info:
            print(message + ' q loss: %.4f volume: %.4f' % (q_loss.mean().item(), v_loss.mean().item()))
        
        loss = q_loss + (valid.float()+0.1) * v_loss
        loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        z1.grad[z1.grad != z1.grad] = 0
        z2.grad[z2.grad != z2.grad] = 0
        z3.grad[z3.grad != z3.grad] = 0
        
        optimizer.step()
    a_best,b_best,c_best =  get_abc(x_minus, y_minus, ones, z1_best,
                                    x_minus, y_plus, ones, z2_best,
                                    x_plus, y_plus, ones, z3_best)   
    return a_best,b_best,c_best


def det(a11,a12,a13,
        a21,a22,a23,
        a31,a32,a33):
    d = a11*(a22*a33-a23*a32) -a12*(a21*a33-a23*a31)+a13*(a21*a32-a22*a31)
    return d


def get_abc(a11, a12, a13, b1,
            a21, a22, a23, b2,
            a31, a32, a33, b3):
    #solve the equation Ax=b
    #each element of A and b can be any shape
    frac = det(a11, a12, a13, 
               a21, a22, a23,
               a31, a32, a33)
    if (frac == 0).sum()>0:
        print(a11, a12, a13, 
              a21, a22, a23,
              a31, a32, a33)
        raise Exception('The det of the coefficients maxtrix is zero')
    x1 = det(b1, a12, a13, 
             b2, a22, a23,
             b3, a32, a33)
    x1 = x1 / frac
    
    x2 = det(a11, b1, a13, 
             a21, b2, a23,
             a31, b3, a33)
    x2 = x2 / frac
    
    x3 = det(a11, a12, b1, 
             a21, a22, b2,
             a31, a32, b3)
    x3 = x3 / frac
    return x1, x2, x3

if __name__ == '__main__':
    # x_minus = torch.Tensor([0,0.3])
    # x_plus = torch.Tensor([0.1,2])
    # y_minus = torch.Tensor([0,0.1])
    # y_plus = torch.Tensor([0.1,1])
    x_minus = torch.Tensor([0.9062])
    x_plus = torch.Tensor([0.9295])
    y_minus = torch.Tensor([0.1032])
    y_plus = torch.Tensor([5.3253])
    # a0 = torch.zeros(x_minus.shape, device = x_minus.device)
    # b0 = torch.zeros(x_minus.shape, device = x_minus.device)
    z10 = torch.tanh(x_minus) * torch.sigmoid(y_minus)
    z20 = torch.tanh(x_minus) * torch.sigmoid(y_minus)
    z30 = torch.tanh(x_minus) * torch.sigmoid(y_minus)
    a,b,c = train_lower(z10,z20,z30,x_minus, x_plus, y_minus, y_plus, 
                        first.qualification_loss_lower_standard, '1l', lr=1e-2,
                        max_iter = 500)
    
    num=0
    plot_surface(x_minus[num], x_plus[num],y_minus[num], y_plus[num], 
                 a[num],b[num],c[num])























