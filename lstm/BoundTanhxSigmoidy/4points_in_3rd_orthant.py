#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 15:20:54 2019

@author: root
"""
import torch
import utils
import matplotlib.pyplot as plt

def find_y2(x_minus, x_plus, y_minus, y_plus, y1):
    #given a point P(x_plus, y1) on x3-x4, 
    #we try to find a point M(x_minus, y2) y2 on x1-x2
    #such that they have the same tangent along y axis
    #on x3-x4, the function is tanh(x_plus) sigmoid(y)
    #on x1-x2, the function is tanh(x_minus) sigmoid(y)
    y2 = torch.zeros(y1.shape, device=x_minus.device)
    z = torch.zeros(y1.shape, device=x_minus.device)
    
    b0 = torch.tanh(x_plus) * torch.sigmoid(y1) * (1-torch.sigmoid(y1))
    
    b_upper = (torch.tanh(x_minus) * torch.sigmoid(y_minus) * 
               (1-torch.sigmoid(y_minus)))
    b_lower = (torch.tanh(x_minus) * torch.sigmoid(y_plus) * 
               (1-torch.sigmoid(y_plus)))
    
    touch_y_minus = (b0 >= b_upper)
    if touch_y_minus.sum()>0:
        y2[touch_y_minus] = y_minus[touch_y_minus]
        z[touch_y_minus] = (torch.tanh(x_minus) * torch.sigmoid(y_minus))[touch_y_minus]
    
    touch_y_plus = (b0 <= b_lower)
    if touch_y_plus.sum()>0:
        y2[touch_y_plus] = y_plus[touch_y_plus]
        z[touch_y_plus] = (torch.tanh(x_minus) * torch.sigmoid(y_plus))[touch_y_plus]
    
    between = (b0 < b_upper) * (b0 > b_lower)
    
    if between.sum()>0:
        y_temp,z_temp = binary_search_upper(x_minus[between], x_plus[between], 
                            y_minus[between], y_plus[between], b0[between])
        y2[between] = y_temp
        z[between] = z_temp
    return y2,z

def binary_search_upper(x_minus, x_plus, y_minus, y_plus, b0):
    #we want to find the point on x1-x2 that have the tangent as b0
    #the function on x1-x2 is tanh(x_minus) * sigmoid(y)
    y_upper = y_plus.data.clone()
    y_lower = y_minus.data.clone()
    
    for i in range(10):
        y = (y_upper + y_lower)/2
        b = (torch.tanh(x_minus) * torch.sigmoid(y) * 
               (1-torch.sigmoid(y)))
        idx = (b-b0)>0
        y_lower[idx] = y[idx]
        idx = 1-idx
        y_upper[idx] = y[idx]
    
    u_minus = torch.tanh(x_minus)
    v_l = torch.sigmoid(y_lower)
    v_u = torch.sigmoid(y_upper)
    b_l = (torch.tanh(x_minus) * torch.sigmoid(y_lower) * 
               (1-torch.sigmoid(y_lower)))
    b_u = (torch.tanh(x_minus) * torch.sigmoid(y_upper) * 
               (1-torch.sigmoid(y_upper)))
    #line 1's function is z = u_minus * v_l + b_l * (y-y_l)
    #line 2's function is z = u_minus * v_u + b_u * (y-y_u)
    #we want to find their cross point (y,z)
    y = (u_minus * v_u - u_minus*v_l -b_u*y_upper + b_l*y_lower)/(b_l-b_u)
    z = u_minus*v_l + b_l*(y-y_lower)    
    return y,z

def get_abc_upper(y1,y2,z,x_minus, x_plus, y_minus, y_plus):
    #the plane passes y1 and tangent it
    #the plane also passes x_minus, y2, z
    b = torch.tanh(x_plus) * torch.sigmoid(y1) * (1-torch.sigmoid(y1))
    #a*x_plus + b*y1 + c = tanh(x_plus) sigmoid(y1)
    #a*x_minus + b*y2 + c = z
    #a*(x_plus-x_minus) + b*(y1-y2) = tanh(x_plus) sigmoid(y1) - z
    a = (torch.tanh(x_plus) * torch.sigmoid(y1) - z - b*(y1-y2))/(x_plus-x_minus)
    c = z - a*x_minus - b*y2
    return a,b,c

def estimate_gradient_upper(y, eps, x_minus, x_plus, y_minus, y_plus):
    y1 = y-eps
    y2,z = find_y2(x_minus, x_plus, y_minus, y_plus, y1)
    a,b,c = get_abc_upper(y1,y2,z,x_minus, x_plus, y_minus, y_plus)
    volume1 = utils.get_volume(a,b,c,x_minus, x_plus, y_minus, y_plus)
    
    y1 = y+eps
    y2,z = find_y2(x_minus, x_plus, y_minus, y_plus, y1)
    a,b,c = get_abc_upper(y1,y2,z,x_minus, x_plus, y_minus, y_plus)
    volume2 = utils.get_volume(a,b,c,x_minus, x_plus, y_minus, y_plus)
    gradient = (volume2 - volume1) / (2*eps)
    return gradient
    
def binary_search_y1(x_minus, x_plus, y_minus, y_plus):
    eps = 1e-4
    y_lower = y_minus.data.clone() #gradient(y_lower) < 0 
    y_upper = y_plus.data.clone() #gradient(y_upper) > 0 
    y1 = (y_lower + y_upper)/2
    for i in range(10):
        y1 = (y_lower + y_upper)/2
        g = estimate_gradient_upper(y1, eps, x_minus, x_plus, y_minus, y_plus)
        idx = (g>0)
        y_upper[idx] = y1[idx]
        
        idx = 1-idx
        y_lower[idx] = y1[idx]
    return (y_upper+y_lower)/2


def binary_search_lower(x_minus, x_plus, y_minus, y_plus, a0):
    #we want to find the point on x1-x4 that have the tangent as a0
    #the function on x1-x4 is tanh(x) * sigmoid(y_minus)
    x_upper = x_plus.data.clone()
    x_lower = x_minus.data.clone()
    
    for i in range(10):
        x = (x_upper + x_lower)/2
        a = torch.sigmoid(y_minus) * (1-torch.tanh(x)**2)
        idx = (a-a0)>0
        x_upper[idx] = x[idx]
        idx = 1-idx
        x_lower[idx] = x[idx]
    
    v_minus = torch.sigmoid(y_minus)
    u_l = torch.tanh(x_lower)
    u_u = torch.tanh(x_upper)
    a_l = torch.sigmoid(y_minus) * (1-torch.tanh(x_lower)**2)
    a_u = torch.sigmoid(y_minus) * (1-torch.tanh(x_upper)**2)
    #line 1's function is z = v_minus * u_l + a_l * (x-x_l)
    #line 2's function is z = v_minus * u_u + a_u * (x-x_u)
    #we want to find their cross point (y,z)
    x = (v_minus * u_u - v_minus*u_l -a_u*x_upper + a_l*x_lower)/(a_l-a_u)
    z = v_minus*u_l + a_l*(x-x_lower)    
    return x,z

def find_x2(x_minus, x_plus, y_minus, y_plus, x1):
    #given a point Q(x1, y_plus) on x2-x3, 
    #we try to find a point N(x2, y_minus) on x1-x4
    #such that they have the same tangent along x axis
    #on x2-x3, the function is sigmoid(y_plus) tanh(x) 
    #on x1-x4, the function is sigmoid(y_minus) tanh(x) 
    x2 = torch.zeros(x1.shape, device=x_minus.device)
    z = torch.zeros(x1.shape, device=x_minus.device)
    
    a0 = (1-torch.tanh(x1)**2) * torch.sigmoid(y_plus)
    
    a_upper = (1-torch.tanh(x_plus)**2) * torch.sigmoid(y_minus)
    a_lower = (1-torch.tanh(x_minus)**2) * torch.sigmoid(y_minus)
    
    touch_x_minus = (a0 <= a_lower)
    if touch_x_minus.sum()>0:
        x2[touch_x_minus] = x_minus[touch_x_minus]
        z[touch_x_minus] = (torch.tanh(x_minus) * torch.sigmoid(y_minus))[touch_x_minus]
    
    touch_x_plus = (a0 >= a_upper)
    if touch_x_plus.sum()>0:
        x2[touch_x_plus] = x_plus[touch_x_plus]
        z[touch_x_plus] = (torch.tanh(x_plus) * torch.sigmoid(y_minus))[touch_x_plus]
    
    between = (a0 < a_upper) * (a0 > a_lower)
    
    if between.sum()>0:
        x_temp,z_temp = binary_search_lower(x_minus[between], x_plus[between], 
                            y_minus[between], y_plus[between], a0[between])
        x2[between] = x_temp
        z[between] = z_temp
    return x2,z

def get_abc_lower(x1,x2,z,x_minus, x_plus, y_minus, y_plus):
    #the plane passes (x1, y_plus, u1*v_plus) and tangent it
    #the plane also passes x2, y_minus, z
    a = (1 - torch.tanh(x1)**2) * torch.sigmoid(y_plus)
    #a*x1 + b*y_plus + c = tanh(x1) sigmoid(y_plus)
    #a*x2 + b*y_minus + c = z
    #a*(x1-x2) + b*(y_plus-y_minus) = tanh(x1) sigmoid(y_plus) - z
    b = (torch.tanh(x1) * torch.sigmoid(y_plus) - z - a*(x1-x2))/(y_plus-y_minus)
    c = z - a*x2 - b*y_minus
    return a,b,c

def estimate_gradient_lower(x, eps, x_minus, x_plus, y_minus, y_plus):
    x1 = x-eps
    x2,z = find_x2(x_minus, x_plus, y_minus, y_plus, x1)
    a,b,c = get_abc_lower(x1,x2,z,x_minus, x_plus, y_minus, y_plus)
    volume1 = utils.get_volume(a,b,c,x_minus, x_plus, y_minus, y_plus)
    
    x1 = x+eps
    x2,z = find_x2(x_minus, x_plus, y_minus, y_plus, x1)
    a,b,c = get_abc_lower(x1,x2,z,x_minus, x_plus, y_minus, y_plus)
    volume2 = utils.get_volume(a,b,c,x_minus, x_plus, y_minus, y_plus)
    gradient = (volume2 - volume1) / (2*eps)
    return gradient
    
def binary_search_x1(x_minus, x_plus, y_minus, y_plus):
    eps = 1e-3
    x_lower = x_minus.data.clone() #gradient(y_lower) < 0 
    x_upper = x_plus.data.clone() #gradient(y_upper) > 0 
    x1 = (x_lower + x_upper)/2
    for i in range(10):
        x1 = (x_lower + x_upper)/2
        g = estimate_gradient_lower(x1, eps, x_minus, x_plus, y_minus, y_plus)
        idx = (g<0)
        x_upper[idx] = x1[idx]
        
        idx = 1-idx
        x_lower[idx] = x1[idx]
    return (x_upper+x_lower)/2

def main_lower(x_minus, x_plus, y_minus, y_plus, plot=False, num=0):
    # x1 = (x_minus + x_plus) / 2
    # x1 = x_minus
    x1 = binary_search_x1(x_minus, x_plus, y_minus, y_plus)
    # y1 = binary_search_y1(x_minus, x_plus, y_minus, y_plus)
    x2,z = find_x2(x_minus, x_plus, y_minus, y_plus, x1)
    a,b,c = get_abc_lower(x1,x2,z,x_minus, x_plus, y_minus, y_plus)
    volume = utils.get_volume(a,b,c,x_minus, x_plus, y_minus, y_plus)
    
    if plot:
        utils.plot_surface(x_minus[num], x_plus[num],y_minus[num], y_plus[num],
                               a[num],b[num],c[num])
    
    # x = torch.linspace(x_minus.item(), x_plus.item(),100)
    # v = torch.zeros(x.shape)
    # g = torch.zeros(x.shape)
    # for i in range(len(x)):
    #     x2,z = find_x2(x_minus, x_plus, y_minus, y_plus, torch.Tensor([x[i]]))
    #     a,b,c = get_abc_lower(x[i],x2,z,x_minus, x_plus, y_minus, y_plus)
    #     v[i] = utils.get_volume(a,b,c,x_minus, x_plus, y_minus, y_plus)
    #     g[i] = estimate_gradient_lower(torch.Tensor([x[i]]), 1e-3, x_minus, x_plus, y_minus, y_plus)
    # # v = 
    # plt.figure()
    # plt.plot(x.numpy(),v.numpy())
    # plt.figure()
    # plt.plot(x.numpy(),g.numpy())
    return a,b,c,volume, x1, x2

def main_upper(x_minus, x_plus, y_minus, y_plus, plot=False, num=0):
    
    y1 = binary_search_y1(x_minus, x_plus, y_minus, y_plus)
    y2,z = find_y2(x_minus, x_plus, y_minus, y_plus, y1)
    a,b,c = get_abc_upper(y1,y2,z,x_minus, x_plus, y_minus, y_plus)
    volume = utils.get_volume(a,b,c,x_minus, x_plus, y_minus, y_plus)
    
    if plot:
        utils.plot_surface(x_minus[num], x_plus[num],y_minus[num], y_plus[num],
                               a[num],b[num],c[num])
    return a,b,c,volume,y1,y2

if __name__ == '__main__':
    x_minus = torch.Tensor([-5])
    x_plus = torch.Tensor([-0.1])
    y_minus = torch.Tensor([-2])
    y_plus = torch.Tensor([-0.1])
    
    x_plus = -torch.rand(100)*2
    x_minus = x_plus - torch.rand(100)*3
    
    y_plus = -torch.rand(100)*2
    y_minus = y_plus - torch.rand(100)*3
    
    a,b,c,v, x1, x2 = main_lower(x_minus, x_plus, y_minus, y_plus)
    
    # y1 = (y_plus + y_plus) / 2
    # y1 = binary_search_y1(x_minus, x_plus, y_minus, y_plus)
    # y2,z = find_y2(x_minus, x_plus, y_minus, y_plus, y1)
    # a,b,c = get_abc_upper(y1,y2,z,x_minus, x_plus, y_minus, y_plus)
    # volume = utils.get_volume(a,b,c,x_minus, x_plus, y_minus, y_plus)
    # utils.plot_surface(x_minus, x_plus,y_minus, y_plus, a,b,c)
    
    # y = torch.linspace(y_minus.item(), y_plus.item(),100)
    # v = torch.zeros(y.shape)
    # for i in range(len(y)):
    #     y2,z = find_y2(x_minus, x_plus, y_minus, y_plus, torch.Tensor([y[i]]))
    #     a,b,c = get_abc_upper(y[i],y2,z,x_minus, x_plus, y_minus, y_plus)
    #     v[i] = utils.get_volume(a,b,c,x_minus, x_plus, y_minus, y_plus)
    # # v = 
    # plt.figure()
    # plt.plot(y.numpy(),v.numpy())
    
    # print(0)
    
    
    
    
    
    
    
    
    
    
    
    
    