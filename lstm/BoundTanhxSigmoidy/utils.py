#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 19:32:03 2018

@author: root
"""

import torch
import matplotlib.pyplot as plt
# import torch.optim as optim
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch.nn.functional as F

# def sigmoid(x):
#     return 1/(1+torch.exp(-x))
def sigmoid(x):
    #numpy operation
    return 1/(1+np.exp(-x))

# def d_sigmoid(x):
#     temp = torch.sigmoid(x)
#     return temp*(1-temp)

# def d_tanh(x):
#     return (1 - torch.tanh(x) ** 2)

def plane(x,y):
    #the surface is z = tanh(x) sigmoid(y)
    #the function returns the tangent plane at point x,y
    #z = a*x + b*y + c
    #x and y should be torch tensors
    s = torch.sigmoid(y)
    t = torch.tanh(x)
    z = s*t
    a = s*(1 - t**2)
    b = t*s*(1-s)
    c = z - a * x - b * y
    return a,b,c

# def sigmoid_lmax(x_minus, y_minus):
#     alpha = -torch.tanh(x_minus)
#     s = torch.sigmoid(y_minus)
#     l_max = alpha * s * ((1-s)*y_minus - 1)
#     return l_max
    
# def sigmoid_loss(a,b,c, x_minus, y_minus, l_max, confidence=-0.01):
#     #a,b,c, x_minus, y_minus, l_max could be any shape tensor
#     #they should be of the same shape

#     l = a * x_minus + c
#     alpha  = -torch.tanh(x_minus)
#     loss = torch.zeros(a.shape, device=x_minus.device)
#     idx = (l <= l_max).float()
#     loss =loss + torch.clamp(b-l/2, min=confidence) * idx
    
#     idx = 1-idx
#     k = -(l+alpha*torch.sigmoid(y_minus))/y_minus
#     loss = loss + torch.clamp(b-k, min=confidence) * idx
#     return loss
#     # if l <= l_max:
#     #     #we require b <= l/2
#     #     return torch.clamp(b-l/2, min=confidence)
#     # else:
#     #     k = -(l+alpha*torch.sigmoid(y_minus))/y_minus
#     #     #we require b <= k
#     #     return torch.clamp(b-k, min=confidence)
# def sigmoid_lmin(x_plus, y_minus):
#     alpha = torch.tanh(x_plus)
#     s = torch.sigmoid(y_minus)
#     l_min = alpha * s * (1-(1-s)*y_minus)
#     return l_min
    
# def sigmoid_loss_lower(a,b,c, x_plus, y_minus, l_min, confidence=-0.01):
#     #a,b,c, x_minus, y_minus, l_max could be any shape tensor
#     #they should be of the same shape

#     l = a * x_plus + c
#     alpha  = torch.tanh(x_plus)
#     loss = torch.zeros(a.shape, device=x_plus.device)
#     idx = (l >= l_min).float()
#     #b >= l/2, l/2-b<=0, minimize l/2-b
#     loss = loss + torch.clamp(l/2-b, min=confidence) * idx
    
#     idx = 1-idx
#     k = -(l-alpha*torch.sigmoid(y_minus))/y_minus
#     loss = loss + torch.clamp(k-b, min=confidence) * idx
#     return loss

    
# def tanh_lmax(x_plus, y_minus):
#     e = torch.sigmoid(y_minus)
#     t = torch.tanh(x_plus)
#     l_max = e*(t - (1-t**2)*x_plus)
#     return l_max
    
# def tanh_loss(a,b,c, x_plus, y_minus, l_max, confidence=-0.01):
#     #a,b,c, x_plus, y_minus, l_max could be any shape tensor
#     #they should be of the same shape
#     l = b * y_minus + c
#     e = torch.sigmoid(y_minus)
#     loss = torch.zeros(a.shape, device=x_plus.device)
    
#     idx = (l<=l_max).float()
#     loss = loss + torch.clamp(e-l-a, min=confidence)*idx
    
#     idx = 1-idx
#     k = (e*torch.tanh(x_plus)-l) / x_plus
#     loss = loss + torch.clamp(k-a, min=confidence)*idx
#     return loss
#     # if l <= l_max:
#     #     #we require a >= e-l
#     #     return torch.clamp(e-l-a, min=confidence)
#     # #when e-l-a < confidence,  we stop minimize it
#     # else:
#     #     k = (e*torch.tanh(x_plus)-l) / x_plus
#     #     #we require a >= k
#     #     return torch.clamp(k-a, min=confidence)
#     # #when k-a < confidence,  we stop minimize it
# def tanh_lmin(x_minus, y_minus):
#     e = torch.sigmoid(y_minus)
#     t = torch.tanh(x_minus)
#     l_min = e*(t - (1-t**2)*x_minus)
#     return l_min
    
# def tanh_loss_lower(a,b,c, x_minus, y_minus, l_min, confidence=-0.01):
#     #a,b,c, x_plus, y_minus, l_max could be any shape tensor
#     #they should be of the same shape
#     l = b * y_minus + c
#     e = torch.sigmoid(y_minus)
#     loss = torch.zeros(a.shape, device = x_minus.device)
    
#     idx = (l>=l_min).float()
#     loss = loss + torch.clamp(e+l-a, min=confidence)*idx
    
#     idx = 1-idx
#     k = (e*torch.tanh(x_minus)-l) / x_minus
#     loss = loss + torch.clamp(k-a, min=confidence) * idx
#     return loss

    
# def qualification_loss(a,b,c,x_minus, x_plus, y_minus, y_plus, tanh_l_max,
#                        sigmoid_l_max, confidence=-0.01):
#     #check whether a*x + b*y + c is above z(x) in the rectangular area
#     #we require loss1-loss7 <=0
    
#     #h(x,y) = a*x + b*y + c
#     #z(x,y) = tanh(x) * sigmoid(y)
#     #x1 (x_minus, y_minus)
#     #x2 (x_minus, y_plus)
#     #x3 (x_plus, y_plus)
#     #x4 (x_plus, y_minus)
#     #A (x_minus, 0) z = 0.5 * tanh(x_minus), h = a*x_minus + c
#     #B (0, y_plus) z = 0, h = b*y_plus + c
#     #C (x_plus, 0) z = 0.5 * tanh(x_plus), h = a*x_plus + c
#     #D (0, y_minus) z = 0, h = b*y_minus + c
#     #O (0,0) z=0, h =c
#     loss1 = 0.5 * torch.tanh(x_minus) - (a*x_minus + c) #A
#     valid = (loss1<=0)
#     # print('loss1', loss1)
#     loss1 = torch.clamp(loss1, min = confidence).mean()
    
#     loss2 = (torch.tanh(x_minus) * torch.sigmoid(y_plus) - 
#              (a*x_minus + b*y_plus + c) ) #x2
#     valid = valid * (loss2<=0)
#     # print('loss2', loss2)
#     loss2 = torch.clamp(loss2, min = confidence).mean()
    
#     loss3 = 0 - (b*y_plus + c)#B
#     valid = valid * (loss3<=0)
#     # print('loss3', loss3)
#     loss3 = torch.clamp(loss3, min = confidence).mean()
    
#     loss4 = 0 - c #O
#     valid = valid * (loss4<=0)
#     # print('loss4', loss4)
#     loss4 = torch.clamp(loss4, min = confidence).mean()
    
#     loss5 = 0 - (b*y_minus + c) #D
#     valid = valid * (loss5<=0)
#     # print('loss5', loss5)
#     loss5 = torch.clamp(loss5, min = confidence).mean()
    
#     #D-x_4
#     loss6 = tanh_loss(a,b,c, x_plus, y_minus, tanh_l_max, 
#                       confidence=confidence)
#     valid = valid * (loss6<=0)
#     # print('loss6', loss6)
#     loss6 = loss6.mean()
#     #A-x_1
#     loss7 = sigmoid_loss(a,b,c, x_minus, y_minus, sigmoid_l_max, 
#                          confidence=confidence)
#     valid = valid * (loss7<=0)
#     # print('loss7', loss7)
#     loss7 = loss7.mean()
    
#     loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7
#     return loss, valid

# def qualification_loss_lower(a,b,c,x_minus, x_plus, y_minus, y_plus, 
#                              tanh_l_min,
#                        sigmoid_l_min, confidence=-0.01):
#     #check whether a*x + b*y + c is above z(x) in the rectangular area
#     #we require loss1-loss7 <=0
    
#     #h(x,y) = a*x + b*y + c
#     #z(x,y) = tanh(x) * sigmoid(y)
#     #x1 (x_minus, y_minus)
#     #x2 (x_minus, y_plus)
#     #x3 (x_plus, y_plus)
#     #x4 (x_plus, y_minus)
#     #A (x_minus, 0) z = 0.5 * tanh(x_minus), h = a*x_minus + c
#     #B (0, y_plus) z = 0, h = b*y_plus + c
#     #C (x_plus, 0) z = 0.5 * tanh(x_plus), h = a*x_plus + c
#     #D (0, y_minus) z = 0, h = b*y_minus + c
#     #O (0,0) z=0, h =c
#     # h-z <=0, minimize h-z
    
    
#     loss1 = b*y_plus + c #B
#     valid = (loss1<=0)
#     # print('loss1', loss1)
#     loss1 = torch.clamp(loss1, min = confidence).mean()
    
#     loss2 = ((a*x_plus + b*y_plus + c)-
#              torch.tanh(x_plus) * torch.sigmoid(y_plus)) #x3
#     valid = valid * (loss2<=0)
#     # print('loss2', loss2)
#     loss2 = torch.clamp(loss2, min = confidence).mean()
    
#     loss3 = (a*x_plus + c) - 0.5 * torch.tanh(x_plus)#C
#     valid = valid * (loss3<=0)
#     # print('loss3', loss3)
#     loss3 = torch.clamp(loss3, min = confidence).mean()
    
#     loss4 = c - 0 #O
#     valid = valid * (loss4<=0)
#     # print('loss4', loss4)
#     loss4 = torch.clamp(loss4, min = confidence).mean()
    
#     loss5 = (b*y_minus + c) - 0 #D
#     valid = valid * (loss5<=0)
#     # print('loss5', loss5)
#     loss5 = torch.clamp(loss5, min = confidence).mean()
    
#     #D-x_4
    
#     loss6 = tanh_loss_lower(a,b,c, x_minus, y_minus, tanh_l_min, 
#                       confidence=confidence)
#     valid = valid * (loss6<=0)
#     # print('loss6', loss6)
#     loss6 = loss6.mean()
#     #A-x_1
    
#     loss7 = sigmoid_loss_lower(a,b,c, x_plus, y_minus, sigmoid_l_min, 
#                          confidence=confidence)
#     valid = valid * (loss7<=0)
#     # print('loss7', loss7)
#     loss7 = loss7.mean()
    
#     loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7
#     return loss, valid
    
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
    Z = np.tanh(X)*sigmoid(Y)
    
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
    # fig = plt.figure()
    
    # plt.contour(X, Y, Z-H)
    # plt.colorbar()
    return H-Z

# from matplotlib.ticker import MultipleLocator
# import scipy.io as sio
def plot_2_surface(x_minus, x_plus,y_minus, y_plus, a1,b1,c1,
                 a2,b2,c2, plot=True, num_points=30):
    #a1,b1,c1 should be the upper plane
    #a2,b2,c2 should be the lower plane
    x_minus, x_plus,y_minus, y_plus = x_minus.item(), x_plus.item(), y_minus.item(), y_plus.item()
    
    n = num_points
    x = np.linspace(x_minus,x_plus,n)
    y = np.linspace(y_minus,y_plus,n)
    X, Y = np.meshgrid(x, y)
    Z = np.tanh(X)*sigmoid(Y)
    
    if plot:
        fig = plt.figure(figsize=[15,12])
        ax = Axes3D(fig)
        ax.plot_surface(X, Y, Z)
        fontsize = 20
        ax.set_xlabel('z', fontsize=fontsize)
        ax.set_ylabel('v', fontsize=fontsize)
        ax.set_zlim3d(-1.3, 1.3)
        # ax.set_zlabel('Z', fontsize=fontsize)
        # ax.grid(b=True, which='both',axis='x',linewidth=10)
        # grid(color='r', linestyle='-', linewidth=2)
        linewidth = 0
        color = 'grey'
        ax.xaxis._axinfo["grid"].update({"linewidth":linewidth, "color" : color})
        ax.yaxis._axinfo["grid"].update({"linewidth":linewidth, "color" : color})
        ax.zaxis._axinfo["grid"].update({"linewidth":linewidth, "color" : color})
        # intervals = float(sys.argv[1])
        # ax.xaxis._axinfo['tickdir'] = 2
        # ax.xaxis._axinfo['tick'].update({'inward_factor': 0.4,
        #                 'outward_factor': 0.2})
        # minorLocator = MultipleLocator(base=2)
        # ax.yaxis.set_minor_locator(minorLocator)
        # ax.xaxis.set_minor_locator(minorLocator)
    
    a1,b1,c1 = a1.item(), b1.item(), c1.item()
    H1 = a1*X + b1*Y + c1
    
    if plot:
        ax.plot_surface(X, Y, H1)
    
    a2,b2,c2 = a2.item(), b2.item(), c2.item()
    H2 = a2*X + b2*Y + c2
    if plot:
        ax.plot_surface(X, Y, H2)
        plt.show()
    # sio.savemat('planes.mat', {'X':X, 'Y':Y, 'Z':Z,
    #                            'H1':H1, 'H2':H2})
    return H1-Z, H2-Z
    
def det(a11,a12,a13,
        a21,a22,a23,
        a31,a32,a33):
    d = a11*(a22*a33-a23*a32) -a12*(a21*a33-a23*a31)+a13*(a21*a32-a22*a31)
    return d


if __name__ == '__main__':
   print(0)
   num = (1,7)
   plot_2_surface(lstm.yg_l[0][num], lstm.yg_u[0][num], lstm.yi_l[0][num], lstm.yi_u[0][num],
                  lstm.beta_l_ig[0][num],lstm.alpha_l_ig[0][num],lstm.gamma_l_ig[0][num],
              lstm.beta_u_ig[0][num],lstm.alpha_u_ig[0][num],lstm.gamma_u_ig[0][num])
    # a,b,c = plane(x_plus,y_plus)
   # # a=torch.Tensor([1])
   # # b=torch.Tensor([1])
   # # c=torch.Tensor([0])
   # v = get_volume(a,b,c,x_minus, x_plus, y_minus, y_plus)
   # print('a,b,c:',a,b,c)
   # print('v:',v)
   
   # tanh_lmax = tanh_lmax(x_plus, y_minus)
   # sigmoid_lmax = sigmoid_lmax(x_minus, y_minus)
   # print('tanh_lmax/sigmoid(y_minus)', tanh_lmax/torch.sigmoid(y_minus))
   # print('sigmoid_lmax/|tanh(x_minus|)', -sigmoid_lmax/torch.tanh(x_minus))
    
   # q_loss = qualification_loss(a,b,c,x_minus, x_plus, y_minus, y_plus, 
   #                             tanh_lmax,
   #                     sigmoid_lmax, confidence=-10)
    
    
    
    
    