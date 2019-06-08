#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 09:39:20 2019

@author: root
"""

upper = __import__('4points_in_23_orthant_upper_new')
lower = __import__('4points_in_23_orthant_lower')
import utils
import torch

def main_upper(x_minus, x_plus, y_minus, y_plus, plot=False, num=0, print_info = True):
    if print_info:
        print('14 orthant upper: using 23 orthant lower function')
    x_minus_new = -x_plus
    x_plus_new = -x_minus
    
    a,b,c = lower.main_lower(x_minus_new, x_plus_new, 
                            y_minus, y_plus, print_info = print_info)
    b = -b
    c = -c
    if plot:
        utils.plot_surface(x_minus[num], x_plus[num],y_minus[num], y_plus[num],
                           a[num],b[num],c[num])
    return a.detach(),b.detach(),c.detach()


def main_lower(x_minus, x_plus, y_minus, y_plus, plot=False, num=0, print_info = True):
    if print_info:
        print('14 orthant lower: using 23 orthant upper function')
    x_minus_new = -x_plus
    x_plus_new = -x_minus
    
    a,b,c = upper.main_upper(x_minus_new, x_plus_new, 
                            y_minus, y_plus, print_info = print_info)
    b = -b
    c = -c
    if plot:
        utils.plot_surface(x_minus[num], x_plus[num],y_minus[num], y_plus[num],
                           a[num],b[num],c[num])
    return a.detach(),b.detach(),c.detach()   

if __name__ == '__main__':
    
    x_minus = torch.Tensor([0])
    x_plus = torch.Tensor([1])
    y_minus = torch.Tensor([-1])
    y_plus = torch.Tensor([1])

    print_info = False
    a_u,b_u,c_u = main_upper(x_minus, x_plus, y_minus, y_plus, print_info = print_info)
    a_l,b_l,c_l = main_lower(x_minus, x_plus, y_minus, y_plus, print_info = print_info)
    num = 0
    v1, v2 = utils.plot_2_surface(x_minus[num], x_plus[num],y_minus[num], y_plus[num], 
                 a_l[num],b_l[num],c_l[num],a_u[num], b_u[num], c_u[num])

