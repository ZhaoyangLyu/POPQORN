#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 09:28:06 2019

@author: root
"""
import sys
sys.path.append('BoundTanhxSigmoidy')

import torch
from utils import plane, get_volume, plot_surface, plot_2_surface
import time


first = __import__('4points_in_1st_orthant')
#main_lower/upper

second = __import__('4points_in_2nd_orthant')
#main_lower/upper

third = __import__('4points_in_3rd_orthant_new')
#main_lower/upper

forth = __import__('4points_in_4th_orthant')
#main_lower/upper

four = __import__('4points_in_4orthants_new')
#main_lower/upper

one_two = __import__('4points_in_12_orthant')
#main_lower/upper

one_four = __import__('4points_in_14_orthant')
#main_lower/upper

two_three_upper = __import__('4points_in_23_orthant_upper_new')
#main_upper

two_three_lower = __import__('4points_in_23_orthant_lower')
#main_lower

three_four = __import__('4points_in_34_orthant')
#main_upper/lower

# import activation_multi_process
from use_1D_line_bound_2D_activation import line_bounding_2D_activation
from use_constant_bound_2D_activation import constant_bounding_2D_activation

def bound_tanh_sigmoid(x_minus, x_plus, y_minus, y_plus,
                       fine_tune_c=True, use_1D_line=False, use_constant=False,
                       print_info = True):
    if (x_minus>x_plus).sum()>0 or (y_minus>y_plus).sum()>0:
        print(x_minus-x_plus, (x_minus-x_plus).max())
        print(y_minus-y_plus, (y_minus-y_plus).max())
        raise Exception('x_plus must be strictly larger than x_minus and y_plus must be strictly larger than y_minus')
    
    if use_1D_line:
        a_l,b_l,c_l,a_u,b_u,c_u = line_bounding_2D_activation(
                x_minus, x_plus, y_minus, y_plus, tanh=True)
        return a_l,b_l,c_l,a_u,b_u,c_u

    if use_constant:
        a_l,b_l,c_l,a_u,b_u,c_u = constant_bounding_2D_activation(
                x_minus, x_plus, y_minus, y_plus, tanh=True)
        return a_l,b_l,c_l,a_u,b_u,c_u
    
    temp = (x_plus-x_minus)<1e-3
    x_plus[temp] = x_plus[temp] + 1e-3
    x_minus[temp] = x_minus[temp] - 1e-3
    
    temp = (y_plus-y_minus)<1e-3
    y_plus[temp] = y_plus[temp] + 1e-3
    y_minus[temp] = y_minus[temp] - 1e-3
    
    a_l = torch.zeros(x_minus.shape, device = x_minus.device)
    b_l = torch.zeros(x_minus.shape, device = x_minus.device)
    c_l = torch.zeros(x_minus.shape, device = x_minus.device)
    
    a_u = torch.zeros(x_minus.shape, device = x_minus.device)
    b_u = torch.zeros(x_minus.shape, device = x_minus.device)
    c_u = torch.zeros(x_minus.shape, device = x_minus.device)
    
    
    first_orthant = (x_minus>=0) * (y_minus>=0)
    if first_orthant.sum()>0:
        a_l[first_orthant], b_l[first_orthant], c_l[first_orthant] = first.main_lower(
                x_minus[first_orthant], x_plus[first_orthant], 
                y_minus[first_orthant], y_plus[first_orthant], print_info = print_info)
        
        a_u[first_orthant], b_u[first_orthant], c_u[first_orthant] = first.main_upper(
                x_minus[first_orthant], x_plus[first_orthant], 
                y_minus[first_orthant], y_plus[first_orthant], print_info = print_info)
        
    second_orthant = (x_plus<=0) * (y_minus>=0)
    if second_orthant.sum()>0:
        a_l[second_orthant], b_l[second_orthant], c_l[second_orthant] = second.main_lower(
                x_minus[second_orthant], x_plus[second_orthant], 
                y_minus[second_orthant], y_plus[second_orthant], print_info = print_info)
        
        a_u[second_orthant], b_u[second_orthant], c_u[second_orthant] = second.main_upper(
                x_minus[second_orthant], x_plus[second_orthant], 
                y_minus[second_orthant], y_plus[second_orthant], print_info = print_info)
        
    third_orthant = (x_plus<=0) * (y_plus<=0)
    if third_orthant.sum()>0:
        a_l[third_orthant], b_l[third_orthant], c_l[third_orthant] = third.main_lower(
                x_minus[third_orthant], x_plus[third_orthant], 
                y_minus[third_orthant], y_plus[third_orthant], print_info = print_info)
        
        a_u[third_orthant], b_u[third_orthant], c_u[third_orthant] = third.main_upper(
                x_minus[third_orthant], x_plus[third_orthant], 
                y_minus[third_orthant], y_plus[third_orthant], print_info = print_info)
        
    forth_orthant = (x_minus>=0) * (y_plus<=0)
    if forth_orthant.sum()>0:
        a_l[forth_orthant], b_l[forth_orthant], c_l[forth_orthant] = forth.main_lower(
                x_minus[forth_orthant], x_plus[forth_orthant], 
                y_minus[forth_orthant], y_plus[forth_orthant], print_info = print_info)
        
        a_u[forth_orthant], b_u[forth_orthant], c_u[forth_orthant] = forth.main_upper(
                x_minus[forth_orthant], x_plus[forth_orthant], 
                y_minus[forth_orthant], y_plus[forth_orthant], print_info = print_info)
        
    four_orthant = (x_minus<0) * (x_plus>0) * (y_minus<0) * (y_plus>0)
    if four_orthant.sum()>0:
        a_l[four_orthant], b_l[four_orthant], c_l[four_orthant] = four.main_lower(
                x_minus[four_orthant], x_plus[four_orthant], 
                y_minus[four_orthant], y_plus[four_orthant], print_info = print_info)
        
        a_u[four_orthant], b_u[four_orthant], c_u[four_orthant] = four.main_upper(
                x_minus[four_orthant], x_plus[four_orthant], 
                y_minus[four_orthant], y_plus[four_orthant], print_info = print_info)
    
    one_two_orthant = (x_minus<0) * (x_plus>0) * (y_minus>=0)
    if one_two_orthant.sum()>0:
        a_l[one_two_orthant], b_l[one_two_orthant], c_l[one_two_orthant] = one_two.main_lower(
                x_minus[one_two_orthant], x_plus[one_two_orthant], 
                y_minus[one_two_orthant], y_plus[one_two_orthant], print_info = print_info)
        
        a_u[one_two_orthant], b_u[one_two_orthant], c_u[one_two_orthant] = one_two.main_upper(
                x_minus[one_two_orthant], x_plus[one_two_orthant], 
                y_minus[one_two_orthant], y_plus[one_two_orthant], print_info = print_info)
        
    two_three_orthant = (x_plus<=0) * (y_minus<0) * (y_plus>0)
    if two_three_orthant.sum()>0:
        a_l[two_three_orthant], b_l[two_three_orthant], c_l[two_three_orthant] = two_three_lower.main_lower(
                x_minus[two_three_orthant], x_plus[two_three_orthant], 
                y_minus[two_three_orthant], y_plus[two_three_orthant], print_info = print_info)
        
        a_u[two_three_orthant], b_u[two_three_orthant], c_u[two_three_orthant] = two_three_upper.main_upper(
                x_minus[two_three_orthant], x_plus[two_three_orthant], 
                y_minus[two_three_orthant], y_plus[two_three_orthant], print_info = print_info)
    
    three_four_orthant = (x_minus<0) * (x_plus>0) * (y_plus<=0)
    if three_four_orthant.sum()>0:
        a_l[three_four_orthant], b_l[three_four_orthant], c_l[three_four_orthant] = three_four.main_lower(
                x_minus[three_four_orthant], x_plus[three_four_orthant], 
                y_minus[three_four_orthant], y_plus[three_four_orthant], print_info = print_info)
        
        a_u[three_four_orthant], b_u[three_four_orthant], c_u[three_four_orthant] = three_four.main_upper(
                x_minus[three_four_orthant], x_plus[three_four_orthant], 
                y_minus[three_four_orthant], y_plus[three_four_orthant], print_info = print_info)
    
    one_four_orthant = (y_minus<0) * (y_plus>0) * (x_minus>=0)
    if one_four_orthant.sum()>0:
        a_l[one_four_orthant], b_l[one_four_orthant], c_l[one_four_orthant] = one_four.main_lower(
                x_minus[one_four_orthant], x_plus[one_four_orthant], 
                y_minus[one_four_orthant], y_plus[one_four_orthant], print_info = print_info)
        
        a_u[one_four_orthant], b_u[one_four_orthant], c_u[one_four_orthant] = one_four.main_upper(
                x_minus[one_four_orthant], x_plus[one_four_orthant], 
                y_minus[one_four_orthant], y_plus[one_four_orthant], print_info = print_info)
    
    
    
    add_idx = (first_orthant + second_orthant + third_orthant + forth_orthant
                + four_orthant + one_two_orthant + two_three_orthant + 
                three_four_orthant + one_four_orthant)
    if (add_idx != 1).sum() > 0:
        raise Exception('not all cases are included or some cases are computed more than once')
    
    if fine_tune_c:
        c_l,c_u = validate(a_l,b_l,c_l,a_u,b_u,c_u,x_minus, x_plus, y_minus, y_plus, 
                           verify_and_modify_all = True,
                           max_iter=100, plot=False, eps=1e8, print_info = print_info)
    return a_l,b_l,c_l,a_u,b_u,c_u


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
            print('tanh sigmoid iter: %d num: %d hl-f max %.6f mean %.6f hu-f min %.6f mean %.6f' % 
                (i,n, hl_fl.max(), hl_fl.mean(), hu_fu.min(), hu_fu.mean()))
        if hl_fl.max() > eps: #we want hl_fl.max() < 0
            print(x_minus_new[n], x_plus_new[n],y_minus_new[n], y_plus_new[n], 
                                       a_l_new[n],b_l_new[n],c_l_new[n],
                                       a_u_new[n],b_u_new[n],c_u_new[n])
            plot_surface(x_minus_new[n], x_plus_new[n],y_minus_new[n], y_plus_new[n], 
                                       a_l_new[n],b_l_new[n],c_l_new[n])
            print('hl-f max',hl_fl.max())
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
            print('hu-f min',hu_fu.min())
            raise Exception('upper plane fail')
            break
        if hu_fu.min()<0 and verify_and_modify_all:
            c_u_new[n] = c_u_new[n] - hu_fu.min() * 2
    c_l_new = c_l_new.view(original_shape)
    c_u_new = c_u_new.view(original_shape)
    return c_l_new, c_u_new

if __name__ == '__main__':
    # tensor([-0.0914]) tensor([4.0102]) tensor([0.4729]) 
    # tensor([5.1217]) tensor([0.]) tensor([0.]) tensor([0.]) 
    # tensor([0.0648]) tensor([0.0559]) tensor([0.6228])
    length = 3
    x_minus = torch.Tensor([-length])
    x_plus = torch.Tensor([length])
    y_minus = torch.Tensor([-1])
    y_plus = torch.Tensor([length])
    
    num = [0]
    device = torch.device('cpu')
    # x_minus = ((torch.rand(num, device=device) - 0.5) * 10)
    # x_plus = (torch.rand(num, device=device)*5 + x_minus)
    # y_minus = ((torch.rand(num, device=device)-0.5) * 10)
    # y_plus = (torch.rand(num, device=device)*5 + y_minus)
    
    print_info = False
    start = time.time()
    a_l,b_l,c_l,a_u,b_u,c_u = bound_tanh_sigmoid(x_minus, x_plus, y_minus, y_plus,
                            fine_tune_c=False,use_1D_line=False,
                            use_constant=False, print_info = print_info)
    end = time.time()
    v1, v2 = plot_2_surface(x_minus[num], x_plus[num],y_minus[num], y_plus[num], 
                 a_l[num],b_l[num],c_l[num],a_u[num], b_u[num], c_u[num])
    # validate(a_l,b_l,c_l,a_u,b_u,c_u,x_minus, x_plus, y_minus, y_plus, 
    #           max_iter=100,plot=False, eps=1e-4, print_info = print_info)
    print('time used:',end-start)
    






