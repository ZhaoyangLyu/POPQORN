#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 18:58:16 2019

@author: root
"""

import torch
import torch.nn as nn
import lstm_utils
import bound_tanhx_sigmoidy as tanh_sigmoid
import bound_x_sigmoidy as x_sigmoid
from get_bound_for_general_activation_function import getConvenientGeneralActivationBound

import copy

def get_q(p):
    if p == 1:
        q = float('inf')
    elif p == 'inf' or p == float('inf'):
        q = 1 
    else:
        q = p / (p-1)
    return q

class My_lstm(nn.Module):
    def __init__(self,net, device,WF=None, bF=None, seq_len=None, a0=None,
                 c0 = None, print_info = False):
        super(My_lstm, self).__init__()
        self.input_size = net.input_size
        self.hidden_size = net.hidden_size
        Wx = net.weight_ih_l0
        #(W_ii|W_if|W_ig|W_io), of shape (4*hidden_size , input_size)
        self.Wix = Wx[0*self.hidden_size:1*self.hidden_size,:]
        self.Wfx = Wx[1*self.hidden_size:2*self.hidden_size,:]
        self.Wgx = Wx[2*self.hidden_size:3*self.hidden_size,:]
        self.Wox = Wx[3*self.hidden_size:4*self.hidden_size,:]
        
        
        Wa = net.weight_hh_l0
        self.Wia = Wa[0*self.hidden_size:1*self.hidden_size,:]
        self.Wfa = Wa[1*self.hidden_size:2*self.hidden_size,:]
        self.Wga = Wa[2*self.hidden_size:3*self.hidden_size,:]
        self.Woa = Wa[3*self.hidden_size:4*self.hidden_size,:]
        
        #all W (hidden_size, input_size)
        
        b1 = net.bias_ih_l0
        b2 = net.bias_hh_l0
        b = b1+b2
        
        #all b (hidden_size)
        
        self.bi = b[0*self.hidden_size:1*self.hidden_size]
        self.bf = b[1*self.hidden_size:2*self.hidden_size]
        self.bg = b[2*self.hidden_size:3*self.hidden_size]
        self.bo = b[3*self.hidden_size:4*self.hidden_size]
        
        
        self.seq_len = seq_len
        self.init_h()
        self.init_yc()
        self.x = None #(batch, seq_len, input_size)
        
        #final output is W a[:,-1,:] + b
        self.W = WF
        self.b = bF
        self.device = device
        
        #intitial hidden state and cell state
        self.a0 = a0
        self.c0 = c0
        
        #indicate whether to use 1D line to bound 2D activations  
        self.use_1D_line = False
        self.use_constant = False

        self.print_info = print_info
        
    def get_final_output(self, x, a0=None, c0=None):
        if self.W is None or self.b is None:
            raise Exception('The final output W or b is None')
        a,_,_,_,_,_ = self.forward(x, a0=a0, c0=c0)
        a = (a[:,-1,:]).unsqueeze(2) #batch hidden 1
        #W out hidden
        out = torch.matmul(self.W,a).squeeze(2) #batch out
        #b (out)
        out = out + self.b.unsqueeze(0)
        return out
        
    def init_h(self):
        #important, in the paper Irene use alpha, beta, gamma
        #in my code, I use a,b,c
        #alpha corresponds to b
        #beta corresponds to a
        #gamma corresponds to c
        #initialize upper/lower plane coefficients
        #each time step bound sigmoid(yo[:,m,:])*tanh(c[:,m,:])
        # initial_value = torch.zeros(batch, self.seq_len, self.hidden_size)
        initial_value = [None] * self.seq_len 
        #each element is expected to have size (batch, hidden)
        self.alpha_l_oc = copy.deepcopy(initial_value)
        self.beta_l_oc = copy.deepcopy(initial_value)
        self.gamma_l_oc = copy.deepcopy(initial_value)
        
        self.alpha_u_oc = copy.deepcopy(initial_value)
        self.beta_u_oc = copy.deepcopy(initial_value)
        self.gamma_u_oc = copy.deepcopy(initial_value)
        
        #each time step bound c[:,m-1,:]*sigmoid(yf[:,m,:])
        self.alpha_l_fc = copy.deepcopy(initial_value)
        self.beta_l_fc = copy.deepcopy(initial_value)
        self.gamma_l_fc = copy.deepcopy(initial_value)
        
        self.alpha_u_fc = copy.deepcopy(initial_value)
        self.beta_u_fc = copy.deepcopy(initial_value)
        self.gamma_u_fc = copy.deepcopy(initial_value)
        
        #each time step bound sigmoid(yi[:,m,:])*tanh(yg[:,m,:])
        self.alpha_l_ig = copy.deepcopy(initial_value)
        self.beta_l_ig = copy.deepcopy(initial_value)
        self.gamma_l_ig = copy.deepcopy(initial_value)
        
        self.alpha_u_ig = copy.deepcopy(initial_value)
        self.beta_u_ig = copy.deepcopy(initial_value)
        self.gamma_u_ig = copy.deepcopy(initial_value)
    
    def get_hfc(self, m):
        #compute hfc of the m time step
        #m could range from 1 to seq_len
        #bound c[:,m-1,:]*sigmoid(yf[:,m,:])
        if m>1:
            b_l,a_l,c_l,b_u,a_u,c_u = x_sigmoid.main(
                    self.c_l[m-1-1], self.c_u[m-1-1], 
                    self.yf_l[m-1], self.yf_u[m-1],
                    use_1D_line = self.use_1D_line,
                    use_constant = self.use_constant, 
                    print_info = self.print_info)
            self.alpha_l_fc[m-1] = a_l.detach()
            self.alpha_u_fc[m-1] = a_u.detach()
            self.beta_l_fc[m-1] = b_l.detach()
            self.beta_u_fc[m-1] = b_u.detach()
            self.gamma_l_fc[m-1] = c_l.detach()
            self.gamma_u_fc[m-1] = c_u.detach()
            return a_l,b_l,c_l,a_u,b_u,c_u
        if m == 1:
            # bound c0 * sigmoid(yf1)
            zeros = torch.zeros(self.yf_l[m-1].shape, device=self.device)
            if self.c0 is None:
                self.alpha_l_fc[m-1] = zeros.data.clone()
                self.alpha_u_fc[m-1] = zeros.data.clone()
                self.beta_l_fc[m-1] = zeros.data.clone()
                self.beta_u_fc[m-1] = zeros.data.clone()
                self.gamma_l_fc[m-1] = zeros.data.clone()
                self.gamma_u_fc[m-1] = zeros.data.clone()
            else:
                # bound c0 * sigmoid(yf1)
                #c0 is constant, we only need to bound 1d sigmoid
                #alpha * yf + beta * c + gamma
                #kl * yf1 + bl <= sigmoid(yf1) <= ku * yf1 + bu
                #if c0_i >= 0
                #c0_i (kl_i * yf1_i + bl_i) <= c0_i sigmoid(yf1)_i <= c0_i (ku_i * yf1_i + bu_i)
                ##if c0_i < 0
                #co_i (kl_u * yf1_i + bu_i) <= c0_i sigmoid(yf1)_i <= c0_i (kl_i * yf1_i + bl_i)
                kl, bl, ku, bu = getConvenientGeneralActivationBound(self.yf_l[m-1],
                                                            self.yf_u[m-1], 'sigmoid')
                I = (self.c0 >= 0).float()
                KU = I * ku + (1-I) * kl
                BU = I * bu + (1-I) * bl
                KL = (1-I) * ku + I * kl
                BL = (1-I) * bu + I * bl
                
                self.alpha_u_fc[m-1] = self.c0 * KU
                self.gamma_u_fc[m-1] = self.c0 * BU
                self.beta_u_fc[m-1] = zeros.data.clone()
                
                self.alpha_l_fc[m-1] = self.c0 * KL
                self.gamma_l_fc[m-1] = self.c0 * BL
                self.beta_l_fc[m-1] = zeros.data.clone()
        return 0
    
    def get_hig(self, m):
        #compute hig of the m time step
        #m could range from 1 to seq_len
        #each time step bound tanh(yg[:,m,:]) * sigmoid(yi[:,m,:])
        # a_l,b_l,c_l,a_u,b_u,c_u = tanh_sigmoid.bound_tanh_sigmoid(
        #     self.yg_l[m-1], self.yg_u[m-1], self.yi_l[m-1], self.yi_u[m-1])
        b_l,a_l,c_l,b_u,a_u,c_u = tanh_sigmoid.bound_tanh_sigmoid(
            self.yg_l[m-1], self.yg_u[m-1], self.yi_l[m-1], self.yi_u[m-1],
            use_1D_line = self.use_1D_line,
            use_constant = self.use_constant,
            print_info = self.print_info)
            
        self.alpha_l_ig[m-1] = a_l.detach()
        self.alpha_u_ig[m-1] = a_u.detach()
        self.beta_l_ig[m-1] = b_l.detach()
        self.beta_u_ig[m-1] = b_u.detach()
        self.gamma_l_ig[m-1] = c_l.detach()
        self.gamma_u_ig[m-1] = c_u.detach()
        return a_l,b_l,c_l,a_u,b_u,c_u
    
    def get_hoc(self, m):
        #compute hoc of the m time step
        #m could range from 1 to seq_len
        #bound tanh(c[:,m,:]) * sigmoid(yo[:,m,:])
        b_l,a_l,c_l,b_u,a_u,c_u = tanh_sigmoid.bound_tanh_sigmoid(
            self.c_l[m-1].detach(), self.c_u[m-1].detach(), 
            self.yo_l[m-1].detach(), self.yo_u[m-1].detach(),
            use_1D_line = self.use_1D_line,
            use_constant = self.use_constant,
            print_info = self.print_info)
        self.alpha_l_oc[m-1] = a_l.detach()
        self.alpha_u_oc[m-1] = a_u.detach()
        self.beta_l_oc[m-1] = b_l.detach()
        self.beta_u_oc[m-1] = b_u.detach()
        self.gamma_l_oc[m-1] = c_l.detach()
        self.gamma_u_oc[m-1] = c_u.detach()
        return a_l,b_l,c_l,a_u,b_u,c_u
    
    def init_yc(self):
        #initialize the lower/upper bound of y, c and a
        initial_value = [None] * self.seq_len
        self.yi_l = copy.deepcopy(initial_value)
        self.yi_u = copy.deepcopy(initial_value)
        self.yf_l = copy.deepcopy(initial_value)
        self.yf_u = copy.deepcopy(initial_value)
        self.yg_l = copy.deepcopy(initial_value)
        self.yg_u = copy.deepcopy(initial_value)
        self.yo_l = copy.deepcopy(initial_value)
        self.yo_u = copy.deepcopy(initial_value)
        self.c_l = copy.deepcopy(initial_value)
        self.c_u = copy.deepcopy(initial_value)
        self.a_l = copy.deepcopy(initial_value)
        self.a_u = copy.deepcopy(initial_value)
    
    def attach_data(self,x):
        self.x = x

    def reset_seq_len(self, seq_len):
        self.seq_len = seq_len
        self.init_h()
        self.init_yc()
        print('seq_len has been reset to %d. y,a,c and plane coefficients have also been re-initialized' % seq_len)
        return 0  
      
    def forward(self, x, a0=None, c0=None, use_x_seq_len=False):
        #we already assume a0=c0=0
        # x is of shape (batch, seq_len, input_size)
        #if a0 and c0 are not None, we will use them with priority
        #if they are None, we will check whether self.a0 and self.c0 are None
        #If they are None either, we will assume a0 and c0 are 0
        with torch.no_grad():
            if use_x_seq_len:
                seq_len = x.shape[1]
            else:
                if self.seq_len is None:
                    seq_len = x.shape[1]
                else:
                    seq_len = x.shape[1]
                    if seq_len != self.seq_len:
                        raise Exception('the sequence length of the input is %d, but it should be %d' % (seq_len, self.seq_len))
            batch = x.shape[0]
            a = torch.zeros(batch, seq_len, self.hidden_size, device=x.device)
            c = torch.zeros(batch, seq_len, self.hidden_size, device=x.device)
            yi = torch.zeros(batch, seq_len, self.hidden_size, device=x.device)
            yf = torch.zeros(batch, seq_len, self.hidden_size, device=x.device)
            yg = torch.zeros(batch, seq_len, self.hidden_size, device=x.device)
            yo = torch.zeros(batch, seq_len, self.hidden_size, device=x.device)
            
            for k in range(seq_len):
                if k>0:
                    yi[:,k,:] = (mat_mul(self.Wix, x[:,k,:], self.bi)+
                          mat_mul(self.Wia, a[:,k-1,:]))
                    yf[:,k,:] = (mat_mul(self.Wfx, x[:,k,:], self.bf)+
                          mat_mul(self.Wfa, a[:,k-1,:]))
                    yg[:,k,:] = (mat_mul(self.Wgx, x[:,k,:], self.bg)+
                          mat_mul(self.Wga, a[:,k-1,:]))
                    yo[:,k,:] = (mat_mul(self.Wox, x[:,k,:], self.bo)+
                          mat_mul(self.Woa, a[:,k-1,:]))
                    c[:,k,:] = (torch.sigmoid(yf[:,k,:]) * c[:,k-1,:]+
                               torch.sigmoid(yi[:,k,:]) * torch.tanh(yg[:,k,:]))
                    a[:,k,:] = torch.sigmoid(yo[:,k,:]) * torch.tanh(c[:,k,:])
                else: #k==0
                    if (not a0 is None) and (not c0 is None):
                        yi[:,k,:] = (mat_mul(self.Wix, x[:,k,:], self.bi)+
                              mat_mul(self.Wia, a0))
                        yf[:,k,:] = (mat_mul(self.Wfx, x[:,k,:], self.bf)+
                              mat_mul(self.Wfa, a0))
                        yg[:,k,:] = (mat_mul(self.Wgx, x[:,k,:], self.bg)+
                              mat_mul(self.Wga, a0))
                        yo[:,k,:] = (mat_mul(self.Wox, x[:,k,:], self.bo)+
                              mat_mul(self.Woa, a0))
                        c[:,k,:] = (torch.sigmoid(yf[:,k,:]) * c0+
                                   torch.sigmoid(yi[:,k,:]) * torch.tanh(yg[:,k,:]))
                        a[:,k,:] = torch.sigmoid(yo[:,k,:]) * torch.tanh(c[:,k,:])
                    elif (not self.a0 is None) and (not self.c0 is None):
                        #Wa shape (hidden, hidden)
                        #a0 shape (bacth, hidden)
                        yi[:,k,:] = (mat_mul(self.Wix, x[:,k,:], self.bi)+
                              mat_mul(self.Wia, self.a0))
                        yf[:,k,:] = (mat_mul(self.Wfx, x[:,k,:], self.bf)+
                              mat_mul(self.Wfa, self.a0))
                        yg[:,k,:] = (mat_mul(self.Wgx, x[:,k,:], self.bg)+
                              mat_mul(self.Wga, self.a0))
                        yo[:,k,:] = (mat_mul(self.Wox, x[:,k,:], self.bo)+
                              mat_mul(self.Woa, self.a0))
                        c[:,k,:] = (torch.sigmoid(yf[:,k,:]) * self.c0+
                                   torch.sigmoid(yi[:,k,:]) * torch.tanh(yg[:,k,:]))
                        a[:,k,:] = torch.sigmoid(yo[:,k,:]) * torch.tanh(c[:,k,:])
                    else:
                        yi[:,k,:] = mat_mul(self.Wix, x[:,k,:], self.bi)
                        yf[:,k,:] = mat_mul(self.Wfx, x[:,k,:], self.bf)
                        yg[:,k,:] = mat_mul(self.Wgx, x[:,k,:], self.bg)
                        yo[:,k,:] = mat_mul(self.Wox, x[:,k,:], self.bo)
                        c[:,k,:] = torch.sigmoid(yi[:,k,:]) * torch.tanh(yg[:,k,:])
                        a[:,k,:] = torch.sigmoid(yo[:,k,:]) * torch.tanh(c[:,k,:])
        return a,c,yi,yf,yg,yo
        
    def get_Wa_b(self,W,b,m,x,eps,p, save_a=False, eps_idx=None):
        #m could range from 0 to seq_len
        #compute the minimum and maximum of W a_m + b
        
        #eps could be a number or a tensor of shape (batch,1)
        
        #only when W is identity and b = 0
        #save_a should be set to True
        #the results will be saved to self.al_m, self.au_m
        
        #eps_idx should be a tesnor of shape (seq_len), indicate whether to
        #perturb the corresponding frame
        with torch.no_grad():
            if eps_idx is None:
                eps_idx = torch.ones(self.seq_len, device=x.device)
            if len(W.shape) == 2:
                W = W.unsqueeze(0)
            if m == 0:
                #return minimum and maximum
                if not self.a0 is None:
                    Wa0_b = mat_mul(W,self.a0,b=b)
                else:
                    Wa0_b = b
                return Wa0_b, Wa0_b
            else:
                # q = 1/(1-1/p)
                q = get_q(p)
                #first step k = m
                Wx_l,Wx_u, Wa_l, Wa_u, b_l, b_u, A_fc_delta,Ou_fc_theta = self.get_Wa_b_one_step(
                        m, W, W, 0, 0)
                
                maximum = (mat_mul(Wx_u, x[:,m-1,:]) + 
                               eps_idx[m-1]*eps*torch.norm(Wx_u, p=q,dim=2) +
                               b_u+b)
                minimum = (mat_mul(Wx_l, x[:,m-1,:]) - 
                               eps_idx[m-1]*eps*torch.norm(Wx_l, p=q,dim=2) +
                               b_l+b)
                
                # if m=1, Wa, A_fc_delta and Ou_fc_theta we get here are
                # Wa1, A_fc_delta1 and Ou_fc_theta1
                
                if m>1:
                    for k in range(m-1, 0,-1):
                        #k from m-1 to 1
                        Wx_l,Wx_u, Wa_l, Wa_u, b_l, b_u, A_fc_delta,Ou_fc_theta = self.get_Wa_b_one_step(
                                k, Wa_l, Wa_u, A_fc_delta, Ou_fc_theta)
                        maximum = (maximum + mat_mul(Wx_u, x[:,k-1,:]) + 
                                   eps_idx[k-1]*eps*torch.norm(Wx_u, p=q,dim=2) + b_u)
                        
                        minimum = (minimum + mat_mul(Wx_l, x[:,k-1,:]) - 
                                   eps_idx[k-1]*eps*torch.norm(Wx_l, p=q,dim=2) + b_l)
                    # if m=1, after the loop,  Wa, A_fc_delta and Ou_fc_theta we get here are
                    # Wa1, A_fc_delta1 and Ou_fc_theta1
                if not self.a0 is None:
                    minimum = minimum + mat_mul(Wa_l, self.a0)
                    maximum = maximum + mat_mul(Wa_u, self.a0)
                if not self.c0 is None:
                    minimum = minimum + mat_mul(Ou_fc_theta, self.c0)
                    maximum = maximum + mat_mul(A_fc_delta, self.c0)
                if save_a:
                    self.a_l[m-1] = minimum
                    self.a_u[m-1] = maximum
                return minimum, maximum
        return 0
                        
    def get_Wa_b_one_step(self, k, Wa_l, Wa_u, A_fc_delta, Ou_fc_theta):
        #k could range from 1 to m
        #if k=m, Wa_l and Wa_u should be W_Fa, 
        #A_fc_delta, Ou_fc_theta should be 0
        #if k<m, Wa_l = Wa_l_k+1 , Wa_u = Wa_u_k+1
        #A_fc_delta = A_fc_delta_k+1, Ou_fc_theta = Ou_fc_theta_k+1
        with torch.no_grad():
            #part1
            lamida_oc, delta_oc, fai_oc, omiga_oc, theta_oc, psai_oc = separate_group(
                        self.alpha_l_oc[k-1], self.alpha_u_oc[k-1], 
                        self.beta_l_oc[k-1], self.beta_u_oc[k-1], 
                        self.gamma_l_oc[k-1], self.gamma_u_oc[k-1], 
                        Wa_l, Wa_u)
            
            #part2
            A_oc_lamida = Wa_u * lamida_oc
            A_oc_delta = Wa_u * delta_oc + A_fc_delta
            A_oc_fai = Wa_u * fai_oc
            
            Ou_oc_omiga = Wa_l * omiga_oc
            Ou_oc_theta = Wa_l * theta_oc + Ou_fc_theta
            Ou_oc_psai = Wa_l * psai_oc
            
            #part3
            lamida_fc, delta_fc, fai_fc, omiga_fc, theta_fc, psai_fc = separate_group(
                        self.alpha_l_fc[k-1], self.alpha_u_fc[k-1], 
                        self.beta_l_fc[k-1], self.beta_u_fc[k-1], 
                        self.gamma_l_fc[k-1], self.gamma_u_fc[k-1], 
                        A_oc_delta, Ou_oc_theta)
                
            lamida_ig, delta_ig, fai_ig, omiga_ig, theta_ig, psai_ig = separate_group(
                    self.alpha_l_ig[k-1], self.alpha_u_ig[k-1], 
                    self.beta_l_ig[k-1], self.beta_u_ig[k-1], 
                    self.gamma_l_ig[k-1], self.gamma_u_ig[k-1], 
                    A_oc_delta, Ou_oc_theta)
            
            #part4
            A_fc_lamida = A_oc_delta * lamida_fc
            A_fc_delta_k = A_oc_delta * delta_fc
            A_fc_fai = A_oc_delta * fai_fc
            A_ig_lamida = A_oc_delta * lamida_ig
            A_ig_delta = A_oc_delta * delta_ig
            A_ig_fai = A_oc_delta * fai_ig
            
            Ou_fc_omiga = Ou_oc_theta * omiga_fc
            Ou_fc_theta_k = Ou_oc_theta * theta_fc
            Ou_fc_psai = Ou_oc_theta * psai_fc
            Ou_ig_omiga = Ou_oc_theta * omiga_ig
            Ou_ig_theta = Ou_oc_theta * theta_ig
            Ou_ig_psai = Ou_oc_theta * psai_ig
            
            #part 5
            Wx_u_k = (torch.matmul(A_oc_lamida, self.Wox)+
                      torch.matmul(A_fc_lamida, self.Wfx)+
                      torch.matmul(A_ig_lamida, self.Wix)+
                      torch.matmul(A_ig_delta, self.Wgx))
            Wa_u_k = (torch.matmul(A_oc_lamida, self.Woa)+
                      torch.matmul(A_fc_lamida, self.Wfa)+
                      torch.matmul(A_ig_lamida, self.Wia)+
                      torch.matmul(A_ig_delta, self.Wga))
            b_u_k = (mat_mul(A_oc_lamida, self.bo)+
                     mat_mul(A_fc_lamida, self.bf)+
                     mat_mul(A_ig_lamida, self.bi)+
                     mat_mul(A_ig_delta, self.bg))
            b_u_k = b_u_k + (A_oc_fai+A_fc_fai+A_ig_fai).sum(dim=2)
            
            Wx_l_k = (torch.matmul(Ou_oc_omiga, self.Wox)+
                      torch.matmul(Ou_fc_omiga, self.Wfx)+
                      torch.matmul(Ou_ig_omiga, self.Wix)+
                      torch.matmul(Ou_ig_theta, self.Wgx))
            Wa_l_k = (torch.matmul(Ou_oc_omiga, self.Woa)+
                      torch.matmul(Ou_fc_omiga, self.Wfa)+
                      torch.matmul(Ou_ig_omiga, self.Wia)+
                      torch.matmul(Ou_ig_theta, self.Wga))
            b_l_k = (mat_mul(Ou_oc_omiga, self.bo)+
                     mat_mul(Ou_fc_omiga, self.bf)+
                     mat_mul(Ou_ig_omiga, self.bi)+
                     mat_mul(Ou_ig_theta, self.bg))
            b_l_k = b_l_k + (Ou_oc_psai+Ou_fc_psai+Ou_ig_psai).sum(dim=2)
        return Wx_l_k,Wx_u_k, Wa_l_k, Wa_u_k, b_l_k, b_u_k, A_fc_delta_k,Ou_fc_theta_k        
            
    def get_y(self, m,  eps, x=None, p=2, eps_idx=None):
        #m could range from 1 to seq_len
        #compute the minimum and maximum of yi_m yf_m yg_m y0_m
        #this function is already complete
        
        #eps could be a number or a tensor of shape (batch,1)
        with torch.no_grad():
            if x is None and self.x is None:
                raise Exception('you must feed data to the function or attach data to the model')
            if eps_idx is None:
                eps_idx = torch.ones(self.seq_len, device=x.device)
                
            Wa_b_i_min, Wa_b_i_max = self.get_Wa_b(self.Wia, self.bi, m-1,x,eps,p,eps_idx=eps_idx)
            Wx_i_min, Wx_i_max =  Wx_extreme(self.Wix,x[:,m-1,:],eps=eps*eps_idx[m-1],p=p)
            yi_min = Wa_b_i_min + Wx_i_min
            yi_max = Wa_b_i_max + Wx_i_max
            
            Wa_b_f_min, Wa_b_f_max = self.get_Wa_b(self.Wfa, self.bf, m-1,x,eps,p,eps_idx=eps_idx)
            Wx_f_min, Wx_f_max =  Wx_extreme(self.Wfx,x[:,m-1,:],eps=eps*eps_idx[m-1],p=p)
            yf_min = Wa_b_f_min + Wx_f_min
            yf_max = Wa_b_f_max + Wx_f_max
            
            Wa_b_g_min, Wa_b_g_max = self.get_Wa_b(self.Wga, self.bg, m-1,x,eps,p,eps_idx=eps_idx)
            Wx_g_min, Wx_g_max =  Wx_extreme(self.Wgx,x[:,m-1,:],eps=eps*eps_idx[m-1],p=p)
            yg_min = Wa_b_g_min + Wx_g_min
            yg_max = Wa_b_g_max + Wx_g_max
            
            Wa_b_o_min, Wa_b_o_max = self.get_Wa_b(self.Woa, self.bo, m-1,x,eps,p,eps_idx=eps_idx)
            Wx_o_min, Wx_o_max =  Wx_extreme(self.Wox,x[:,m-1,:],eps=eps*eps_idx[m-1],p=p)
            yo_min = Wa_b_o_min + Wx_o_min
            yo_max = Wa_b_o_max + Wx_o_max
            
            self.yi_l[m-1] = yi_min 
            self.yi_u[m-1] = yi_max
            self.yf_l[m-1] = yf_min
            self.yf_u[m-1] = yf_max
            self.yg_l[m-1] = yg_min
            self.yg_u[m-1] = yg_max
            self.yo_l[m-1] = yo_min
            self.yo_u[m-1] = yo_max
            
        return yi_min,yi_max,yf_min,yf_max,yg_min,yg_max,yo_min,yo_max
    
    def get_c(self, v,  eps, x=None, p=2, eps_idx=None):
        #v could range from 1 to seq_len
        #compute the minimum and maximum of c_v
        
        #eps could be a number or a tensor of shape (batch,1)
        
        #use A to denote capital lambda
        #use Ou to denote capital omega
        with torch.no_grad():
            if x is None and self.x is None:
                raise Exception('you must feed data to the function or attach data to the model')
                
            if eps_idx is None:
                eps_idx = torch.ones(self.seq_len, device=x.device)
            A_fc_lamida_v = batch_diag(self.alpha_u_fc[v-1])
            A_fc_delta_v = batch_diag(self.beta_u_fc[v-1])
            A_fc_fai_v = batch_diag(self.gamma_u_fc[v-1])
            
            A_ig_lamida_v = batch_diag(self.alpha_u_ig[v-1])
            A_ig_delta_v = batch_diag(self.beta_u_ig[v-1])
            A_ig_fai_v = batch_diag(self.gamma_u_ig[v-1])
            
            Ou_fc_omiga_v = batch_diag(self.alpha_l_fc[v-1])
            Ou_fc_theta_v = batch_diag(self.beta_l_fc[v-1])
            Ou_fc_psai_v = batch_diag(self.gamma_l_fc[v-1])
            
            Ou_ig_omiga_v = batch_diag(self.alpha_l_ig[v-1])
            Ou_ig_theta_v = batch_diag(self.beta_l_ig[v-1])
            Ou_ig_psai_v = batch_diag(self.gamma_l_ig[v-1])
            
            #all A's and Ou's are of shape (batch, hidden, hidden)
            Wx_u_v =(torch.matmul(A_fc_lamida_v, self.Wfx) + #(batch, hidden, input)
                     torch.matmul(A_ig_lamida_v, self.Wix) +
                     torch.matmul(A_ig_delta_v, self.Wgx) )
            
            Wa_u_v =(torch.matmul(A_fc_lamida_v, self.Wfa) + #(batch, hidden, hidden)
                     torch.matmul(A_ig_lamida_v, self.Wia) +
                     torch.matmul(A_ig_delta_v, self.Wga) )
            
            b_u_v =(torch.matmul(A_fc_lamida_v, self.bf) + #(batch, hidden)
                     torch.matmul(A_ig_lamida_v, self.bi) +
                     torch.matmul(A_ig_delta_v, self.bg))
            b_u_v = b_u_v + (A_fc_fai_v + A_ig_fai_v).sum(dim=2)
            
            
            Wx_l_v =(torch.matmul(Ou_fc_omiga_v, self.Wfx) + #(batch, hidden, input)
                     torch.matmul(Ou_ig_omiga_v, self.Wix) +
                     torch.matmul(Ou_ig_theta_v, self.Wgx) )
            
            Wa_l_v =(torch.matmul(Ou_fc_omiga_v, self.Wfa) + #(batch, hidden, hidden)
                     torch.matmul(Ou_ig_omiga_v, self.Wia) +
                     torch.matmul(Ou_ig_theta_v, self.Wga) )
            
            b_l_v =(torch.matmul(Ou_fc_omiga_v, self.bf) + #(batch, hidden)
                     torch.matmul(Ou_ig_omiga_v, self.bi) +
                     torch.matmul(Ou_ig_theta_v, self.bg))
            b_l_v = b_l_v + (Ou_fc_psai_v + Ou_ig_psai_v).sum(dim=2)
            
            
            # q = 1/(1-1/p)
            q = get_q(p)
            maximum = (mat_mul(Wx_u_v, x[:,v-1,:]) + 
                       eps_idx[v-1]*eps*torch.norm(Wx_u_v, p=q,dim=2) +
                       b_u_v)
            #Wx (batch, hidden, input)
            #x (batch, input)
            minimum = (mat_mul(Wx_l_v, x[:,v-1,:]) - 
                       eps_idx[v-1]*eps*torch.norm(Wx_l_v, p=q,dim=2) +
                       b_l_v)
            if v==1:
                if not self.a0 is None:
                    maximum = maximum + mat_mul(Wa_u_v, self.a0)
                    minimum = minimum + mat_mul(Wa_l_v, self.a0)
                if not self.c0 is None:
                    maximum = maximum + mat_mul(A_fc_delta_v, self.c0)
                    minimum = minimum + mat_mul(Ou_fc_theta_v, self.c0)
            elif v>1:
                Wa_l = Wa_l_v
                Wa_u = Wa_u_v
                A_fc_delta = A_fc_delta_v
                Ou_fc_theta = Ou_fc_theta_v
                for k in range(v-1,0,-1):
                    # print('in the iterration')
                    Wx_l,Wx_u, Wa_l, Wa_u, b_l, b_u, A_fc_delta,Ou_fc_theta = self.get_c_one_step(
                            k, Wa_l, Wa_u, A_fc_delta, Ou_fc_theta)
                    
                    maximum = (maximum + mat_mul(Wx_u, x[:,k-1,:]) + 
                       eps_idx[k-1]*eps*torch.norm(Wx_u, p=q,dim=2) + b_u)
                    minimum = (minimum + mat_mul(Wx_l, x[:,k-1,:]) - 
                       eps_idx[k-1]*eps*torch.norm(Wx_l, p=q,dim=2) + b_l)
                    
                if not self.a0 is None:
                    maximum = maximum + mat_mul(Wa_u, self.a0)
                    minimum = minimum + mat_mul(Wa_l, self.a0)
                if not self.c0 is None:
                    maximum = maximum + mat_mul(A_fc_delta, self.c0)
                    minimum = minimum + mat_mul(Ou_fc_theta, self.c0)
                    
            self.c_l[v-1] = minimum
            self.c_u[v-1] = maximum
            # print(maximum-minimum)
            
            
            return minimum, maximum
            
        return 0
    
    def get_c_one_step(self, k, Wa_l, Wa_u, A_fc_delta, Ou_fc_theta):
        #k could range from 1 to v-1
        # Wa_l = Wa_l_k+1 , Wa_u = Wa_u_k+1
        #A_fc_delta = A_fc_delta_k+1, Ou_fc_theta = Ou_fc_theta_k+1
        with torch.no_grad():
            #part1
            lamida_oc, delta_oc, fai_oc, omiga_oc, theta_oc, psai_oc = separate_group(
                        self.alpha_l_oc[k-1], self.alpha_u_oc[k-1], 
                        self.beta_l_oc[k-1], self.beta_u_oc[k-1], 
                        self.gamma_l_oc[k-1], self.gamma_u_oc[k-1], 
                        Wa_l, Wa_u)
            
            #part2
            A_oc_lamida = Wa_u * lamida_oc
            A_oc_delta = Wa_u * delta_oc + A_fc_delta
            A_oc_fai = Wa_u * fai_oc
            
            Ou_oc_omiga = Wa_l * omiga_oc
            Ou_oc_theta = Wa_l * theta_oc + Ou_fc_theta
            Ou_oc_psai = Wa_l * psai_oc
            
            #part3
            lamida_fc, delta_fc, fai_fc, omiga_fc, theta_fc, psai_fc = separate_group(
                        self.alpha_l_fc[k-1], self.alpha_u_fc[k-1], 
                        self.beta_l_fc[k-1], self.beta_u_fc[k-1], 
                        self.gamma_l_fc[k-1], self.gamma_u_fc[k-1], 
                        A_oc_delta, Ou_oc_theta)
                
            lamida_ig, delta_ig, fai_ig, omiga_ig, theta_ig, psai_ig = separate_group(
                    self.alpha_l_ig[k-1], self.alpha_u_ig[k-1], 
                    self.beta_l_ig[k-1], self.beta_u_ig[k-1], 
                    self.gamma_l_ig[k-1], self.gamma_u_ig[k-1], 
                    A_oc_delta, Ou_oc_theta)
            
            #part4
            A_fc_lamida = A_oc_delta * lamida_fc
            A_fc_delta_k = A_oc_delta * delta_fc
            A_fc_fai = A_oc_delta * fai_fc
            A_ig_lamida = A_oc_delta * lamida_ig
            A_ig_delta = A_oc_delta * delta_ig
            A_ig_fai = A_oc_delta * fai_ig
            
            Ou_fc_omiga = Ou_oc_theta * omiga_fc
            Ou_fc_theta_k = Ou_oc_theta * theta_fc
            Ou_fc_psai = Ou_oc_theta * psai_fc
            Ou_ig_omiga = Ou_oc_theta * omiga_ig
            Ou_ig_theta = Ou_oc_theta * theta_ig
            Ou_ig_psai = Ou_oc_theta * psai_ig
            
            #part 5
            Wx_u_k = (torch.matmul(A_oc_lamida, self.Wox)+
                      torch.matmul(A_fc_lamida, self.Wfx)+
                      torch.matmul(A_ig_lamida, self.Wix)+
                      torch.matmul(A_ig_delta, self.Wgx))
            Wa_u_k = (torch.matmul(A_oc_lamida, self.Woa)+
                      torch.matmul(A_fc_lamida, self.Wfa)+
                      torch.matmul(A_ig_lamida, self.Wia)+
                      torch.matmul(A_ig_delta, self.Wga))
            b_u_k = (mat_mul(A_oc_lamida, self.bo)+
                     mat_mul(A_fc_lamida, self.bf)+
                     mat_mul(A_ig_lamida, self.bi)+
                     mat_mul(A_ig_delta, self.bg))
            b_u_k = b_u_k + (A_oc_fai+A_fc_fai+A_ig_fai).sum(dim=2)
            
            Wx_l_k = (torch.matmul(Ou_oc_omiga, self.Wox)+
                      torch.matmul(Ou_fc_omiga, self.Wfx)+
                      torch.matmul(Ou_ig_omiga, self.Wix)+
                      torch.matmul(Ou_ig_theta, self.Wgx))
            Wa_l_k = (torch.matmul(Ou_oc_omiga, self.Woa)+
                      torch.matmul(Ou_fc_omiga, self.Wfa)+
                      torch.matmul(Ou_ig_omiga, self.Wia)+
                      torch.matmul(Ou_ig_theta, self.Wga))
            b_l_k = (mat_mul(Ou_oc_omiga, self.bo)+
                     mat_mul(Ou_fc_omiga, self.bf)+
                     mat_mul(Ou_ig_omiga, self.bi)+
                     mat_mul(Ou_ig_theta, self.bg))
            b_l_k = b_l_k + (Ou_oc_psai+Ou_fc_psai+Ou_ig_psai).sum(dim=2)
        return Wx_l_k,Wx_u_k, Wa_l_k, Wa_u_k, b_l_k, b_u_k, A_fc_delta_k,Ou_fc_theta_k 
    
def mat_mul(W,x,b=None):
    #W (hidden,input), W could also be of shape (batch, hidden,input)
    #x (batch, input), x could also be of shape (input)
    #b (hidden)
    with torch.no_grad():
        if len(x.shape) == 2:
            #x.unsqueeze(2) (batch,input,1)
            #W (batch, hidden, input) or (hidden, input)
            z = torch.matmul(W,x.unsqueeze(2)).squeeze(2)#z (batch, hidden)
        elif len(x.shape) == 1:
            #x.unsqueeze(1) (input,1)
            #W (batch, hidden, input) or (hidden, input)
            z = torch.matmul(W,x.unsqueeze(1)).squeeze(2)#z (batch, hidden)
        else:
            raise Exception('x must be 1 dimension or 2 dimension')
        if not (b is None):
            z = z + b.unsqueeze(0)
    return z    
 
def Wx_extreme(W,x0,eps,p=2):
    #W hidden input
    #x batch input
    #eps could be a number or a tensor of shape (batch,1) 
    #return min and max of {Wx | ||x-x0||p <= eps}
    
    with torch.no_grad():
        Wx0 = mat_mul(W,x0) #batch hidden
        
        # q = 1/(1-1/p)
        q = get_q(p)
        W_q_norm = torch.norm(W, p=q, dim=1) #(hidden)
        W_q_norm = W_q_norm.unsqueeze(0) #(1, hidden)
        maximum = Wx0 + eps * W_q_norm
        minimum = Wx0 - eps * W_q_norm
    return minimum, maximum

def separate(alpha_l, alpha_u, Wl, Wu):
    #alpha_l and alpha_u is of shape (batch, hidden_size)
    #Wl and Wu are of shape (batch,M, hidden_size) or (M, hidden_size)
    #lamida[j,r] = alpha_u[r] if Wu[j,r]>=0, alpha_l[r] if Wu[j,r]<0
    #omiga[j,r] = alpha_l[r] if Wl[j,r]>=0, alpha_u[r] if Wl[j,r]<0
    #lamida and omiga are of size (batch, M, hidden)
    if len(Wl.shape) == 2:
        Wl = Wl.unsqueeze(0)
    if len(Wu.shape) == 2:
        Wu = Wu.unsqueeze(0)
    alpha_l = alpha_l.unsqueeze(1) #(batch, 1, hidden_size)
    alpha_u = alpha_u.unsqueeze(1) #(batch, 1, hidden_size)
    
    I_Wu = (Wu>=0).float() #(batch,M, hidden_size) or (1,M, hidden_size)
    lamida = I_Wu * alpha_u + (1-I_Wu) * alpha_l #(batch,M, hidden_size)
    #lamida[j,r] = alpha_u[r] if Wu[j,r]>=0, alpha_l[r] if Wu[j,r]<0
    
    I_Wl = (Wl>=0).float()
    omiga = I_Wl * alpha_l + (1-I_Wl) * alpha_u #(batch,M, hidden_size)
    #omiga[j,r] = alpha_l[r] if Wl[j,r]>=0, alpha_u[r] if Wl[j,r]<0
    return lamida, omiga

def separate_group(alpha_l, alpha_u, beta_l, beta_u, gamma_l, gamma_u, Wl, Wu):
    lamida, omiga = separate(alpha_l, alpha_u, Wl, Wu)
    delta, theta = separate(beta_l, beta_u, Wl, Wu)
    fai, psai = separate(gamma_l, gamma_u, Wl, Wu)
    return lamida, delta, fai, omiga, theta, psai

def batch_diag(l):
    #l is of shape (N,input)
    #D is of shape (N,input,input)
    D = torch.zeros(l.shape[0], l.shape[1], l.shape[1], device=l.device)
    for i in range(l.shape[0]):
        D[i,:,:] = torch.diag(l[i,:])
    return D

def get_k_th_layer_bound(lstm, k,x,eps,p, verify_planes=False, eps_idx=None, print_info = True):
    #eps could be a number or a tensor of shape (batch,1)
    
    if eps_idx is None:
        eps_idx = torch.ones(lstm.seq_len, device=x.device)
    lstm.get_y(m=k,eps=eps,x=x,p=p, eps_idx=eps_idx) #bound y_k
    print('computing bounding planes for c%d * sigmoid(yf%d)' % (k-1,k))
    lstm.get_hfc(k) #bound c_k-1 * sigmoid(yf_k) 
    
    if  verify_planes:
        if k>1:
            print('verifying bounding planes for c%d * sigmoid(yf%d)' % (k-1,k))
            x_sigmoid.validate(lstm.beta_l_fc[k-1], lstm.alpha_l_fc[k-1], lstm.gamma_l_fc[k-1],
                                lstm.beta_u_fc[k-1], lstm.alpha_u_fc[k-1], lstm.gamma_u_fc[k-1],
                                lstm.c_l[k-1-1], lstm.c_u[k-1-1], 
                                lstm.yf_l[k-1], lstm.yf_u[k-1], 
                                max_iter=1000, plot=False, eps=1e8, print_info = print_info)
    
    print('computing bounding planes for tanh(yg%d) * sigmoid(yi%d)' % (k,k))
    lstm.get_hig(k) #bound tanh(yg_k) * sigmoid(yi_k)
    
    if verify_planes:
        print('verifying bounding planes for tanh(yg%d) * sigmoid(yi%d)' % (k,k))
        tanh_sigmoid.validate(lstm.beta_l_ig[k-1],lstm.alpha_l_ig[k-1],lstm.gamma_l_ig[k-1],
                  lstm.beta_u_ig[k-1],lstm.alpha_u_ig[k-1],lstm.gamma_u_ig[k-1],
                  lstm.yg_l[k-1], lstm.yg_u[k-1], lstm.yi_l[k-1], lstm.yi_u[k-1], 
                  max_iter=1000, eps=1e8, verify_and_modify_all = False, plot=False, print_info = print_info)
    
    lstm.get_c(v=k,eps=eps,x=x,p=p, eps_idx=eps_idx) #bound c_k
    print('computing bounding planes for tanh(c%d) * sigmoid(yo%d)' % (k,k))
    lstm.get_hoc(k) #bound tanh(c_k) * sigmoid(yo_k)
    
    if verify_planes:
        print('verifying bounding planes for tanh(c%d) * sigmoid(yo%d)' % (k,k))
        tanh_sigmoid.validate(lstm.beta_l_oc[k-1],lstm.alpha_l_oc[k-1],lstm.gamma_l_oc[k-1],
                  lstm.beta_u_oc[k-1],lstm.alpha_u_oc[k-1],lstm.gamma_u_oc[k-1],
                  lstm.c_l[k-1], lstm.c_u[k-1], lstm.yo_l[k-1], lstm.yo_u[k-1], 
                  max_iter=1000, plot=False, eps=1e8, verify_and_modify_all = False, print_info = print_info)
    
    W = torch.eye(lstm.hidden_size, device=lstm.Wia.device)
    b = 0
    lstm.get_Wa_b(W,b,k,x,eps,p,save_a = True, eps_idx=eps_idx) #bound a_k
    return 0

def get_last_layer_bound(W,b,lstm,x,eps,p, verify_bound=False, reset=False, eps_idx=None):
    #eps could be a number or a tensor of shape (batch,1)
    #bound W a_seq_len + b
    #eps idx could None or a tensor of size (seq_len)
    #it contains 0 or 1, which indicate whether to perturb the corresponding frame
    if type(eps) == torch.Tensor:
        if len(eps.shape) == 1:
            eps_new = eps.unsqueeze(1).detach()
        else:
            eps_new = eps.detach()
    else:
        eps_new = eps
        
    seq_len = lstm.seq_len
    if eps_idx is None:
        eps_idx = torch.ones(seq_len, device=x.device)
    for k in range(1, seq_len+1):
        get_k_th_layer_bound(lstm, k,x,eps_new,p,verify_planes=True,eps_idx=eps_idx, print_info = lstm.print_info)
        if verify_bound:
            lstm_utils.verify_bound(lstm,k,p,eps_new,x, max_iter=20,
                                verify_y=True, verify_c=True,
                                verify_a = True, thred=1e8, 
                                eps_idx=eps_idx, print_info = lstm.print_info)
        
    minimum, maximum = lstm.get_Wa_b(W,b,seq_len,x,eps_new,p,save_a = False,
                                     eps_idx=eps_idx)
    # lstm_utils.verify_final_output(net, x, minimum, maximum,p,eps,
    #                     max_iter = 1000, thred=1e-4)
    lstm_utils.verify_final_output2(lstm,W,b, x, minimum, maximum,p,eps,
                        max_iter = 20, thred=1e8, eps_idx=eps_idx, print_info = lstm.print_info)
    print('-' * 60)
    if reset:
        lstm.init_h()
        lstm.init_yc()
    return minimum, maximum


def cut_lstm(lstm, x, eps_idx):
    #user must maually save lstm.a0 and lstm.c0 if you want to use it latter
    #cause this function will overwrite them
    a,c,_,_,_,_ = lstm(x)
    seq_len = x.shape[1]
    for i in range(seq_len):
        if eps_idx[i] == 1:
            print('The first nonzero element of eps_idx is eps_idx[%d]' % i)
            break
    if i == 0:
        if lstm.a0 is None:
            lstm.a0 = torch.zeros(a[:,i,:].shape, device=a.device)
        if lstm.c0 is None:
            lstm.c0 = torch.zeros(c[:,i,:].shape, device=c.device)
        return lstm, x, eps_idx
    else:
        a0 = a[:,i-1,:]
        c0 = c[:,i-1,:]
        lstm.a0 = a0
        lstm.c0 = c0
        x_new = x[:,i:,:]
        seq_len_new = seq_len - i
        lstm.reset_seq_len(seq_len_new)
        eps_idx_new = eps_idx[i:]
        return lstm, x_new, eps_idx_new
    return 0
    


    
    
    
    