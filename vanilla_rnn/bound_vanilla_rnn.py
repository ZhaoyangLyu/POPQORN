#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 08:36:29 2018

@author: root

"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.nn.parameter import Parameter
from torchvision import datasets
import numpy as np

import get_bound_for_general_activation_function as get_bound
from utils.sample_data import sample_mnist_data
from utils.verify_bound import verify_final_output, verifyMaximumEps

import os
import argparse

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, time_step, activation):
        # the num_layers here is the number of RNN units
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size = input_size,hidden_size=hidden_size,
                          num_layers = 1,batch_first=True, nonlinearity = activation)
        self.out = nn.Linear(hidden_size , output_size )
        self.X = None #data attached to the classifier, and is of size (N,self.time_step,self.input_size)
        self.l = [None]*(time_step+1) #l[0] has no use, l[k] is of size (N,n_k), k from 1 to m
        # l[k] is the pre-activation lower bound of the k-th layer  
        self.u = [None]*(time_step+1) #u[0] has no use, u[k] is of size (N,n_k), k from 1 to m
        #u[k] is the pre-activation upper bound of the k-th layer 
        self.kl = [None]*(time_step+1)
        self.ku = [None]*(time_step+1)
        
        self.bl = [None]*(time_step+1)
        self.bu = [None]*(time_step+1)
        
        self.time_step = time_step #number of vanilla layers 
        self.num_neurons = hidden_size #numbers of neurons in each layer, a scalar
        self.input_size = input_size 
        self.output_size = output_size
        
        self.activation = activation
        if activation == 'tanh':
            self.activation_function = nn.Tanh()
        elif activation == 'relu':
            self.activation_function = nn.ReLU()
        else:
            raise Exception(activation+' activation function is not supported')

        self.W_fa = None  # [output_size, num_neurons] 
        self.W_aa = None  # [num_neurons, num_neurons]
        self.W_ax = None  # [num_neurons, input_size]
        self.b_f = None   # [output_size]
        self.b_ax = None  # [num_neurons]
        self.b_aa = None  # [num_neurons]
        self.a_0 = None   # initial hidden state     
        
    def forward(self,X):
        # X is of size (batch, seq_len, input_size)
        r_out, h_n = self.rnn(X, self.a_0) # RNN usage: output, hn = rnn(input,h0)
        # r_out is of size (batch, seq_len, hidden_size)
        # h_n is of size (batch, num_layers, hidden_size)
        out = self.out(r_out[:,-1,:])
        return out      
    
    def clear_intermediate_variables(self):
        time_step = self.time_step
        self.l = [None]*(time_step+1) 
        self.u = [None]*(time_step+1) 
        self.kl = [None]*(time_step+1)
        self.ku = [None]*(time_step+1)
        self.bl = [None]*(time_step+1)
        self.bu = [None]*(time_step+1)
        
    def reset_seq_len(self, seq_len):
        self.time_step = seq_len
        self.clear_intermediate_variables()
    
    def get_preactivation_output(self, X):
        #k range from 1 to self.num_layers
        #get the pre-relu output of the k-th layer
        #X is of shape (N, seq_len, in_features)
        with torch.no_grad():
            device = X.device
            N = X.shape[0]
            seq_len = X.shape[1]
            h = torch.zeros([N, seq_len+1, hidden_size], device = device)
            pre_h = torch.zeros([N, seq_len+1, hidden_size], device = device)
            b_ax = self.b_ax.unsqueeze(0).expand(N,-1)
            b_aa = self.b_aa.unsqueeze(0).expand(N,-1)
            for i in range(seq_len):
                pre_h[:,i+1,:] = (torch.matmul(self.W_ax, X[:,i,:].unsqueeze(2)).squeeze(2)
                    + b_ax +
                    torch.matmul(self.W_aa, h[:,i,:].unsqueeze(2)).squeeze(2) + b_aa)
                h[:,i+1,:] = self.activation_function(pre_h[:,i+1,:])
        return pre_h[:,1:,:], h[:,1:,:]
                
    def attachData(self,X):
        # X is of size N*C*H*W
        if torch.numel(X) == (X.shape[0] * self.input_dimension):
        #if X.shape[1]*X.shape[2]*X.shape[3] == self.input_dimension:
            X = X.view(-1, self.input_dimension)
            self.X = X
        else:
            raise Exception('The input dimension must be %d' % self.input_dimension)
    
    def extractWeight(self, clear_original_model=True):
        with torch.no_grad():
            self.W_fa = self.out.weight  # [output_size, num_neurons] 
            self.W_aa = self.rnn.weight_hh_l0  # [num_neurons, num_neurons]
            self.W_ax = self.rnn.weight_ih_l0  # [num_neurons, input_size]
            self.b_f = self.out.bias   # [output_size]
            self.b_ax = self.rnn.bias_ih_l0  # [num_neurons]
            self.b_aa = self.rnn.bias_hh_l0  # [num_neurons]
            # a_t = tanh(W_ax x_(t-1) + b_ax + W_aa a_(t-1) + b_aa)
            if clear_original_model:
                self.rnn = None
        return 0
        
    def compute2sideBound(self, eps, p, v, X = None, Eps_idx = None):
        # X here is of size [batch_size, layer_index m, input_size]
        #eps could be a real number, or a tensor of size N
        #p is a real number
        #m is an integer
        #print(self.W[m].shape[0])
        with torch.no_grad():
            n = self.W_ax.shape[1]  # input_size
            s = self.W_ax.shape[0]  # hidden_size     
            idx_eps = torch.zeros(self.time_step, device=X.device)
            idx_eps[Eps_idx-1] = 1
            if X is None:
                X = self.X
            N = X.shape[0]  # number of images, batch size
            if self.a_0 is None:
                a_0 = torch.zeros(N, s, device = X.device)
            else:
                a_0 = self.a_0
            if type(eps) == torch.Tensor:
                eps = eps.to(X.device)        
            if p == 1:
                q = float('inf')
            elif p == 'inf' or p==float('inf'):
                q = 1 
            else:
                q = p / (p-1)
            
            yU = torch.zeros(N, s, device = X.device)  # [N,s]
            yL = torch.zeros(N, s, device = X.device)  # [N,s]
            
            W_ax = self.W_ax.unsqueeze(0).expand(N,-1,-1)  # [N, s, n]
            W_aa = self.W_aa.unsqueeze(0).expand(N,-1,-1)  # [N, s, s]    
            b_ax = self.b_ax.unsqueeze(0).expand(N,-1)  # [N, s]
            b_aa = self.b_aa.unsqueeze(0).expand(N,-1)
            
            # v-th terms, three terms        
            ## first term
            if type(eps) == torch.Tensor:                      
                #eps is a tensor of size N 
                yU = yU + idx_eps[v-1]*eps.unsqueeze(1).expand(-1,
                                       s)*torch.norm(W_ax,p=q,dim=2)  # eps ||A^ {<v>} W_ax||q    
                yL = yL - idx_eps[v-1]*eps.unsqueeze(1).expand(-1,
                                       s)*torch.norm(W_ax,p=q,dim=2)  # eps ||Ou^ {<v>} W_ax||q      
            else:
                yU = yU + idx_eps[v-1]*eps*torch.norm(W_ax,p=q,dim=2)  # eps ||A^ {<v>} W_ax||q    
                yL = yL - idx_eps[v-1]*eps*torch.norm(W_ax,p=q,dim=2)  # eps ||Ou^ {<v>} W_ax||q  
            ## second term
            if v == 1:
                X = X.view(N,1,n)
            yU = yU + torch.matmul(W_ax,X[:,v-1,:].view(N,n,1)).squeeze(2)  # A^ {<v>} W_ax x^{<v>}            
            yL = yL + torch.matmul(W_ax,X[:,v-1,:].view(N,n,1)).squeeze(2)  # Ou^ {<v>} W_ax x^{<v>}       
            ## third term
            yU = yU + b_aa+ b_ax  # A^ {<v>} (b_a + Delta^{<v>})
            yL = yL + b_aa+ b_ax  # Ou^ {<v>} (b_a + Theta^{<v>})
                            
            if not (v == 1):
                # k from v-1 to 1 terms
                for k in range(v-1,0,-1): 
                    if k == v-1:
                        ## compute A^{<v-1>}, Ou^{<v-1>}, Delta^{<v-1>} and Theta^{<v-1>}
                        ### 1. compute slopes alpha and intercepts beta
                        kl, bl, ku, bu = get_bound.getConvenientGeneralActivationBound(
                                self.l[k], self.u[k], self.activation)
                        
                        bl = bl/kl
                        bu = bu/ku
                        
                        self.kl[k] = kl  # [N, s]
                        self.ku[k] = ku  # [N, s]
                        self.bl[k] = bl  # [N, s]
                        self.bu[k] = bu  # [N, s]
                        alpha_l = kl.unsqueeze(1).expand(-1, s, -1)  # [N, s, s]
                        alpha_u = ku.unsqueeze(1).expand(-1, s, -1)  # [N, s, s]
                        beta_l = bl.unsqueeze(1).expand(-1, s, -1)  # [N, s, s]
                        beta_u = bu.unsqueeze(1).expand(-1, s, -1)  # [N, s, s]
                        ### 2. compute lambda^{<v-1>}, omega^{<v-1>}, Delta^{<v-1>} and Theta^{<v-1>}
                        I = (W_aa >= 0).float()  # [N, s, s]
                        lamida = I*alpha_u + (1-I)*alpha_l                  
                        omiga = I*alpha_l + (1-I)*alpha_u
                        Delta = I*beta_u + (1-I)*beta_l  # [N, s, s], this is the transpose of the delta defined in the paper
                        Theta = I*beta_l + (1-I)*beta_u  # [N, s, s]
                        ### 3. clear l[k] and u[k] to release memory
                        self.l[k] = None
                        self.u[k] = None
                        ### 4. compute A^{<v-1>} and Ou^{<v-1>}
                        A = W_aa * lamida  # [N, s, s]
                        Ou = W_aa * omiga  # [N, s, s]
                    else:
                        ## compute A^{<k>}, Ou^{<k>}, Delta^{<k>} and Theta^{<k>}
                        ### 1. compute slopes alpha and intercepts beta
                        alpha_l = self.kl[k].unsqueeze(1).expand(-1, s, -1)
                        alpha_u = self.ku[k].unsqueeze(1).expand(-1, s, -1)
                        beta_l = self.bl[k].unsqueeze(1).expand(-1, s, -1)  # [N, s, s]
                        beta_u = self.bu[k].unsqueeze(1).expand(-1, s, -1)  # [N, s, s]
                        ### 2. compute lambda^{<k>}, omega^{<k>}, Delta^{<k>} and Theta^{<k>}
                        I = (torch.matmul(A,W_aa) >= 0).float()  # [N, s, s]
                        lamida = I*alpha_u + (1-I)*alpha_l                  
                        Delta = I*beta_u + (1-I)*beta_l  # [N, s, s], this is the transpose of the delta defined in the paper
                        I = (torch.matmul(Ou,W_aa) >= 0).float()  # [N, s, s]
                        omiga = I*alpha_l + (1-I)*alpha_u
                        Theta = I*beta_l + (1-I)*beta_u  # [N, s, s]
                        ### 3. compute A^{<k>} and Ou^{<k>}
                        A = torch.matmul(A,W_aa) * lamida  # [N, s, s]
                        Ou = torch.matmul(Ou,W_aa) * omiga  # [N, s, s]
                    ## first term
                    if type(eps) == torch.Tensor:                
                        #eps is a tensor of size N 
                        yU = yU + idx_eps[k-1]*eps.unsqueeze(1).expand(-1,
                                   s)*torch.norm(torch.matmul(A,W_ax),p=q,dim=2)  # eps ||A^ {<k>} W_ax||q    
                        yL = yL - idx_eps[k-1]*eps.unsqueeze(1).expand(-1,
                                   s)*torch.norm(torch.matmul(Ou,W_ax),p=q,dim=2)  # eps ||Ou^ {<k>} W_ax||q      
                    else:
                        yU = yU + idx_eps[k-1]*eps*torch.norm(torch.matmul(A,W_ax),p=q,dim=2)  # eps ||A^ {<k>} W_ax||q    
                        yL = yL - idx_eps[k-1]*eps*torch.norm(torch.matmul(Ou,W_ax),p=q,dim=2)  # eps ||Ou^ {<k>} W_ax||q  
                    ## second term
                    yU = yU + torch.matmul(A,torch.matmul(W_ax,X[:,k-1,:].view(N,n,1))).squeeze(2)  # A^ {<k>} W_ax x^{<k>}            
                    yL = yL + torch.matmul(Ou,torch.matmul(W_ax,X[:,k-1,:].view(N,n,1))).squeeze(2)  # Ou^ {<k>} W_ax x^{<k>}       
                    ## third term
                    yU = yU + torch.matmul(A,(b_aa+b_ax).view(N,s,1)).squeeze(2)+(A*Delta).sum(2)  # A^ {<k>} (b_a + Delta^{<k>})
                    yL = yL + torch.matmul(Ou,(b_aa+b_ax).view(N,s,1)).squeeze(2)+(Ou*Theta).sum(2)  # Ou^ {<k>} (b_a + Theta^{<k>})
                # compute A^{<0>}
                A = torch.matmul(A,W_aa)  # (A^ {<1>} W_aa) * lambda^{<0>}
                Ou = torch.matmul(Ou,W_aa)  # (Ou^ {<1>} W_aa) * omega^{<0>}
            else:
                A = W_aa  # A^ {<0>}, [N, s, s]
                Ou = W_aa  # Ou^ {<0>}, [N, s, s]
            yU = yU + torch.matmul(A,a_0.view(N,s,1)).squeeze(2)  # A^ {<0>} * a_0
            yL = yL + torch.matmul(Ou,a_0.view(N,s,1)).squeeze(2)  # Ou^ {<0>} * a_0
                        
            self.l[v] = yL
            self.u[v] = yU
        return yL,yU
    
    def computeLast2sideBound(self, eps, p, v, X = None, Eps_idx = None):
        with torch.no_grad():
            n = self.W_ax.shape[1]  # input_size
            s = self.W_ax.shape[0]  # hidden_size     
            t = self.W_fa.shape[0]  # output_size
            idx_eps = torch.zeros(self.time_step, device=X.device)
            idx_eps[Eps_idx-1] = 1
            if X is None:
                X = self.X
            N = X.shape[0]  # number of images, batch size
            if self.a_0 is None:
                a_0 = torch.zeros(N, s, device = X.device)
            else:
                a_0 = self.a_0
            if type(eps) == torch.Tensor:
                eps = eps.to(X.device)        
            if p == 1:
                q = float('inf')
            elif p == 'inf':
                q = 1
            else:
                q = p / (p-1)
            
            yU = torch.zeros(N, t, device = X.device)  # [N,s]
            yL = torch.zeros(N, t, device = X.device)  # [N,s]
            
            W_ax = self.W_ax.unsqueeze(0).expand(N,-1,-1)  # [N, s, n]
            W_aa = self.W_aa.unsqueeze(0).expand(N,-1,-1)  # [N, s, s]   
            W_fa = self.W_fa.unsqueeze(0).expand(N,-1,-1)  # [N, t, s]  
            b_ax = self.b_ax.unsqueeze(0).expand(N,-1)  # [N, s]
            b_aa = self.b_aa.unsqueeze(0).expand(N,-1)  # [N, s]
            b_f = self.b_f.unsqueeze(0).expand(N,-1)  # [N, t]
                            
            # k from time_step+1 to 1 terms
            for k in range(v-1,0,-1): 
                if k == v-1:
                    ## compute A^{<v-1>}, Ou^{<v-1>}, Delta^{<v-1>} and Theta^{<v-1>}
                    ### 1. compute slopes alpha and intercepts beta
                    kl, bl, ku, bu = get_bound.getConvenientGeneralActivationBound(
                                self.l[k], self.u[k], self.activation)
                    
                    bl = bl/kl
                    bu = bu/ku
                    
                    self.kl[k] = kl  # [N, s]
                    self.ku[k] = ku  # [N, s]
                    self.bl[k] = bl  # [N, s]
                    self.bu[k] = bu  # [N, s]
                    alpha_l = kl.unsqueeze(1).expand(-1, t, -1)  # [N, t, s]
                    alpha_u = ku.unsqueeze(1).expand(-1, t, -1)  # [N, t, s]
                    beta_l = bl.unsqueeze(1).expand(-1, t, -1)  # [N, t, s]
                    beta_u = bu.unsqueeze(1).expand(-1, t, -1)  # [N, t, s]
                    ### 2. compute lambda^{<v-1>}, omega^{<v-1>}, Delta^{<v-1>} and Theta^{<v-1>}
                    I = (W_fa >= 0).float()  # [N, t, s]
                    lamida = I*alpha_u + (1-I)*alpha_l                  
                    omiga = I*alpha_l + (1-I)*alpha_u
                    Delta = I*beta_u + (1-I)*beta_l  # [N, t, s], this is the transpose of the delta defined in the paper
                    Theta = I*beta_l + (1-I)*beta_u  # [N, t, s]
                    ### 3. clear l[k] and u[k] to release memory
                    self.l[k] = None
                    self.u[k] = None
                    ### 4. compute A^{<v-1>} and Ou^{<v-1>}
                    A = W_fa * lamida  # [N, t, s]
                    Ou = W_fa * omiga  # [N, t, s]
                else:
                    ## compute A^{<k>}, Ou^{<k>}, Delta^{<k>} and Theta^{<k>}
                    ### 1. compute slopes alpha and intercepts beta
                    alpha_l = self.kl[k].unsqueeze(1).expand(-1, t, -1)
                    alpha_u = self.ku[k].unsqueeze(1).expand(-1, t, -1)
                    beta_l = self.bl[k].unsqueeze(1).expand(-1, t, -1)  # [N, t, s]
                    beta_u = self.bu[k].unsqueeze(1).expand(-1, t, -1)  # [N, t, s]
                    ### 2. compute lambda^{<k>}, omega^{<k>}, Delta^{<k>} and Theta^{<k>}
                    I = (torch.matmul(A,W_aa) >= 0).float()  # [N, t, s]
                    lamida = I*alpha_u + (1-I)*alpha_l                  
                    Delta = I*beta_u + (1-I)*beta_l  # [N, s, s], this is the transpose of the delta defined in the paper
                    I = (torch.matmul(Ou,W_aa) >= 0).float()  # [N, t, s]
                    omiga = I*alpha_l + (1-I)*alpha_u
                    Theta = I*beta_l + (1-I)*beta_u  # [N, s, s]
                    ### 3. compute A^{<k>} and Ou^{<k>}
                    A = torch.matmul(A,W_aa) * lamida  # [N, s, s]
                    Ou = torch.matmul(Ou,W_aa) * omiga  # [N, s, s]
                ## first term
                if type(eps) == torch.Tensor:                
                    #eps is a tensor of size N 
                    yU = yU + idx_eps[k-1]*eps.unsqueeze(1).expand(-1,
                               t)*torch.norm(torch.matmul(A,W_ax),p=q,dim=2)  # eps ||A^ {<k>} W_ax||q    
                    yL = yL - idx_eps[k-1]*eps.unsqueeze(1).expand(-1,
                               t)*torch.norm(torch.matmul(Ou,W_ax),p=q,dim=2)  # eps ||Ou^ {<k>} W_ax||q      
                else:
                    yU = yU + idx_eps[k-1]*eps*torch.norm(torch.matmul(A,W_ax),p=q,dim=2)  # eps ||A^ {<k>} W_ax||q    
                    yL = yL - idx_eps[k-1]*eps*torch.norm(torch.matmul(Ou,W_ax),p=q,dim=2)  # eps ||Ou^ {<k>} W_ax||q  
                ## second term
                yU = yU + torch.matmul(A,torch.matmul(W_ax,X[:,k-1,:].view(N,n,1))).squeeze(2)  # A^ {<k>} W_ax x^{<k>}            
                yL = yL + torch.matmul(Ou,torch.matmul(W_ax,X[:,k-1,:].view(N,n,1))).squeeze(2)  # Ou^ {<k>} W_ax x^{<k>}       
                ## third term
                yU = yU + torch.matmul(A,(b_aa+b_ax).view(N,s,1)).squeeze(2)+(A*Delta).sum(2)  # A^ {<k>} (b_a + Delta^{<k>})
                yL = yL + torch.matmul(Ou,(b_aa+b_ax).view(N,s,1)).squeeze(2)+(Ou*Theta).sum(2)  # Ou^ {<k>} (b_a + Theta^{<k>})
            # compute A^{<0>}
            A = torch.matmul(A,W_aa)  # (A^ {<1>} W_aa) * lambda^{<0>}
            Ou = torch.matmul(Ou,W_aa)  # (Ou^ {<1>} W_aa) * omega^{<0>}
            yU = yU + torch.matmul(A,a_0.view(N,s,1)).squeeze(2)  # A^ {<0>} * a_0
            yL = yL + torch.matmul(Ou,a_0.view(N,s,1)).squeeze(2)  # Ou^ {<0>} * a_0
            yU = yU + b_f
            yL = yL + b_f
        return yL,yU
    
    def getLastLayerBound(self, eps, p, X = None, clearIntermediateVariables=False, Eps_idx = None):
        #eps could be a real number, or a tensor of size N
        with torch.no_grad():
            if self.X is None and X is None:
                raise Exception('You must first attach data to the net or feed data to this function')
            if self.W_fa is None or self.W_aa is None or self.W_ax is None or self.b_f is None or self.b_ax is None or self.b_aa is None:
                self.extractWeight()
            if X is None:
                X = self.X
            if Eps_idx is None:
                Eps_idx = torch.arange(1,self.time_step+1)
            for k in range(1,self.time_step+1):
                # k from 1 to self.time_step
                yL,yU = self.compute2sideBound(eps, p, k, X=X[:,0:k,:], Eps_idx = Eps_idx)
            yL,yU = self.computeLast2sideBound(eps, p, self.time_step+1, X, Eps_idx = Eps_idx)    
                #in this loop, self.u, l, Il, WD are reused
            if clearIntermediateVariables:
                self.clear_intermediate_variables()
        return yL, yU
        
    def getMaximumEps(self, p, true_label, target_label, eps0 = 1, max_iter = 100, 
                      X = None, acc = 0.001, gx0_trick = True, Eps_idx = None):
        #when u_eps-l_eps < acc, we stop searching
        with torch.no_grad():
            if self.X is None and X is None:
                raise Exception('You must first attach data to the net or feed data to this function')
            if X is None:
                X = self.X
            N = X.shape[0]
            #        time_step=X.shape[1]
            if Eps_idx is None:
                Eps_idx = torch.tensor(range(1,self.time_step+1))#, device = X.device)
                # print('Eps_idx.device', Eps_idx.device)
            if max(Eps_idx) > self.time_step:
                raise Exception('The perturbed frame index should not exceed the number of time step')
            idx=torch.arange(N)
            l_eps = torch.zeros(N, device = X.device) #lower bound of eps
            u_eps = torch.ones(N, device = X.device) * eps0 #upper bound of eps
            
            # use gx0_trick: the "equivalent output node" is equal to N (number of samples in one batch)
            if gx0_trick == True:
                 print("W_fa size = {}".format(self.W_fa.shape))
                 print("b_f size = {}".format(self.b_f.shape))             
                 self.W_fa = Parameter(self.W_fa[true_label,:]-self.W_fa[target_label,:])
                 self.b_f = Parameter(self.b_f[true_label]-self.b_f[target_label])
                 print("after gx0_trick W_fa size = {}".format(self.W_fa.shape))
                 print("after gx0_trick b_f size = {}".format(self.b_f.shape))
            yL, yU = self.getLastLayerBound(u_eps, p, X = X,  
                                             clearIntermediateVariables=True, Eps_idx = Eps_idx)
            if gx0_trick: 
                 lower = yL[idx,idx]
                 increase_u_eps = lower > 0 
                 print("initial batch f_c^L - f_j^U = {}".format(lower))
                 print("increase_u_eps = {}".format(increase_u_eps))
            else:
                 # in the paper, f_c^L = true_lower, f_j^U = target_upper 
                 true_lower = yL[idx,true_label]
                 target_upper = yU[idx,target_label]
                     
                 #indicate whether to further increase the upper bound of eps 
                 increase_u_eps = true_lower > target_upper 
                 print("initial batch f_c^L - f_j^U = {}".format(true_lower - target_upper))
                 
            ## 1. Find initial upper and lower bound for binary search
            while (increase_u_eps.sum()>0):
                #find true and nontrivial lower bound and upper bound of eps
                num = increase_u_eps.sum()
                l_eps[increase_u_eps] = u_eps[increase_u_eps]
                u_eps[increase_u_eps ] = u_eps[increase_u_eps ] * 2
                yL, yU = self.getLastLayerBound(u_eps[increase_u_eps], p, 
                            X=X[increase_u_eps,:],clearIntermediateVariables=True, Eps_idx = Eps_idx)
                #yL and yU only for those equal to 1 in increase_u_eps
                #they are of size (num,_)
                if gx0_trick:
                     lower = yL[torch.arange(num),idx[increase_u_eps]]
                     temp = lower > 0
                     print("f_c - f_j = {}".format(lower))
                else:
                     true_lower = yL[ torch.arange(num),true_label[increase_u_eps]]
                     target_upper = yU[torch.arange(num),target_label[increase_u_eps]]
                     temp = true_lower > target_upper #size num
                     print("f_c - f_j = {}".format(true_lower- target_upper))
                increase_u_eps[increase_u_eps ] = temp
                
            print('Finished finding upper and lower bound')
            print('The upper bound we found is \n', u_eps)
            print('The lower bound we found is \n', l_eps)
            
            search = (u_eps-l_eps) > acc
            #indicate whether to further perform binary search
            
            #for i in range(max_iter):
            iteration = 0 
            while(search.sum()>0):
                #perform binary search
                
                print("search = {}".format(search))
                if iteration > max_iter:
                    print('Have reached the maximum number of iterations')
                    break
                #print(search)
                num = search.sum()
                eps = (l_eps[search]+u_eps[search])/2
                yL, yU = self.getLastLayerBound(eps, p, X=X[search,:],
                                clearIntermediateVariables=True, Eps_idx = Eps_idx)
                print("torch.arange(num) = {}".format(torch.arange(num)))
                if gx0_trick:
                     lower = yL[torch.arange(num),idx[search]]
                     temp = lower > 0
                else:
                     true_lower = yL[torch.arange(num),true_label[search]]
                     target_upper = yU[torch.arange(num),target_label[search]]
                     temp = true_lower>target_upper
                search_copy = search.data.clone()
                #            print('search ', search.device)
                #            print('temp ', temp.device)
                search[search] = temp 
                #set all active units in search to temp
                #original inactive units in search are still inactive
                
                l_eps[search] = eps[temp]
                #increase active and true_lower>target_upper units in l_eps 
                
                u_eps[search_copy-search] = eps[temp==0]
                #decrease active and true_lower<target_upper units in u_eps
                
                # search = (u_eps - l_eps) > acc #reset active units in search
                search = (u_eps - l_eps) / ((u_eps+l_eps)/2+1e-8) > acc
                print('----------------------------------------')
                if gx0_trick:
                     print('f_c - f_j = {}'.format(lower))
                else:
                     print('f_c - f_j = {}'.format(true_lower-target_upper))
                print('u_eps - l_eps = {}'.format(u_eps - l_eps))
                
                iteration = iteration + 1
        return l_eps, u_eps


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description = 'Compute Certified Bound for Vanilla RNNs')

    parser.add_argument('--hidden-size', default = 64, type = int, metavar = 'HS',
                        help = 'hidden layer size (default: 64)')
    parser.add_argument('--time-step', default = 7, type = int, metavar = 'TS',
                        help = 'number of slices to cut the 28*28 image into, it should be a factor of 28 (default: 7)')
    parser.add_argument('--activation', default = 'tanh', type = str, metavar = 'a',
                        help = 'nonlinearity used in the RNN, can be either tanh or relu (default: tanh)')
    parser.add_argument('--work-dir', default = '../models/mnist_classifier/rnn_7_64_tanh/', type = str, metavar = 'WD',
                        help = 'the directory where the pretrained model is stored and the place to save the computed result')
    parser.add_argument('--model-name', default = 'rnn', type = str, metavar = 'MN',
                        help = 'the name of the pretrained model (default: rnn)')
    
    parser.add_argument('--cuda', action='store_true',
                        help='whether to allow gpu for training')
    parser.add_argument('--cuda-idx', default = 0, type = int, metavar = 'CI',
                        help = 'the index of the gpu to use if allow gpu usage (default: 0)')

    parser.add_argument('--N', default = 100, type = int,
                        help = 'number of samples to compute bounds for (default: 100)')
    parser.add_argument('--p', default = 2, type = int,
                        help = 'p norm, if p > 100, we will deem p = infinity (default: 2)')
    parser.add_argument('--eps0', default = 0.1, type = float,
                        help = 'the start value to search for epsilon (default: 0.1)')
    args = parser.parse_args()


    allow_gpu = args.cuda
    
    if torch.cuda.is_available() and allow_gpu:
        device = torch.device('cuda:%s' % args.cuda_idx)
    else:
        device = torch.device('cpu')
    
    N = args.N  # number of samples to handle at a time.
    p = args.p  # p norm
    if p > 100:
        p = float('inf')

    eps0 = args.eps0
    input_size = int(28*28 / args.time_step)
    hidden_size = args.hidden_size
    output_size = 10
    time_step = args.time_step
    activation = args.activation
    work_dir = args.work_dir
    model_name = args.model_name
    model_file = work_dir + model_name
    save_dir = work_dir + '%s_norm_bound/' % str(p)
    
    #load model
    rnn = RNN(input_size, hidden_size, output_size, time_step, activation)
    rnn.load_state_dict(torch.load(model_file, map_location='cpu'))
    rnn.to(device)
    
    
    X,y,target_label = sample_mnist_data(N, time_step, device, num_labels=10,
                    data_dir='../data/mnist', train=False, shuffle=True, 
                    rnn=rnn, x=None, y=None)
    
        
    rnn.extractWeight(clear_original_model=False)
    
    l_eps, u_eps = rnn.getMaximumEps(p=p, true_label=y, 
                    target_label=target_label, eps0=eps0,
                      max_iter=100, X=X, 
                      acc=1e-3, gx0_trick = True, Eps_idx = None)
    verifyMaximumEps(rnn, X, l_eps, p, y, target_label, 
                        eps_idx = None, untargeted=False, thred=1e-8)

    os.makedirs(save_dir, exist_ok=True)
    torch.save({'l_eps':l_eps, 'u_eps':u_eps, 'X':X, 'true_label':y, 
                'target_label':target_label}, save_dir+'certified_bound')
    print('Have saved the complete result to' + save_dir+'certified_bound')
    print('statistics of l_eps:')
    print('(min, mean, max, std) = (%.4f, %.4f, %.4f, %.4f) ' % (l_eps.min(), l_eps.mean(), l_eps.max(), l_eps.std()))
    
        
  
