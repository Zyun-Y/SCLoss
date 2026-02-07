
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
import torch.nn as nn


Directable={'upper_left':[-1,-1],'up':[0,-1],'upper_right':[1,-1],'left':[-1,0],'right':[1,0],'lower_left':[-1,1],'down':[0,1],'lower_right':[1,1]}
TL_table = ['lower_right','down','lower_left','right','left','upper_right','up','upper_left']


def get_zero_one_mask(mask, K=1):
    # print(mask.shape)
    [batch,channels,rows, cols] = mask.shape

    inverse_mask = 1-mask.clone()


    inverse_static_mask = inverse_mask.clone().repeat(1,8,1, 1)

    inverse_direction_mask = torch.ones([batch,8,rows, cols]).cuda()

    inverse_direction_mask[:,1,:rows-K, :] = inverse_mask[:,0,K:rows,:].clone()
    inverse_direction_mask[:,6,K:rows,:] = inverse_mask[:,0,0:rows-K,:].clone()
    inverse_direction_mask[:,3,:,:cols-K] = inverse_mask[:,0,:,K:cols].clone()
    inverse_direction_mask[:,4,:,K:cols] =  inverse_mask[:,0,:,:cols-K].clone()
    inverse_direction_mask[:,0,0:rows-K,0:cols-K] = inverse_mask[:,0,K:rows,K:cols].clone()
    inverse_direction_mask[:,2,0:rows-K,K:cols]= inverse_mask[:,0,K:rows,0:cols-K].clone()
    inverse_direction_mask[:,5,K:rows,0:cols-K] = inverse_mask[:,0,0:rows-K,K:cols].clone()
    inverse_direction_mask[:,7,K:rows,K:cols] = inverse_mask[:,0,0:rows-K,0:cols-K].clone()

    # inverse_direction_mask: shifted version of flipped mask. from channel 0 -7: shift to upper right, up,...
    ## inverse_mask_conn: the conn mask of the inverse mask (flipped)
    ## 1-inverse_mask_conn: pixels where not all 0s (1-1 or 1-0)
    inverse_mask_conn = inverse_static_mask*inverse_direction_mask

    

    zero_one_mask = inverse_static_mask*(1-inverse_mask_conn)

    ## zero_one_mask: each 1 means current value is 0 but has an 1 neighbor at that direction

    return zero_one_mask


def conn_mask_gen(mask, K=1):
    [batch,_,rows, cols] = mask.shape
    conn = torch.zeros([batch,8,rows, cols]).cuda()
    up = torch.zeros([batch,rows, cols]).cuda()#move the orignal mask to up
    down = torch.zeros([batch,rows, cols]).cuda()
    left = torch.zeros([batch,rows, cols]).cuda()
    right = torch.zeros([batch,rows, cols]).cuda()
    up_left = torch.zeros([batch,rows, cols]).cuda()
    up_right = torch.zeros([batch,rows, cols]).cuda()
    down_left = torch.zeros([batch,rows, cols]).cuda()
    down_right = torch.zeros([batch,rows, cols]).cuda()


    up[:,:rows-K, :] = mask[:,0,K:rows,:]
    down[:,K:rows,:] = mask[:,0,0:rows-K,:]
    left[:,:,:cols-K] = mask[:,0,:,K:cols]
    right[:,:,K:cols] = mask[:,0,:,:cols-K]
    up_left[:,0:rows-K,0:cols-K] = mask[:,0,K:rows,K:cols]
    up_right[:,0:rows-K,K:cols] = mask[:,0,K:rows,0:cols-K]
    down_left[:,K:rows,0:cols-K] = mask[:,0,0:rows-K,K:cols]
    down_right[:,K:rows,K:cols] = mask[:,0,0:rows-K,0:cols-K]

    conn[:,0] = mask[:,0]*down_right
    conn[:,1] = mask[:,0]*down
    conn[:,2] = mask[:,0]*down_left
    conn[:,3] = mask[:,0]*right
    conn[:,4] = mask[:,0]*left
    conn[:,5] = mask[:,0]*up_right
    conn[:,6] = mask[:,0]*up
    conn[:,7] = mask[:,0]*up_left
    conn = conn.float()
    return conn

class SCloss(nn.Module):
    def __init__(self):
        super(SCloss, self).__init__()
        self.BCEloss = nn.BCELoss(reduction = 'none')

    def negative_loss(self, c_map, target, K, T=10):
        negative_mask = 1-target
        negative_mask = conn_mask_gen(negative_mask, K)
        p1 = torch.zeros(c_map.shape[0],8, c_map.shape[2], c_map.shape[3]).cuda()
        p2 = torch.zeros(c_map.shape[0],8, c_map.shape[2], c_map.shape[3]).cuda()
        
        for i in range(8):
            p1[:,i] = negative_mask[:,i].clone()*c_map[:,0]
            # p2[:,i] = self.shift_diag(negative_mask[:,7-i].clone()*c_map[:,0],Directable[TL_table[i]])
            p2[:,i] = self.shift_diag(negative_mask[:,7-i].clone()*c_map[:,0],Directable[TL_table[i]])
            for j in range(K-1):
                p2[:,i] = self.shift_diag(p2[:,i],Directable[TL_table[i]])

        term1 = negative_mask*(-torch.log(1-p2+0.0001)-torch.log(1-p1+0.0001)) #### 0.00001 defines how large if there is a zero prediction in 1 area
        term2 = negative_mask*(-torch.log(1-p1*p2+0.0001)+torch.exp(-p1*p2))

        negative_loss = term1/(term2+0.00001)
        negative_loss = (negative_mask*negative_loss)
        return negative_loss

    def positive_loss(self, c_map, target, T =10, K = 1):
        p1 = torch.zeros(c_map.shape[0],8, c_map.shape[2], c_map.shape[3]).cuda()
        p2 = torch.zeros(c_map.shape[0],8, c_map.shape[2], c_map.shape[3]).cuda()
        conn_mask = conn_mask_gen(target, K)

        for i in range(8):
            p1[:,i] = conn_mask[:,i].clone()*c_map[:,0]
            p2[:,i] = self.shift_diag(conn_mask[:,7-i].clone()*c_map[:,0],Directable[TL_table[i]])
            for j in range(K-1):
                p2[:,i] = self.shift_diag(p2[:,i],Directable[TL_table[i]])



        term1 = conn_mask*(-torch.log(p2+0.0001)-torch.log(p1+0.0001)) #### 0.00001 defines how large if there is a zero prediction in 1 area
        term2 = conn_mask*(-torch.log(p1*p2+0.0001)+torch.exp(-p1*p2))

        postive_loss = term1/(term2+0.00001)
        postive_loss = (conn_mask*postive_loss)

        return postive_loss

    def neighbor_loss(self, c_map, target, T1 = 0.8, T2 = 0.2, K=1):

        zero_one_mask = get_zero_one_mask(target, K)
        # imsave('zero_one_mask.png',zero_one_mask[0,1].cpu().data.numpy())
        p2 = zero_one_mask.clone()*c_map
        p1_mask = torch.zeros(c_map.shape[0],8, c_map.shape[2], c_map.shape[3]).cuda()
        

        for i in range(8):
            p2[:,i] = self.shift_diag(p2[:,i].clone(),Directable[TL_table[i]])
            p1_mask[:,i] = self.shift_diag(zero_one_mask[:,i].clone(),Directable[TL_table[i]])

            for j in range(K-1):
                p2[:,i] = self.shift_diag(p2[:,i].clone(),Directable[TL_table[i]])
                p1_mask[:,i] = self.shift_diag(p1_mask[:,i].clone(),Directable[TL_table[i]])

        p1 = p1_mask*c_map

        term1 = (-T1*p1_mask*torch.log(1-p2+0.0001) - p1_mask*T2*torch.log(p1+0.00001)) 
        term2 = (-p1_mask*torch.log(1-p1*p2+0.0001)+p1_mask*torch.exp(-p2))



        neighbor_CELoss = term1/(term2+0.00001)
        neighbor_CELoss = (p1_mask*neighbor_CELoss).mean()
        return neighbor_CELoss

    def forward(self, c_map, target):
        hori_translation = torch.zeros([c_map.shape[1],c_map.shape[3],c_map.shape[3]])
        for i in range(c_map.shape[3]-1):
            hori_translation[:,i,i+1] = torch.tensor(1.0)
        verti_translation = torch.zeros([c_map.shape[1],c_map.shape[2],c_map.shape[2]])
        for j in range(c_map.shape[2]-1):
            verti_translation[:,j,j+1] = torch.tensor(1.0)

        hori_translation = hori_translation.float()
        verti_translation = verti_translation.float()

        target = target.type(torch.FloatTensor).cuda()

        self.hori_translation = hori_translation.repeat(c_map.shape[0],1,1,1).cuda()
        self.verti_translation = verti_translation.repeat(c_map.shape[0],1,1,1).cuda()

        c_map = F.sigmoid(c_map)
        
        nb_loss_k1 = self.neighbor_loss(c_map, target, T1 = 0.8, T2 = 0.2 , K=1)
        nb_loss_k2 = self.neighbor_loss(c_map, target, T1 = 0.8, T2 = 0.2 , K=2)

        n_loss_k1 = self.negative_loss(c_map, target, K=1, T=0.5)
        n_loss_k2 = self.negative_loss(c_map, target, K=2, T=0.5)
        p_loss_k1 = self.positive_loss(c_map, target, T = 0.5, K = 1)
        p_loss_k2 = self.positive_loss(c_map, target, T = 0.5, K = 2)

        loss = 0.7*(n_loss_k1 + 0.5*n_loss_k2) + 0.3*(p_loss_k1 + 0.5* p_loss_k2) + 0.7*(0.5*nb_loss_k1 + 0.5*nb_loss_k2)#+ p_loss
        loss = loss.mean() 

        return loss

    def shift_diag(self,img,shift):
        batch, row, column = img.size()

        if shift[0]: ###horizontal
            img = torch.bmm(img.view(-1,row,column),self.hori_translation.view(-1,column,column)) if shift[0]==1 else torch.bmm(img.view(-1,row,column),self.hori_translation.transpose(3,2).view(-1,column,column))
        if shift[1]: ###vertical
            img = torch.bmm(self.verti_translation.transpose(3,2).view(-1,row,row),img.view(-1,row,column)) if shift[1]==1 else torch.bmm(self.verti_translation.view(-1,row,row),img.view(-1,row,column))
        return img.view(batch, row, column)