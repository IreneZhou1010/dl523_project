#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 13:13:06 2023

@author: IreneZhou
"""

import numpy as np
# import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
# import cv2
from sklearn.cluster import KMeans


import torch
import torch.nn as nn
# import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

#%% import data
root = '/Dataset1_benchmark'
# 'projectnb/dl523/students/izhou/'

def read_tiff(path):
    img = Image.open(path)
    images = []
    for i in range(img.n_frames):
        img.seek(i)
        images.append(np.array(img))
    return np.array(images)

images1 = read_tiff('./Dataset1_benchmark/Gene1.tif')
dim = images1.shape
images_allGenes = np.zeros((12,dim[0],dim[1],dim[2]))
images_allGenes[0,:,:,:] = images1

j = 1
for i in range(2,16): # no 2, 10, 14
    if i == 2 or i == 10 or i == 14:
        continue
    else:
        path = './Dataset1_benchmark/Gene' + str(i) + '.tif'
        image = read_tiff(path)
        images_allGenes[j,:,:,:] = image
        j = j+1


#%% CAE model
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 2006*2005 (batch), 15
        self.fc1 = nn.Linear(8, 6)
        self.encoder = nn.Sequential(
            nn.Linear(12, 10),
            nn.ReLU(),
            nn.Linear(10, 8),
            nn.ReLU(),
            self.fc1,     
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(6,8),
            nn.ReLU(),
            nn.Linear(8,10),
            nn.ReLU(),
            nn.Linear(10,12),
            nn.Sigmoid()
        )
        
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    
########### CAE loss
mse_loss = nn.BCELoss(size_average = False)
def loss_function(W, x, recons_x, h, lam):
    mse = mse_loss(recons_x, x)
    # Since: W is shape of N_hidden x N. So, we do not need to transpose it as
    # opposed to #1
    dh = h * (1 - h) # Hadamard product produces size N_batch x N_hidden
    # Sum through the input dimension to improve efficiency, as suggested in #1
    w_sum = torch.sum(Variable(W)**2, dim=1)
    # unsqueeze to avoid issues with torch.mv
    w_sum = w_sum.unsqueeze(1) # shape N_hidden x 1
    dh = dh.reshape((1,len(dh)))
    
    contractive_loss = torch.sum(torch.mm(dh**2, w_sum), 0)
    
    return mse + contractive_loss.mul_(lam)


#%% each slice
# images_allGenes: [gene, z, x, y]
#                  (15, 25, 2006, 2005)

for slice in range(images_allGenes.shape[1]):
    z1 = images_allGenes[:,slice,:,:] # 15,2006,2005
    transform = transforms.ToTensor()
    
    z1 = transform(z1)
    z1 = z1.permute(2,0,1) # 2006 x 2005 x 15
    z1 = torch.reshape(z1,(2006*2005,12))
    print(z1.shape)
    
    # normalize to [0,1]
    z1 = z1/z1.max()
    

    
    
    #%% train
    model = Autoencoder()
    model.double()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 1e-4, momentum = 0.9)
    
    epochs = 10
    encoded_list = list()
    decoded_list = list()
    loss_track = list()
    lam = 1e-4
    print('Start training')
    for epoch in range(epochs):
        for batch in range(len(z1)):
            vec = z1[batch, :]
            # print(vec.shape)
            encoded, decoded = model(vec)
            # loss = criterion(decoded, vec)
            W = model.state_dict()['fc1.weight']
            loss = loss_function(W, vec, decoded, encoded, lam)
            
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch == epochs - 1:
                encoded_list.append(encoded.detach().numpy())   # 15 x 4000000
                decoded_list.append(decoded.detach().numpy())
        loss_track.append(loss.item())    
    
    #%% save outputs
    import pickle
    with open('./CAE_result/CAE_encoded_z'+str(slice),'wb') as fp:
        pickle.dump(encoded_list, fp)
    
    with open('./CAE_result/CAE_decoded_z'+str(slice),'wb') as fp:
        pickle.dump(decoded_list, fp)
        
        
    # with open('CAE_encoded','rb') as fp:
    #     encoded_list = pickle.load(fp)
    # with open('CAE_decoded','rb') as fp:
    #     decoded_list = pickle.load(fp)
    
    
    
    #%% reshape
    encoded_list = np.array(encoded_list)
    encoded_vec = list()
    for i in range(encoded_list.shape[0]):
        encoded_vec.append(encoded_list[i,:])
        
    decoded_list = np.array(decoded_list)
    decoded_vec = list()
    for i in range(decoded_list.shape[0]):
        decoded_vec.append(decoded_list[i,:])
    
    #%% k-means, encoded
    kmeans = KMeans(n_clusters=48) 
    kmeans_encoded = kmeans.fit(encoded_vec)
    
    #%% k-means, decoded
    kmeans = KMeans(n_clusters=48) 
    kmeans_decoded = kmeans.fit(decoded_vec)
    #%%
    labels_encoded = kmeans_encoded.labels_
    labels_decoded = kmeans_decoded.labels_
    # print(labels_encoded.shape)
    # print(labels_decoded.shape)
    labels_encoded = labels_encoded.reshape((images_allGenes.shape[2],images_allGenes.shape[3]))
    labels_decoded = labels_decoded.reshape((images_allGenes.shape[2],images_allGenes.shape[3]))
    
    
    plt.imshow(labels_encoded, interpolation='nearest')
    plt.savefig('./CAE_result/CAE_encoded_z'+str(slice)+'.jpg', dpi = 300)
    plt.show()
    
    plt.imshow(labels_decoded, interpolation='nearest')
    plt.savefig('./CAE_result/CAE_decoded_z'+str(slice)+'.jpg', dpi = 300)
    plt.show()
    
    # plt.plot(loss_track)
    # plt.savefig('./CAE_result/CAE_z1_loss_plot.jpg',dpi = 300)
    # plt.show()
    
    
    
    #%% accuracy
    gt_file = np.load('./Dataset1_benchmark/labeled_mask_centroids.npy')
    
    
    #%% compare labeled centroids with pred
    pred_encoded = list()
    pred_decoded = list()
    acc_encoded = np.zeros([46,3])
    acc_decoded = np.zeros([46,3])
    
    gt = gt_file[slice,:,:]
    
    
    for i in range(1,int(gt.max())+1): # for each cell type, 1-46
        indx = np.where(gt == i) # coordinates of each cell of the same type
        
        # predicted labels for each gt label
        pred_encoded.append(labels_encoded[indx])
        pred_decoded.append(labels_decoded[indx])
        
        # [top 2 pred labels, tot freq]
        values, counts = np.unique(pred_encoded[i-1], return_counts=True)
        if len(counts) >= 3:
            ind_en = np.argpartition(-counts, kth=2)[:2]
            # ind_de = np.argmax(counts)
            acc_encoded[i-1,:] = [values[ind_en][0],values[ind_en][1],np.sum(counts[ind_en])/np.sum(counts)]
        elif len(counts) ==2:
            ind_en = np.argmax(counts)
            acc_encoded[i-1,:] = [values[0],values[1],1]
        elif len(counts) == 1:
            acc_encoded[i-1,:] = [values, 0, 1]
        
            
            
        values, counts = np.unique(pred_decoded[i-1], return_counts=True)
        if len(counts) >= 3:
            ind_de = np.argpartition(-counts, kth=2)[:2]
            # ind_de = np.argmax(counts)
            acc_decoded[i-1,:] = [values[ind_de][0],values[ind_de][1],np.sum(counts[ind_de])/np.sum(counts)]
        elif len(counts) ==2:
            ind_de = np.argmax(counts)
            acc_decoded[i-1,:] = [values[0],values[1],1]
        elif len(counts) == 1:
            acc_decoded[i-1,:] = [values, 0, 1]
    
    np.save('./CAE_result/acc_encoded_z'+str(slice)+'.npy', acc_encoded)
    np.save('./CAE_result/acc_decoded_z'+str(slice)+'.npy', acc_decoded)












