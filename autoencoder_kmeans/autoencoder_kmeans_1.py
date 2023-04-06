#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 10:38:25 2023

@author: IreneZhou
"""

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
# import cv2
from sklearn.cluster import KMeans


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

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
images_allGenes = np.zeros((15,dim[0],dim[1],dim[2]))
images_allGenes[0,:,:,:] = images1

for i in range(2,16):
  path = './Dataset1_benchmark/Gene' + str(i) + '.tif'
  image = read_tiff(path)
  images_allGenes[i-1,:,:,:] = image



#%%  basic k-means
# images_allGenes: [gene, z, x, y]
#                  (15, 25, 2006, 2005)

# z1 = images_allGenes[:,0,:,:]


# kmeans = KMeans(n_clusters=48)
# data = list()

# for i in range(z1.shape[1]):
#     for j in range(z1.shape[2]):
#         data.append(z1[:,i,j])
        
        
# kmeans_result = kmeans.fit(data)

# labels = kmeans_result.labels_

# z1_labeled = labels.reshape((z1.shape[1],z1.shape[2]))

# from matplotlib import pyplot as plt
# plt.imshow(z1_labeled, interpolation='nearest')
# plt.savefig('./Dataset1_benchmark/simple_kmeans/z1_simpleKMeans_48.jpg', dpi = 300)

# plt.show()

#%% autoencoder
# images_allGenes: [gene, z, x, y]
#                  (15, 25, 2006, 2005)
z1 = images_allGenes[:,0,:,:] # 15,2006,2005
transform = transforms.ToTensor()

z1 = transform(z1)
z1 = z1.permute(2,0,1) # 2006 x 2005 x 15
z1 = torch.reshape(z1,(2006*2005,15))
print(z1.shape)

# normalize to [0,1]
z1 = z1/z1.max()


#%% model
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 2006*2005 (batch), 15
        self.encoder = nn.Sequential(
            nn.Linear(15, 12),
            nn.ReLU(),
            nn.Linear(12, 10),
            nn.ReLU(),
            nn.Linear(10,8),     
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(8,10),
            nn.ReLU(),
            nn.Linear(10,12),
            nn.ReLU(),
            nn.Linear(12,15),
            nn.Sigmoid()
        )
        
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

#%% train
model = Autoencoder()
model.double()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-4, momentum = 0.9)

epochs = 10
encoded_list = list()
decoded_list = list()
loss_track = list()
print('Start training')
for epoch in range(epochs):
    for batch in range(len(z1)):
        vec = z1[batch, :]
        # print(vec.shape)
        encoded, decoded = model(vec)
        loss = criterion(decoded, vec)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch == epochs - 1:
            encoded_list.append(encoded.detach().numpy())   # 15 x 4000000
            decoded_list.append(decoded.detach().numpy())
    loss_track.append(loss.item())    

#%% save outputs
import pickle

with open('encoded','wb') as fp:
    pickle.dump(encoded_list, fp)

with open('decoded','wb') as fp:
    pickle.dump(decoded_list, fp)
    
    
# with open('encoded','rb') as fp:
#     encoded_list = pickle.load(fp)
# with open('decoded','rb') as fp:
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
plt.savefig('./z1_encoded_48.jpg', dpi = 300)
plt.show()

plt.imshow(labels_decoded, interpolation='nearest')
plt.savefig('./z1_decoded_48.jpg', dpi = 300)
plt.show()


plt.plot(loss_track)
plt.savefig('./z1_loss_plot.jpg',dpi = 300)





















