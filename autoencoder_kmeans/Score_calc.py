#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 11:05:14 2023

@author: IreneZhou
"""
import matplotlib.pyplot as plt
import numpy as np

simple_kmeans_z10 = np.load('./CAE_result/simple_kmeans_z10.npy')
acc_encoded_z10 = np.load('./CAE_result/acc_encoded_z10.npy')
acc_decoded_z10 = np.load('./CAE_result/acc_decoded_z10.npy')

score_kmeans = np.sum(simple_kmeans_z10[:,2])/46
score_encoded = np.sum(acc_encoded_z10[:,2])/46
score_decoded = np.sum(acc_decoded_z10[:,2])/46

x = [score_kmeans, score_encoded,score_decoded]
y = ['k-means', 'CAE, encoded', 'CAE, decoded']

plt.bar(y,x,width = 0.5)
plt.savefig('./CAE_result/Score_z10.jpg', dpi=300)
plt.show()