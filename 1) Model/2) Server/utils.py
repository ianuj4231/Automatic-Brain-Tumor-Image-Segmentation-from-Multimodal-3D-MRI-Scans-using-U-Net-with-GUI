# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 21:07:11 2023

@author: ANUJ
"""

import numpy as np
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
scaler= MinMaxScaler()

model=load_model("D:/pfm/brats2020/brats_3d.hdf5", compile=False)

def preprocess(t1ce_img, t2_img, flair_img):
    flair_img=scaler.fit_transform(flair_img.reshape(-1, flair_img.shape[-1])).reshape(flair_img.shape)
    t1ce_img=scaler.fit_transform(t1ce_img.reshape(-1, t1ce_img.shape[-1])).reshape(t1ce_img.shape)
    t2_img=scaler.fit_transform(t2_img.reshape(-1, t2_img.shape[-1])).reshape(t2_img.shape)

    stacked_img = np.stack([flair_img,t1ce_img, t2_img ], axis=3)

    cropped_img = stacked_img[56:184, 56:184, 13:141, :]

    return cropped_img


def predict_function(img_data):
    img_data_batched = np.expand_dims(img_data, axis=0)
    prediction = model.predict(img_data_batched)
    prediction_argmax = np.argmax(prediction, axis=4)[0, :, :, :]
    return prediction_argmax

def plot_slices(img, mask, prediction, n_slice):
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    
    axs[0, 0].set_title('Flair Modality', {'color': 'red'}, fontweight='bold', fontsize=14)
    axs[0, 0].imshow(img[:, :, n_slice, 0], cmap='gray')
    axs[0, 1].set_title('T1ce Modality', {'color': 'red'}, fontweight='bold', fontsize=14)
    axs[0, 1].imshow(img[:, :, n_slice, 1], cmap='gray')
    axs[0, 2].set_title(' T2 Modality', {'color': 'red'}, fontweight='bold', fontsize=14)
    axs[0, 2].imshow(img[:, :, n_slice, 2], cmap='gray')
    axs[1, 0].set_title('Ground Truth Mask', {'color': 'red'}, fontweight='bold', fontsize=14)
    axs[1, 0].imshow(mask[:, :, n_slice])
    axs[1, 1].set_title('Voxel-level Segmentation', {'color': 'red'}, fontweight='bold', fontsize=14)
    axs[1, 1].imshow(prediction[:, :, n_slice])
    axs[1, 2].remove()
    
    fig.savefig('output.png') 
    return fig



