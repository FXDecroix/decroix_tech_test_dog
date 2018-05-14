#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import matplotlib.pyplot as plt
from set_dataset import load_dataset
from set_model import setup_model
from keras.utils import np_utils
import pickle
import argparse


# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("data_path", type=str, help="path to the data")
parser.add_argument("res_path", type=str, help="path to the output directory")

args = parser.parse_args()

data_path = args.data_path
res_path = args.res_path

# Load data and train model for image size 256

X_256, y_256 = load_dataset(data_path, 256)
y_256 = np_utils.to_categorical(y_256, num_classes=2)
model256 = setup_model(256)
history256 = model256.fit(X_256, y_256, validation_split=0.2, epochs=1, batch_size=32)

model256.save_weights(os.path.join(res_path,'weights256_v1.h5')) 

with open(os.path.join(res_path,'trainHistoryDict256.p'), 'wb') as file_pi:
    pickle.dump(history256.history, file_pi)

# Load data and train model for image size 512

X_512, y_512 = load_dataset(data_path, 512)
y_512 = np_utils.to_categorical(y_512, num_classes=2)
model512 = setup_model(512)
history512 = model512.fit(X_512, y_512, validation_split=0.2, epochs=1, batch_size=32)

model512.save_weights(os.path.join(res_path,'weights512_v1.h5')) 

with open(os.path.join(res_path,'trainHistoryDict512.p'), 'wb') as file_pi:
    pickle.dump(history512.history, file_pi) 

# Loss Curves
    
plt.figure(figsize=[8,6])
plt.plot(history256.history['loss'],'r',linewidth=3.0)
plt.plot(history256.history['val_loss'],'b',linewidth=3.0)
plt.plot(history512.history['loss'],'g',linewidth=3.0)
plt.plot(history512.history['val_loss'],'k',linewidth=3.0)
plt.legend(['Training loss 256', 'Validation Loss 256', 'Training loss 512', 'Validation Loss 512'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)
 
# Accuracy Curves

plt.figure(figsize=[8,6])
plt.plot(history256.history['acc'],'r',linewidth=3.0)
plt.plot(history256.history['val_acc'],'b',linewidth=3.0)
plt.plot(history512.history['acc'],'r',linewidth=3.0)
plt.plot(history512.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy','Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)

