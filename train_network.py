#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import matplotlib.pyplot as plt
from set_dataset import load_dataset
from set_model import setup_model
from keras.utils import np_utils

img_size = 256
data_path = '/Users/fixdecroix/Downloads/training_challenge'

X, y = load_dataset(data_path, img_size)
y = np_utils.to_categorical(y, num_classes=2)
model = setup_model(img_size)
history = model.fit(X, y, validation_split=0.2, epochs=1, batch_size=32)

model.save_weights(os.path.join(data_path,'weights256_v1.h5'))  

# Loss Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['loss'],'r',linewidth=3.0)
plt.plot(history.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)
 
# Accuracy Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['acc'],'r',linewidth=3.0)
plt.plot(history.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)

