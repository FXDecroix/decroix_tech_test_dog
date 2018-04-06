#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 18:44:29 2018

@author: fixdecroix
"""
import os, sys
import csv
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

width=256
height=256

dir_data = '/Users/fixdecroix/Downloads/training_challenge' # ToDo : argparse
dir_images= os.path.join(dir_data,'images')
extension = '.jpg'

full_dataset = np.empty((10000, 256, 256, 3), dtype=np.uint8) # pre-allocating to avoid making a copy of the data
labels = np.empty((10000,), dtype=np.uint8)

with open(os.path.join(dir_data,'labels.csv')) as labelfile:
    next(labelfile) # skip first line
    labelreader = csv.reader(labelfile, delimiter = ',')
   
    for i, row in enumerate(labelreader):
        img = cv2.imread(os.path.join(dir_images,row[0]+extension), cv2.IMREAD_COLOR)
        full_dataset[i, ...] = cv2.resize(img,(width,height),interpolation = cv2.INTER_AREA) # INTER_AREA better for shrinking
        labels[i] = int(row[1])
        
        sys.stdout.write('\r')
        sys.stdout.write("Load and resize dataset : [%-20s] %d%%" % ('='*int(i/500+1), i/100+1))
        sys.stdout.flush()
        
# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(full_dataset, labels, test_size=0.1, random_state=42)            

            



        
        
        