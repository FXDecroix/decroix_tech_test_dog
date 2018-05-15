#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import csv
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

def load_dataset(path, img_resolution):
    
    dir_images= os.path.join(path,'images')
    extension = '.jpg'
    nb_images = len([f for f in os.listdir(dir_images) if f.endswith('.jpg')])

    full_dataset = np.empty((nb_images, img_resolution, img_resolution, 3), dtype=np.uint8) # pre-allocating to avoid making a copy of the data
    labels = np.empty((nb_images,), dtype=np.uint8)

    with open(os.path.join(path,'labels.csv')) as labelfile:
        next(labelfile) # skip first line
        labelreader = csv.reader(labelfile, delimiter = ',')
   
        for i, row in enumerate(labelreader):
        
            img = image.load_img(os.path.join(dir_images,row[0]+extension), target_size=(img_resolution, img_resolution))
            img = image.img_to_array(img)
                
            full_dataset[i, ...] = preprocess_input(img)
            labels[i] = int(row[1])
            
            sys.stdout.write('\r')
            sys.stdout.write("Load and resize dataset : [%-20s] %d%%" % ('='*int(i/500+1), i/100+1))
            sys.stdout.flush()
            
    sys.stdout.write('\n')
    return full_dataset, labels

def split_augment_data(full_dataset, labels, batch_size):
    
    X_train, X_test, y_train, y_test = train_test_split(full_dataset, labels, test_size=0.2, random_state=42)
    
    gen = ImageDataGenerator(horizontal_flip = True,
                         vertical_flip = True,
                         width_shift_range = 0.1,
                         height_shift_range = 0.1,
                         zoom_range = 0.1,
                         rotation_range = 10
                        )
        
    train_generator = gen.flow(X_train, y_train, batch_size = batch_size)
    val_generator = gen.flow(X_test, y_test, batch_size = batch_size)
    
    return train_generator, val_generator, len(X_train), len(X_test)
    
    

    







        
        
        