#!/usr/bin/env python3
# -*- coding: utf-8 -*-
   
from keras.applications.resnet50 import ResNet50  
from keras.models import Model   
from keras.layers import GlobalAveragePooling2D, Dense

def setup_model(img_resolution):
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape = (img_resolution, img_resolution, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(2, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

if __name__ == '__main__':
    model = setup_model(256)
    model.summary()

