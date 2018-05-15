#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import matplotlib.pyplot as plt
from .set_dataset import load_dataset, split_augment_data
from .set_model import setup_model
from keras.utils import np_utils
import pickle
import argparse

def train_model(data_path, res_path, img_resolution, epochs=100, batch_size=32, save_weights=True, save_history=True):
    
    data, labels = load_dataset(data_path, img_resolution)
    #labels = np_utils.to_categorical(labels, num_classes=1)
    
    train_generator, val_generator, len_train, len_val = split_augment_data(data, labels, batch_size)
    
    model = setup_model(img_resolution)
            
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=len_train/batch_size,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=len_val/batch_size)
    
    if save_weights==True:
        weights_filename = 'weights'+str(img_resolution)+'.h5'
        model.save_weights(os.path.join(res_path,weights_filename))
    if save_history==True:
        history_filename = 'history'+str(img_resolution)+'.p'
        with open(os.path.join(res_path,history_filename), 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
    
    return model, history

def display_loss_acc_curves(histories, resolutions):
    
    loss_legend = []
    plt.figure(figsize=[8,6])
    for i in range(len(histories)):
        plt.plot(histories[i].history['loss'],linewidth=3.0)
        plt.plot(histories[i].history['val_loss'],linewidth=3.0)
        loss_legend.append('Validation Loss '+ str(resolutions[i]))
        loss_legend.append('Training Loss '+ str(resolutions[i]))
    plt.legend(loss_legend,fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.title('Loss Curves',fontsize=16)
    
    acc_legend = []
    plt.figure(figsize=[8,6])
    for i in range(len(histories)):
        plt.plot(histories[i].history['acc'],linewidth=3.0)
        plt.plot(histories[i].history['val_acc'],linewidth=3.0)
        acc_legend.append('Validation Accuracy '+ str(resolutions[i]))
        acc_legend.append('Training accuracy '+ str(resolutions[i]))
    plt.legend(acc_legend,fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.title('Accuracy Curves',fontsize=16)
                

if __name__ == '__main__':
    
    # Parse arguments
    
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str, help="path to the data")
    parser.add_argument("res_path", type=str, help="path to the output directory")
    parser.add_argument("-e", "--epochs", type=int, help="number of epochs",default=32)
    parser.add_argument("-b", "--batch_size", type=int, help="batch_size",default=100)
    
    args = parser.parse_args()
    data_path = args.data_path
    res_path = args.res_path
    epochs = args.epochs
    batch_size = args.batch_size
    
    # Train models

    model256, history256 = train_model(data_path, res_path, 256, epochs=epochs, batch_size=batch_size)
    model512, history512 = train_model(data_path, res_path, 512, epochs=epochs, batch_size=batch_size)

    histories = [history256, history512]
    resolutions = [256,512]

    # Display loss and accuracy
    
    display_loss_acc_curves(histories, resolutions)
