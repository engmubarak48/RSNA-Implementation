# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 10:26:31 2019

@author: Jama Hussein Mohamud
"""

import glob, pylab, pandas as pd
import pydicom, numpy as np
from os import listdir
from os.path import isfile, join
import matplotlib.pylab as plt
import os
import seaborn as sns

from keras import layers
from keras.applications import DenseNet121
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback, ModelCheckpoint
from keras.models import Sequential
from keras.optimizers import Adam
from tqdm import tqdm
import cv2

#%%

test_images_dir = 'rsna-intracranial-hemorrhage-detection/stage_1_test_images/'
train_images_dir = 'rsna-intracranial-hemorrhage-detection/stage_1_train_images/'
train = pd.read_csv('rsna-intracranial-hemorrhage-detection/stage_1_train.csv')
train.head()

#%%

len(os.listdir(train_images_dir))

#%%

#Creating an empty test dataframe
test = pd.DataFrame(os.listdir(test_images_dir), columns = ['filename'])
test = pd.DataFrame(test['filename'].apply(lambda st: st.split('.')[0] + ".dcm"))
print(test.shape)
test.head()

#%%

#Creating a filename column in train dataframe
train['filename'] = train['ID'].apply(lambda st: "ID_" + st.split('_')[1] + ".dcm")
train['type'] = train['ID'].apply(lambda st: st.split('_')[2])
print(train.shape)
train.head()

#%%

train = train[['Label', 'filename', 'type']].drop_duplicates().pivot(index='filename', columns='type', values='Label').reset_index()
train = train.set_index("filename")
train.head()

#%%

def get_first_of_dicom_field_as_int(x):
    #get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
    if type(x) == pydicom.multival.MultiValue:
        return int(x[0])
    else:
        return int(x)
    
def get_windowing(data):
    dicom_fields = [data[('0028','1050')].value, #window center
                    data[('0028','1051')].value, #window width
                    data[('0028','1052')].value, #intercept
                    data[('0028','1053')].value] #slope
    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]

def sigmoid_window(img, window_center, window_width, U=1.0, eps=(1.0 / 255.0)):
    _, _, intercept, slope = get_windowing(img)
    img = img.pixel_array * slope + intercept
    ue = np.log((U / eps) - 1.0)
    W = (2 / window_width) * ue
    b = ((-2 * window_center) / window_width) * ue
    z = W * img + b
    img = U / (1 + np.power(np.e, -1.0 * z))
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return img

def sigmoid_bsb_window(img):
    brain_img = sigmoid_window(img, 40, 80)
    subdural_img = sigmoid_window(img, 80, 200)
    bone_img = sigmoid_window(img, 600, 2000)
    
    bsb_img = np.zeros((brain_img.shape[0], brain_img.shape[1], 3))
    bsb_img[:, :, 0] = brain_img
    bsb_img[:, :, 1] = subdural_img
    bsb_img[:, :, 2] = bone_img
    return bsb_img


#%%
    
def get_input(filename, data_dir):
    path = data_dir + filename#.replace('.dcm', '.png')
    img = pydicom.dcmread(path)
    #img = img.pixel_array
    img = sigmoid_bsb_window(img)
    
    return(img)
    
def get_output(filename, dataframe):
    y_col = ['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']
    
    #dataframe must be filename as index column
    filename = filename#.replace('.dcm', '.png')
    label = dataframe.loc[filename][y_col].values
    return label

def preprocess_input(img):
    return cv2.resize(img, (224,224))

def image_generator(data_dir, dataframe, batch_size=64):
    i=0
    filenames = np.random.permutation(dataframe.index)
    while True:
        batch_paths = filenames[i:i+batch_size]#np.random.choice(a=os.listdir(data_dir), size=batch_size)
        i += batch_size
        batch_inputs = np.zeros((batch_size, 224, 224, 3))#[]
        batch_outputs = np.zeros(((batch_size, 6))) #[]
        
        for index, input_path in enumerate(batch_paths):
            _input = get_input(input_path, data_dir)
            _output = get_output(input_path, dataframe)
            _input = preprocess_input(_input)
            
            batch_inputs[index, :, :, :] = _input
            batch_outputs[index, :] = _output 
            
        batch_x = np.array(batch_inputs)
        batch_y = np.split(np.array(batch_outputs), indices_or_sections=6, axis=1)

        yield(batch_x, batch_y)
        

#%%

train_data = train.sample(frac=0.9,random_state=0)
val_data = train_data.drop(train_data.index)


train_gen = image_generator(train_images_dir, train_data, batch_size=4)
val_gen = image_generator(train_images_dir, val_data, batch_size=4)
test_gen = image_generator(test_images_dir, test, batch_size=4)

#%%

import tensorflow as tf

num_classes = 6

tfk = tf.keras
tfka = tf.keras.applications

baseline_model = tfka.vgg19.VGG19(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

for layer in baseline_model.layers:
    layer.trainable = False

x = tfk.layers.Flatten()(baseline_model.output)
x = tfk.layers.Dense(128, activation="relu")(x)
x = tfk.layers.Dropout(0.25)(x)
x = tfk.layers.Dense(256, activation="relu")(x)
x = tfk.layers.Dropout(0.25)(x)
x = tfk.layers.Dense(1024, activation="relu")(x)
x = tfk.layers.Dropout(0.25)(x)
outputs = [tfk.layers.Dense(1, activation="sigmoid")(x) for i in range(6)]

model = tfk.Model(inputs=baseline_model.inputs, outputs=outputs)

print(model.summary())


#%%

model.compile(loss="binary_crossentropy", optimizer="adam", loss_weights=[2]+[1.0]*5)

checkpoint = tf.keras.callbacks.ModelCheckpoint("models/model.hdf5", save_best_only=True,verbose=1)

#%%

model.fit_generator(train_gen, epochs=5, steps_per_epoch=1000,
                    validation_steps=200, verbose=1, validation_data=val_gen, callbacks=[checkpoint])














