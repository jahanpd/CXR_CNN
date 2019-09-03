import os
from pathlib import Path
import numpy as np
import pandas as pd
from algorithms.CNN_regression import convNN as CNN
import tensorflow as tf
from imblearn.over_sampling import SMOTE
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

path = "/media/jahan/solo/"
os.chdir(path)

csv = "CheXpert-v1.0-small/train.csv"
test = "CheXpert-v1.0-small/valid.csv"

labels = pd.read_csv(csv)
valid = pd.read_csv(test)
paths = labels.Path
age = labels.Age
test_path = valid.Path
test_age = valid.Age

scaler = MinMaxScaler()
scaler.fit(age.values.reshape((-1,1)))

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    cnn = CNN(image_paths=paths.values, labels=scaler.transform(age.values.reshape((-1,1))),
              test_x = test_path, test_y=scaler.transform(test_age.values.reshape((-1,1))),
            save_path="/home/jahan/Documents/CXR_CNN/output/age/")
    cnn.build()

with strategy.scope():
    history = cnn.train(100)
