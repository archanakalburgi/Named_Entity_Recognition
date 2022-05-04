This project is done towards fulfilling the requirements of the course **CS 541 Artificial Intelligence** offered by Stevens Institute of Technology.  

Follow the steps to successfully run the notebook :

1. Install required packages in requirement.txt 
2. To run on local machine : execute following command
```sh
pip install -r requirements.txt
```
2. Load the following packages 
```sh
import pickle
import operator
import re
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from plot_keras_history import plot_history
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix
from keras_contrib.utils import save_load_utils

from keras import layers
from keras import optimizers

from keras.models import Model
from keras.models import Input

from keras_contrib.layers import CRF
# from tensorflow_addons.layers import CRF
from keras_contrib import losses
from keras_contrib import metrics
from tensorflow import keras
```
3. Load the data titled 'ner_dataset.csv' using *pandas*
```sh
data_df = pd.read_csv('ner_dataset.csv', encoding="iso-8859-1", header=0)
```
4. Transform the textual data to the numeric data that could be fed to the neural network 
5. Split the data into train and test 
```sh
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1234)
```
6. Train the models
7. Test on the validation set