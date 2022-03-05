import pandas as pd
import argparse
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torchvision import models as models
from torch.utils.data import DataLoader
from models.SVM import *
from models.KNN import *
from models.DT import *
from models.ExtraTrees import *
from models.RandomForest import *
from models.Logistic import *
from models.GradientBoosting import *
import os
import argparse
from utils import *
from train import *

path_dir = "./data/Preprocessed"
file_list = os.listdir(path_dir)
pkl_list = []
for filename in file_list:
    pkl_list.append('./data/Preprocessed/'+filename)
    
train_dataPaths, test_dataPaths = train_test_split(pkl_list, train_size = 0.8, random_state = 1,shuffle=True)

(X_trains, y_trains) = preprocess_from_pklpaths(train_dataPaths)
(X_tests, y_tests)   = preprocess_from_pklpaths(test_dataPaths)

X_train = pd.concat(X_trains)
y_train = pd.concat(y_trains)
X_test  = pd.concat(X_tests)
y_test  = pd.concat(y_tests)


print("start decision tree")
Decision_Tree(X_train,X_test,y_train,y_test)
print("start random forest")
RandomForest(X_train,X_test,y_train,y_test)
print("start extra tree")
ExtraTrees(X_train,X_test,y_train,y_test)
print("start gradient boosting")
GradientBoosting(X_train,X_test,y_train,y_test)
print("start KNeighbors")
KNN(X_train,X_test,y_train,y_test)
print("start logistic regression")
Logistic(X_train,X_test,y_train,y_test)
print("start support vector machine")
SVM(X_train,X_test,y_train,y_test)