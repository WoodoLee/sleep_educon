import argparse
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torchvision import models as models
from torch.utils.data import DataLoader


import os
import argparse
from utils import *
from train import *
from models.LSTM import *


# Define directory
path_dir = "./data/Preprocessed"
file_list = os.listdir(path_dir)
pkl_list = []
for filename in file_list:
    pkl_list.append('./data/Preprocessed/'+filename)


# Train/test set split
train_dataPaths, test_dataPaths = train_test_split(pkl_list, train_size = 0.8, random_state = 1,shuffle=True)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(torch.cuda.get_device_name(0))



# Dataframe pickle to preprocessed dataframe lists

(X_trains, y_trains) = preprocess_from_pklpaths(train_dataPaths)
(X_tests, y_tests) = preprocess_from_pklpaths(test_dataPaths)




# Define hyperparameters

eval_interval = 1000

time_window = 500
sampling_rate = 1
seq_len = int(time_window / sampling_rate)

input_feature_size = X_trains[0].shape[1]
embedding_dim = 32
hidden_dim = 32
output_size = 5
learning_rate = 0.001
     
criterion = nn.CrossEntropyLoss()
 

# Build dataset & dataloader

train_data = SleepDataset(X_trains,y_trains,seq_len=seq_len,sampling_rate=sampling_rate) 
test_data = SleepDataset(X_tests,y_tests,seq_len=seq_len,sampling_rate=sampling_rate)
train_dataloader = DataLoader(train_data,batch_size=32,shuffle=True)
test_dataloader = DataLoader(test_data,batch_size = 32,shuffle=True)

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

# Define model

model = RecurrentClassifier(input_feature_size, embedding_dim, hidden_dim, output_size,model='GRU',act_layer= nn.Sigmoid)
model = model.to(device)

print(f'GRU, window = {time_window}s,sampling = {sampling_rate}s')

# Train & eval

train_net(model,train_dataloader,test_dataloader,n_iter=20,lr=learning_rate,device=device,eval_interval=eval_interval)