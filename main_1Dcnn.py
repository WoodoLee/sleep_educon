
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
from resnet1d.net1d import Net1D
from multi_scale_1d.multi_scale_ori import MSResNet
from resnet1d.resnet1d import ResNet1D


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



 
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

# Define hyperparameters

eval_interval = 100

embedding_dim = 20
hidden_dim = 25
output_size = 5

learning_rate = 0.0001
feature_size = X_trains[0].shape[1]
criterion = nn.CrossEntropyLoss()

time_window = 5000
sampling_rate = 1
seq_len = int(time_window / sampling_rate)


print(f'1DCNN, window = {time_window}s,sampling = {sampling_rate}s')


# Define model

model = Net1D(
        in_channels=feature_size,
        base_filters=16,
        ratio=1.0,
        filter_list = [16, 32, 32],
        m_blocks_list = [2, 2, 3],
        kernel_size=3,
        stride=2,
        groups_width=8,
        verbose=False,
        n_classes=5)

# Build dataset & dataloader

train_data = SleepDataset(X_trains,y_trains,seq_len=500,sampling_rate=1,channel_first=True)# 1DCNN uses (batch, channel, seq_len) input, while lstm uses (batch, seq_len, channel). 
test_data = SleepDataset(X_tests,y_tests,seq_len=500,sampling_rate=1,channel_first=True)
train_dataloader = DataLoader(train_data,batch_size=32,shuffle=True)
test_dataloader = DataLoader(test_data,batch_size = 32,shuffle=True)


model = model.to(device)

# Train & eval

train_net(model,train_dataloader,test_dataloader,n_iter=20,lr=learning_rate,device=device,eval_interval=eval_interval)