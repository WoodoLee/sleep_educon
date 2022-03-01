import pandas as pd
import argparse
#from rich import print
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from torchvision import models as models

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable
#import pretty_errors
from torch.utils.data import Dataset, DataLoader
import torch.autograd as autograd
import os

from resnet1d.net1d import Net1D
from multi_scale_1d.multi_scale_ori import MSResNet

from resnet1d.resnet1d import ResNet1D

import copy

path_dir = "./data/Preprocessed"
file_list = os.listdir(path_dir)


pkl_list = []
for filename in file_list:
    pkl_list.append('./data/Preprocessed/'+filename)

train_dataPaths, test_dataPaths = train_test_split(pkl_list, train_size = 0.8, random_state = 1,shuffle=True)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(torch.cuda.get_device_name(0))

train_train_ratio = 0.80
train_test_ratio = 0.10
train_valid_ratio = 0.10
dfTrains = []

dfTests = []
embedding_dim = 20
hidden_dim = 25
output_size = 5


def minMax(dfIn, scaler):
            dfgt = dfIn['class_gt'].reset_index()
            dfval = dfIn.drop(columns=['class_gt']).reset_index()
            fitted = scaler.fit(dfval)
            output = scaler.transform(dfval)
            output = pd.DataFrame(output, columns=dfval.columns, index=list(dfval.index.values))
            # output = pd.DataFrame(output, columns=dfval.columns)
            output = pd.concat([output.reset_index() , dfgt.reset_index()], axis=1)
            return output

for dataPaths in [train_dataPaths,test_dataPaths]:
    for dataPath in dataPaths:
        dataPre = pd.read_pickle(dataPath)
        dataPre = dataPre.reset_index()
        dataPreGT = dataPre['class_gt']
        dataPre = pd.read_pickle(dataPath)
        dataPre = dataPre.reset_index()

        # dataPreValues = dataPre.drop(columns=['index','Time',' Sample Count', ' Activity', ' SpO2 Confidence (%)',  ' SpO2 Percent Complete',
        #     ' Low Signal Quality', ' Motion Flag', ' WSPO2 Low Pi', ' Unreliable R',' SpO2 (%)',
        #     ' SpO2 State', ' SCD State', ' SAMPLE Time', ' Walk Steps',' R Value',' Heart Rate (bpm)'
        #     ' Run Steps',' X Axis Acceleration (g)',' Y Axis Acceleration (g)',' Z Axis Acceleration (g)', ' KCal', ' Tot. Act. Energy', ' Ibi Offset', ' HR Confidence (%)',' RR', ' RR Confidence (%)',
        #      ' Operating Mode',' Green Count',' Green2 Count',' IR Count',' Red Count'])
        dataPreValues = dataPre.drop(columns=['index','Time',' Sample Count', ' Activity', ' SpO2 Confidence (%)',  ' SpO2 Percent Complete',
            ' Low Signal Quality', ' Motion Flag', ' WSPO2 Low Pi', ' Unreliable R',' R Value',
            ' SpO2 State', ' SCD State', ' SAMPLE Time', ' Walk Steps',
            ' Run Steps', ' Tot. Act. Energy', ' Ibi Offset', ' HR Confidence (%)',' RR', ' RR Confidence (%)',' KCal',
             ' Operating Mode'])   
        _idx = dataPreValues[dataPreValues[' SpO2 (%)'] == 0].index
        dataPreValues = dataPreValues.drop(_idx)
        
        # kcal, r value, spo2, heart rate only     
        # R value, Kcal, sp02%,sample time은 라벨과 강한 선형 상관관계를 가지는데.. 이 데이터가 뭔지 알아야할거같다.
        #Kcal, sample time은 시간과 정비례한다. 쳐내.

        scaler =MinMaxScaler()
        #scaler = StandardScaler()
        #scaler = RobustScaler()

        



        dataPreValues = minMax(dataPreValues, scaler)

        # 훈련/테스트 분할
        #dfTrain, dfTest = train_test_split(dataPreValues, train_size = train_train_ratio, random_state = 1)
        #dfTrain, dfTest = train_test_split(dataPreValues, train_size = 0.5, random_state = 1,shuffle=True)

        # 훈련/검증 분할
        #dfTest, dfValid = train_test_split(dfTest, train_size = 0.5, random_state = 1)
        #dfTrain = dfTrain.sort_index(ascending=True)
        dataPreValues = dataPreValues.sort_index(ascending=True)
        #dfTest = dfTest.sort_index(ascending=True)

        #dfTrain =  dfTrain.drop(columns = ['level_0', 'index'])
        dataPreValues =  dataPreValues.drop(columns = ['level_0', 'index'])
        if dataPaths == train_dataPaths:
            dfTrains.append(dataPreValues)
        elif dataPaths == test_dataPaths:
            dfTests.append(dataPreValues)
        #dfTest =  dfTest.drop(columns = ['level_0', 'index'])
        #dfValid =  dfValid.drop(columns = ['level_0', 'index'])


def labeling(dfIn):
    dfIn = dfIn.replace('SLEEP-S0', 0)
    dfIn = dfIn.replace('SLEEP-S1', 2)
    dfIn = dfIn.replace('SLEEP-S2', 2)
    dfIn = dfIn.replace('SLEEP-S3', 3)
    dfIn = dfIn.replace('SLEEP-REM', 1)
    dfIn = dfIn.reset_index(drop=True)
    return dfIn


X_trains = []
y_trains = []
X_tests = []
y_tests = []
for dfTrain in dfTrains:

    X_train = dfTrain.drop(columns='class_gt').reset_index(drop=True) 
    
    y_train = dfTrain['class_gt']
    y_train = labeling(y_train)
    X_trains.append(X_train)
    y_trains.append(y_train)
for dfTest in dfTests:

    X_test = dfTest.drop(columns='class_gt').reset_index(drop=True) 
    y_test = dfTest['class_gt']
    y_test = labeling(y_test)
    X_tests.append(X_test)
    y_tests.append(y_test)


Xs = []
ys = []
for i in range(len(X_trains)):
    Xs.append(X_trains[i].iloc[:]) 
    ys.append(y_trains[i].iloc[:]) 



#Encoder. 
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, dropout=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dropout = nn.Dropout(dropout)
      
        self.fc2 = nn.Linear(hidden_features, out_features)
        

    def forward(self, x):

        if len(x.shape) == 3:
            batch = x.shape[0]
            seq = x.shape[1]
            x = x.reshape(batch * seq,-1)
            x = self.fc1(x)
            x = self.dropout(x)
            x = self.act(x)
            x = self.dropout(x)
            x = self.fc2(x)
            
            x = x.reshape(batch,seq,-1)
            return x

        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
       
        
        return x

class SleepDataset(Dataset):
    def __init__(self,input_datas,label_datas,seq_len=50,sampling_rate=5,channel_first=True):
        # 정수형 라벨을 원핫라벨로.

        """
        input_datas = list of input pandas dataframes
        labal_datas = list of label pandas dataframes
        seq_len = lstm에 들어갈 시퀀스 개수
        sampleing_rate = 샘플링 interval sampling_rate = 5이면 5s중 한번씩만 샘플링함.
        25 sampling per second
        """
        self.channel_first= channel_first
        self.num_datas = len(input_datas)
        self.section_marks = [] # used for idx matching. if idx is between section marks, then that idx indicate nth data group.
        self.seq_len = seq_len
        self.sampling_rate = int(sampling_rate * 25)
        self.seq_interval = seq_len * self.sampling_rate
        #label_data = pd.get_dummies(label_data)
        section_mark = 0
        input_len = 0
        for input_data in input_datas:
            input_data = input_data.to_numpy()
            input_len += input_data.shape[0]
            section_mark += input_data.shape[0] - self.seq_interval # 
            self.section_marks.append(section_mark)
        for label_data in label_datas:
            label_data = label_data.to_numpy()
        


        self.input_len = input_len
        self.x_data = np.concatenate(input_datas)
        
        self.y_data = np.concatenate(label_datas)

    def __len__(self):
        return self.input_len - self.num_datas * self.seq_interval

    def __getitem__(self,idx):
        _idx = idx
        for i, section_mark in enumerate(self.section_marks):
            if idx >= section_mark:
                
                _idx = _idx + self.seq_interval
                




        fancy_index = np.arange(_idx,_idx+self.seq_interval,self.sampling_rate)
        x = self.x_data[_idx:_idx+self.seq_interval]
        x = x.reshape((self.seq_len,self.sampling_rate,-1))
        x = torch.FloatTensor(x).mean(dim=1)
        


        #x = torch.FloatTensor(self.x_data[fancy_index,:])
        y = torch.LongTensor(self.y_data[fancy_index])
        if self.channel_first == True:
            x = torch.transpose(x,0,1)
        return (x,y)



feature_size = X_train.shape[1]


criterion = nn.CrossEntropyLoss()
 

learning_rate = 0.001

def eval_net(model,data_loader,device,break_idx = None):
    model.eval()
    ys = []
    ypreds = []
    for i,(x,y) in enumerate(data_loader):
        x = x.to(device)
        y = y[:,-1]
        y = y.to(device)

        with torch.no_grad():
            _,y_pred = model(x).max(1)
        ys.append(y)
        ypreds.append(y_pred)

        if break_idx == i:
            break

    ys = torch.cat(ys)
    ypreds = torch.cat(ypreds)

    acc= (ys == ypreds).float().sum() / len(ys)
    return acc.item()

def train_net(model,train_dataloader,test_dataloader,optimizer_cls = optim.Adam, criterion = nn.CrossEntropyLoss(),n_iter=10,device='cpu',lr = 0.001):
        


        train_losses = []
        train_acc = []
        val_acc = []
        optimizer = optimizer_cls(model.parameters(),lr=lr)
        

        for epoch in range(n_iter):
                running_loss = 0.0
                model.train()
                n = 0
                n_acc = 0
                for i, (xx, yy) in tqdm(enumerate(train_dataloader)):
                
                
                        xx = xx.to(device)

                        yy = yy[:,-1] # yy는 원래 xx의 시퀀스 50개에 대한 라벨을 다 담고있지만, lstm에서 마지막만 쓸거임
                        yy = yy.to(device)
                        
                
                        optimizer.zero_grad()
                        outputs = model(xx)

                        # Calculate Loss: softmax --> cross entropy loss
                        loss = criterion(outputs, yy)


                        # Getting gradients w.r.t. parameters
                        loss.backward()

                        # Updating parameters
                        optimizer.step()
                        if i % 3000 == 0:
                                copy_loader = copy.deepcopy(test_dataloader)
                                val = eval_net(model,copy_loader,device,break_idx = 4000)
                                model.train()
                                print(f'iter = {i} val_acc = {val}')
                        
                        i += 1
                        n += len(xx)
                        _, y_pred = outputs.max(1)
                        n_acc += (yy == y_pred).float().sum().item()
                train_losses.append(running_loss/i)
                train_acc.append(n_acc/n)

                val_acc.append(eval_net(model,test_dataloader,device))

                print(f'epoch : {epoch}, train_acc : {train_acc[-1]}, validation_acc : {val_acc[-1]}',flush = True)


model = Net1D(
        in_channels=feature_size,
        base_filters=16,
        ratio=1.0,
        filter_list = [16, 32, 32],
        m_blocks_list = [2, 2, 3],
        kernel_size=3,
        stride=2,
        groups_width=16,
        verbose=False,
        n_classes=5)


train_data = SleepDataset(X_trains,y_trains,seq_len=500,sampling_rate=1)
test_data = SleepDataset(X_tests,y_tests,seq_len=500,sampling_rate=1)
train_dataloader = DataLoader(train_data,batch_size=32,shuffle=True)
test_dataloader = DataLoader(test_data,batch_size = 32,shuffle=True)


model = model.to(device)


train_net(model,train_dataloader,test_dataloader,n_iter=20,device=device)