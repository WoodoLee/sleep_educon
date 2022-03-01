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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torchvision import models as models

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable
#import pretty_errors
from torch.utils.data import Dataset, DataLoader
import torch.autograd as autograd

train_dataPaths = ['./data/patient-01_0722.pkl','./data/patient-01_0805.pkl']
test_dataPaths = ['./data/patient-02_0812.pkl']


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(torch.cuda.get_device_name(0))

train_train_ratio = 0.80
train_test_ratio = 0.10
train_valid_ratio = 0.10
dfTrains = []

dfTests = []


def minMax(dfIn, scaler):
            dfgt = dfIn['class_gt'].reset_index()
            dfval = dfIn.drop(columns=['class_gt']).reset_index()
            fitted = min_max_scaler.fit(dfval)
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

        dataPreValues = dataPre.drop(columns=['index','Time',' Sample Count', ' Activity', ' SpO2 Confidence (%)',  ' SpO2 Percent Complete',
            ' Low Signal Quality', ' Motion Flag', ' WSPO2 Low Pi', ' Unreliable R',
            ' SpO2 State', ' SCD State', ' SAMPLE Time', ' Walk Steps',' R Value',
            ' Run Steps', ' KCal', ' Tot. Act. Energy', ' Ibi Offset', ' HR Confidence (%)',' RR', ' RR Confidence (%)', ' Operating Mode'])
        # R value, Kcal, sp02%,sample time은 라벨과 강한 선형 상관관계를 가지는데.. 이 데이터가 뭔지 알아야할거같다.
        #Kcal, sample time은 시간과 정비례한다. 쳐내.

        min_max_scaler = MinMaxScaler()

        



        dataPreValues = minMax(dataPreValues, min_max_scaler)

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
    dfIn = dfIn.replace('SLEEP-S1', 1)
    dfIn = dfIn.replace('SLEEP-S2', 2)
    dfIn = dfIn.replace('SLEEP-S3', 3)
    dfIn = dfIn.replace('SLEEP-REM', 4)
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


class SleepDataset(Dataset):
    def __init__(self,input_datas,label_datas,seq_len=50,sampling_rate=5):
        # 정수형 라벨을 원핫라벨로.

        """
        input_datas = list of input pandas dataframes
        labal_datas = list of label pandas dataframes
        seq_len = lstm에 들어갈 시퀀스 개수
        sampleing_rate = 샘플링 interval sampling_rate = 5이면 5s중 한번씩만 샘플링함.
        25 sampling per second
        """
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
        


        x = torch.FloatTensor(self.x_data[fancy_index,:])
        y = torch.LongTensor(self.y_data[fancy_index])
        return (x,y)


#Encoder. 
class MLP(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, hidden_features)
        self.fc3 = nn.Linear(hidden_features, out_features)
        

    def forward(self, x):

        if len(x.shape) == 3:
            batch = x.shape[0]
            seq = x.shape[1]
            x = x.reshape(batch * seq,-1)
            x = self.fc1(x)
            x = self.act(x)
            x = self.fc2(x)
            x = self.act(x)
            x = self.fc3(x)
            x = x.reshape(batch,seq,-1)
            return x

        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        
        return x


class RecurrentClassifier(nn.Module):

	def __init__(self, feature_size, embedding_dim, hidden_dim, output_size,model='LSTM'):
		"""
		feature_size : 인풋 시퀀스 피쳐 사이즈
		embedding_dim : LSTM 인풋 피쳐 사이즈
		hidden_dim : LSTM 히든 레이어 사이즈.
		인풋 : (batch, seq, feature_size)
		"""
		super(RecurrentClassifier, self).__init__()
		self.model = model
		self.embedding_dim = embedding_dim
		self.hidden_dim = hidden_dim
		self.feature_size = feature_size
		if model == 'LSTM':
			self.rec = nn.LSTM(embedding_dim, hidden_dim, num_layers=1,batch_first=True)
		elif model == 'GRU':
			self.rec = nn.GRU(embedding_dim, hidden_dim, num_layers=1,batch_first=True)
		elif model == 'RNN':
			self.rec = nn.RNN(embedding_dim, hidden_dim, num_layers=1,batch_first=True)
		else:
			assert()

		#self.hidden2out = MLP(hidden_dim, out_features=output_size)
		self.hidden2out = nn.Linear(hidden_dim,output_size)
		self.softmax = nn.LogSoftmax()
		self.encoder = MLP(feature_size,hidden_features=100,out_features=embedding_dim)
		#self.Mlp = (feature_size,embedding_dim)
		

		self.dropout_layer = nn.Dropout(p=0.2)


	def init_hidden(self, batch_size):
		return(autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)),
						autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)))


	def forward(self, batch):
		
		self.hidden = self.init_hidden(batch.size(-1))

		embeds = self.encoder(batch)
		if self.model == 'LSTM':
		
			outputs, (ht, ct) = self.rec(embeds)
		else:
			outputs, ht = self.rec(embeds)

		# ht is the last hidden state of the sequences
		# ht = (1 x batch_size x hidden_dim)
		# ht[-1] = (batch_size x hidden_dim)
		output = self.dropout_layer(ht[-1])
		output = self.hidden2out(output)
		#output = self.softmax(output) #criterion에 softmax를 썼기 때문에 붙이면 안됨.

		return output   #output에 5차원으로 해서 라벨을 정수화한것과 비교...? 라벨이 5차원이되어야함


# 전처리된 X_train을 데이터로더로. 만일 5분에 한번씩 하는 seq를 보고싶으면 샘플링 후 데이터로더에. seq_len은 아마 한번 라벨이 바뀌는 시간 정도가 적당할듯.
# 초당 25번 샘플링. 100을 샘플링 레이트로 하면 4초에 한번 샘플링 하는거. 
train_data = SleepDataset(X_trains,y_trains,seq_len=50,sampling_rate=10)
test_data = SleepDataset(X_tests,y_tests,seq_len=50,sampling_rate=10)
train_dataloader = DataLoader(train_data,batch_size=32,shuffle=True)
test_dataloader = DataLoader(test_data,batch_size = 32,shuffle=True)



feature_size = X_train.shape[1]
embedding_dim = 20
hidden_dim = 25
output_size = 5
 
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

model = RecurrentClassifier(feature_size, embedding_dim, hidden_dim, output_size,model='LSTM')
model = model.to(device)
     
criterion = nn.CrossEntropyLoss()
 

learning_rate = 0.001


def eval_net(model,data_loader,device):
    model.eval()
    ys = []
    ypreds = []
    for x,y in data_loader:
        x = x.to(device)
        y = y[:,-1]
        y = y.to(device)

        with torch.no_grad():
            _,y_pred = model(x).max(1)
        ys.append(y)
        ypreds.append(y_pred)

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
                        
                        
                        i += 1
                        n += len(xx)
                        _, y_pred = outputs.max(1)
                        n_acc += (yy == y_pred).float().sum().item()
                train_losses.append(running_loss/i)
                train_acc.append(n_acc/n)

                val_acc.append(eval_net(model,test_dataloader,device))

                print(f'epoch : {epoch}, train_acc : {train_acc[-1]}, validation_acc : {val_acc[-1]}',flush = True)

                #in the last epoch validation we have to add confusion matrix of inference.
                #plus, we have to implement test_net function.

model_RNN = RecurrentClassifier(feature_size, embedding_dim, hidden_dim, output_size,model='RNN')
model_RNN = model_RNN.to(device)
print('RNN, window = 500s,sampling = 10s')
train_net(model_RNN,train_dataloader,test_dataloader,n_iter=20,device=device)