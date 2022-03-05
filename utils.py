import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from torchvision import models as models
from torch.utils.data import Dataset


def preprocess_from_pklpaths(pkl_paths):
    """
    read pickle to return (Xs,ys)
    Xs : [(time x feature) array of patient 1, (time x feature) array of patient 2, ... ]
    ys = [(time x label)   array of patient 1, (time x label)   array of patient 2, ... ]
    """
    df_values = []
    for dataPath in pkl_paths:
        dataPre = pd.read_pickle(dataPath)
        dataPre = dataPre.reset_index()
        dataPreGT = dataPre['class_gt']
        dataPre = pd.read_pickle(dataPath)
        dataPre = dataPre.reset_index()
        dataPreValues = dataPre.drop(columns=['index','Time',' Sample Count', ' Activity', ' SpO2 Confidence (%)',  ' SpO2 Percent Complete',
            ' Low Signal Quality', ' Motion Flag', ' WSPO2 Low Pi', ' Unreliable R',' R Value',
            ' SpO2 State', ' SCD State', ' SAMPLE Time', ' Walk Steps',
            ' Run Steps', ' Tot. Act. Energy', ' Ibi Offset', ' HR Confidence (%)',' RR', ' RR Confidence (%)',' KCal',
             ' Operating Mode'])   
        _idx = dataPreValues[dataPreValues[' SpO2 (%)'] == 0].index
        dataPreValues = dataPreValues.drop(_idx)
        #scaler = StandardScaler()
        scaler = MinMaxScaler()

        



        dataPreValues = minMax(dataPreValues, scaler)

        dataPreValues = dataPreValues.sort_index(ascending=True)
       
        dataPreValues =  dataPreValues.drop(columns = ['level_0', 'index'])
        df_values.append(dataPreValues)
    Xs = []
    ys = []
    for df in df_values:
        X= df.drop(columns='class_gt').reset_index(drop=True) 
        y= df['class_gt']
        y= labeling(y)
        Xs.append(X)
        ys.append(y)
    return (Xs,ys)





def minMax(dfIn, scaler):
            dfgt = dfIn['class_gt'].reset_index()
            dfval = dfIn.drop(columns=['class_gt']).reset_index()
            fitted = scaler.fit(dfval)
            output = scaler.transform(dfval)
            output = pd.DataFrame(output, columns=dfval.columns, index=list(dfval.index.values))
            # output = pd.DataFrame(output, columns=dfval.columns)
            output = pd.concat([output.reset_index() , dfgt.reset_index()], axis=1)
            return output




def labeling(dfIn):
    dfIn = dfIn.replace('SLEEP-S0', 0)
    dfIn = dfIn.replace('SLEEP-S1', 2)
    dfIn = dfIn.replace('SLEEP-S2', 2)
    dfIn = dfIn.replace('SLEEP-S3', 3)
    dfIn = dfIn.replace('SLEEP-REM', 1)
    dfIn = dfIn.reset_index(drop=True)
    return dfIn







class SleepDataset(Dataset):
    def __init__(self,input_datas,label_datas,seq_len=50,sampling_rate=5,channel_first=False):
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
        self.channel_first = channel_first
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
        if self.channel_first == True:
            x = torch.transpose(x,0,1)
        return (x,y)