import torch
import torch.nn as nn
import torch.autograd as autograd


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.2):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, out_features)
        self.act = act_layer()
        self.dropout = nn.Dropout(drop)
        
        

    def forward(self, x):

        if len(x.shape) == 3:
            batch = x.shape[0]
            seq = x.shape[1]
            x = x.reshape(batch * seq,-1)
            x = self.fc1(x)
            x = self.act(x)
            x = self.dropout(x)
            
            x = x.reshape(batch,seq,-1)
            return x

        x = self.fc1(x)
        x = self.act(x)
        
       
        
        return x

class RecurrentClassifier(nn.Module):

	def __init__(self, feature_size, embedding_dim, hidden_dim, output_size,act_layer = nn.Sigmoid, model='LSTM'):
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
		self.act_layer = act_layer()
		self.feature_size = feature_size
		if model == 'LSTM':
			self.rec = nn.LSTM(embedding_dim, hidden_dim,bidirectional=True,num_layers=3,batch_first=True,dropout=0.5)
		elif model == 'GRU':
			self.rec = nn.GRU(embedding_dim, hidden_dim, num_layers=1,batch_first=True)
		elif model == 'RNN':
			self.rec = nn.RNN(embedding_dim, hidden_dim, num_layers=1,batch_first=True)
		else:
			assert()

		#self.hidden2out = MLP(hidden_dim, out_features=output_size)
		self.outputhead = nn.Sequential(
			nn.Linear(hidden_dim*2,hidden_dim),
			act_layer(),
			nn.Linear(hidden_dim,output_size)
										

		)
		#self.hidden2out = nn.Linear(hidden_dim,output_size)
		self.softmax = nn.LogSoftmax()
		self.encoder = MLP(feature_size,out_features=embedding_dim,act_layer=act_layer)
		#self.Mlp = (feature_size,embedding_dim)
		

		self.dropout_layer = nn.Dropout(p=0.5)


	def init_hidden(self, batch_size):
		return(autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)),
						autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)))


	def forward(self, batch):
		batch_size = batch.shape[0]
		
		self.hidden = self.init_hidden(batch.size(-1))

		embeds = self.encoder(batch)
		if self.model == 'LSTM':
		
			outputs, (ht, ct) = self.rec(embeds)
			ht = outputs[:,-1,:]
		else:
			outputs, ht = self.rec(embeds)

		# ht is the last hidden state of the sequences
		# ht = (1 x batch_size x hidden_dim)
		# ht[-1] = (batch_size x hidden_dim)

		#output = self.dropout_layer(ht[-1])
		output = self.dropout_layer(ht)
		output = self.outputhead(output)
		#output = self.softmax(output) #criterion에 softmax를 썼기 때문에 붙이면 안됨.

		return output   #output에 5차원으로 해서 라벨을 정수화한것과 비교...? 라벨이 5차원이되어야함