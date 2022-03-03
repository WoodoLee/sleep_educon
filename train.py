import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models as models
from tqdm import tqdm
import copy



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
                        if i % 1000 == 999:
                                copy_loader = copy.deepcopy(test_dataloader)
                                val = eval_net(model,copy_loader,device,break_idx = 1000)
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

