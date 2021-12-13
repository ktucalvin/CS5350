import os
import torch
from torch.utils.data.dataset import Dataset
import datasets
import numpy as np
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from NeuralNetworks import torchnn


# :NOTE: Thanks to TA
raw_tr = np.loadtxt(os.path.join('datasets/bank-note', 'train.csv'), delimiter=',')
raw_te = np.loadtxt(os.path.join('datasets/bank-note', 'test.csv'), delimiter=',')

Xtr, ytr, Xte, yte = raw_tr[:,:-1], raw_tr[:,-1].reshape([-1,1]), raw_te[:,:-1], raw_te[:,-1].reshape([-1,1])    

class BankNote(Dataset):
    def __init__(self, data_path, mode):
        super(BankNote, self).__init__()
        raw_tr = np.loadtxt(os.path.join(data_path, 'train.csv'), delimiter=',')
        raw_te = np.loadtxt(os.path.join(data_path, 'test.csv'), delimiter=',')
        
        Xtr, ytr, Xte, yte = \
            raw_tr[:,:-1], raw_tr[:,-1].reshape([-1,1]), raw_te[:,:-1], raw_te[:,-1].reshape([-1,1])
        
        if mode == 'train':
            self.X, self.y = Xtr, ytr
        elif mode == 'test':
            self.X, self.y = Xte, yte
        else:
            raise Exception("Error: Invalid mode option!")
        
    def __getitem__(self, index):
        return self.X[index,:], self.y[index,:]
    
    def __len__(self,):
        # Return total number of samples.
        return self.X.shape[0]
    
    
dataset_train = BankNote('datasets/bank-note', mode='train')
dataset_test = BankNote('datasets/bank-note', mode='test')

train_loader = DataLoader(dataset=dataset_train, batch_size=1, shuffle=True, drop_last=False)
test_loader = DataLoader(dataset=dataset_test, batch_size=1, shuffle=True, drop_last=False)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if batch % 100 == 0:
        #     loss, current = loss.item(), batch * len(X)
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn, label):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    incorrect = 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            for i in range(len(pred)):
                if (y[i] < 0.5 and pred[i] > 0.5) or (y[i] > 0.5 and pred[i] < 0.5):
                    incorrect += 1

    test_loss /= num_batches
    print(f"{label} Error: {incorrect / size :>0.9f}, Avg loss: {test_loss:>8f}")

EPOCHS = 30

def eval_tanh_xavier(depth, width):
    print(f"=== Tanh+Xavier, depth={depth}, width={width}")
    model = torchnn.NeuralNetworkClassifier(depth, width, 4, activation=nn.Tanh)
    model.init_weights(strategy="xavier")
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.BCELoss()

    for t in range(EPOCHS):
        # print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_loader, model, loss_fn, optimizer)
    test_loop(train_loader, model, loss_fn, "Train")
    test_loop(test_loader, model, loss_fn, "Test")
    print()

def eval_relu_he(depth, width):
    print(f"=== ReLU+He, depth={depth}, width={width}")
    model = torchnn.NeuralNetworkClassifier(depth, width, 4, activation=nn.ReLU)
    model.init_weights(strategy="he")
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.BCELoss()

    for t in range(EPOCHS):
        # print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_loader, model, loss_fn, optimizer)
    test_loop(train_loader, model, loss_fn, "Train")
    test_loop(test_loader, model, loss_fn, "Test")
    print()

for depth in [3, 5, 9]:
    for width in [5, 10, 25, 50, 100]:
        eval_tanh_xavier(depth, width)
        eval_relu_he(depth, width)

