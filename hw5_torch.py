import torch
import datasets
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from NeuralNetworks import torchnn

Xtrain, Ytrain, Xtest, Ytest, attributes = datasets.get_bank_note_data()

Xtrain = torch.Tensor(Xtrain)
Ytrain = torch.Tensor(Ytrain)
Xtest = torch.Tensor(Xtest)
Ytest = torch.Tensor(Ytest)

# todo: what batch size?
training_dataset = TensorDataset(Xtrain, Ytrain)
training_loader = DataLoader(training_dataset, batch_size=64, shuffle=True)

# Set up model
test_dataset = TensorDataset(Xtest, Ytest)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

tanh_model = torchnn.NeuralNetworkClassifier(5, 5, Xtrain.shape[1], activation=nn.Tanh)
tanh_model.init_weights(strategy="xavier")
optimizer = torch.optim.Adam(tanh_model.parameters())
loss_fn = nn.CrossEntropyLoss() # binary cross-entropy

print(Xtrain)
print(tanh_model)

size = len(training_loader.dataset)
print(tanh_model(Xtest))
for batch, (X, y) in enumerate(training_loader):
    pred = tanh_model(X)
    loss = loss_fn(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if batch % 100 == 0:
        loss, current = loss.item(), batch * len(X)
        print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

print(tanh_model(Xtest))
