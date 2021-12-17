import datasets
import numpy as np
import torch
from torch import nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import TensorDataset, DataLoader
from NeuralNetworks.torchnn import NeuralNetworkClassifier
from DecisionTree.id3 import ID3
from LogisticRegression import logreg
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

# Setup data
S, val, attributes, labels = datasets.get_ilp_data(binarize=True)

del val[17304] # Holand-Netherlands only exists in test data

X = np.asarray(S).reshape((25000,15))
Y = X[:, -1].astype(np.float64)
X = X[:, :-1]

Xtest = np.asarray(val).reshape((23841,15))
ids = Xtest[:, 0]
Xtest = Xtest[:, 1:]

# To be able to impute missing values
X[X == "?"] = np.nan
Xtest[Xtest == "?"] = np.nan

attr = list(attributes.keys())
numeric_features = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
numeric_features = [attr.index(feat) for feat in numeric_features]
categorical_features = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
categorical_features = [attr.index(feat) for feat in categorical_features]

numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

categorical_transformer = Pipeline(
    steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ("imputer", SimpleImputer(strategy="most_frequent", missing_values=np.nan)),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

clf = Pipeline(
    steps=[("preprocessor", preprocessor)]
)

X = clf.fit_transform(X).toarray()
Xtest = clf.fit_transform(Xtest).toarray()
Y = Y.reshape((-1,1))

class ILPData(Dataset):
    def __init__(self):
        super(ILPData, self).__init__()
        
    def __getitem__(self, index):
        return X[index,:], Y[index,:]
    
    def __len__(self,):
        # Return total number of samples.
        return X.shape[0]

train_loader = DataLoader(dataset=ILPData(), batch_size=64, shuffle=True, drop_last=False)

# Construct model
EPOCHS = 10
model = NeuralNetworkClassifier(10, 15, 107, activation=nn.ReLU)
model.init_weights(strategy="he")


# Train model
optimizer = torch.optim.Adam(model.parameters())
loss_fn = nn.BCELoss()

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

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


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

for t in range(EPOCHS):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_loader, model, loss_fn, optimizer)
    # test_loop(test_loader, model, loss_fn, "Test")
    print()

predictions = model(torch.from_numpy(Xtest))
with open("kaggle-pytorch.csv", "w+", encoding="utf8") as log:
    log.write("ID,Prediction\n")
    for row_id, pred in zip(ids, predictions):
        log.write(f"{row_id},{pred.item()}\n")
        print(f"{row_id},{pred.item()}")
