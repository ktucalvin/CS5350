import datasets
import numpy as np
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
from skorch import NeuralNetBinaryClassifier

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

# X = clf.fit_transform(X).toarray()
# Xtest = clf.fit_transform(Xtest).toarray()
# Y = Y.reshape((-1,1))

# class ILPData(Dataset):
#     def __init__(self):
#         super(ILPData, self).__init__()
        
#     def __getitem__(self, index):
#         return X[index,:], Y[index,:]
    
#     def __len__(self,):
#         # Return total number of samples.
#         return X.shape[0]

# train_loader = DataLoader(dataset=ILPData(), batch_size=1000, shuffle=True, drop_last=False)

# Construct model
EPOCHS = 10
model = NeuralNetworkClassifier(10, 15, 107, activation=nn.ReLU)
model.init_weights(strategy="he")

# Train model
net = NeuralNetBinaryClassifier(model, lr=0.001)#, criterion=nn.BCELoss)
clf.steps.append(("neural_net", net))
clf.fit(X, Y)
