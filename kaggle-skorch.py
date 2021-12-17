import datasets
import numpy as np
from torch import nn
from NeuralNetworks.torchnn import NeuralNetworkClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from skorch import NeuralNetBinaryClassifier, NeuralNetClassifier

# Setup dataset
S, val, attributes, labels = datasets.get_ilp_data(binarize=True)

X = np.asarray(S).reshape((25000,15))
Y = X[:, -1].astype(np.float64)
X = X[:, :-1]

Xtest = np.asarray(val).reshape((23842,15))
Xtest = Xtest[:, 1:]

# To be able to impute missing values
X[X == "?"] = np.nan
Xtest[Xtest == "?"] = np.nan

# Setup neural network

# after one-hot encoding, dimensionality is 107
# module = NeuralNetworkClassifier(10, 25, 107, activation=nn.ReLU)
# module.init_weights(strategy="he")

# net = NeuralNetBinaryClassifier(module)
import numpy as np
from sklearn.datasets import make_classification
from torch import nn
import torch.nn.functional as F

from skorch import NeuralNetClassifier
class MyModule(nn.Module):
    def __init__(self, num_units=10, nonlin=F.relu):
        super(MyModule, self).__init__()

        self.dense0 = nn.Linear(107, num_units)
        self.nonlin = nonlin
        self.dropout = nn.Dropout(0.5)
        self.dense1 = nn.Linear(num_units, 10)
        self.output = nn.Linear(10, 2)

        self.double()

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = self.dropout(X)
        X = F.relu(self.dense1(X))
        X = F.softmax(self.output(X))
        return X


net = NeuralNetClassifier(
    MyModule,
    max_epochs=10,
    lr=0.1,
    # Shuffle training data on each epoch
    iterator_train__shuffle=True,
)

# Setup pipeline
attr = list(attributes.keys())
numeric_features = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
numeric_features = [attr.index(feat) for feat in numeric_features]
categorical_features = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
categorical_features = [attr.index(feat) for feat in categorical_features]

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
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
    steps=[
        ("preprocessor", preprocessor),
        ("net", net)
    ]
)

clf.fit(X, Y)

print(f"Training Accuracy: {clf.score(X, Y) :.3f}")

test_predictions = clf.predict(Xtest)
print(test_predictions)

with open("skorch.csv", "w+", encoding="utf8") as log:
    log.write("ID,Prediction\n")
    id =1
    for pred in test_predictions:
        log.write(f"{id},{pred}\n")
        if not id % 1000:
            print(f"{id},{pred}")
        id += 1
