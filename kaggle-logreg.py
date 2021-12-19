import datasets
import numpy as np
from DecisionTree.id3 import ID3
from LogisticRegression import logreg
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

S, val, attributes, labels = datasets.get_ilp_data(binarize=True)

del val[17304] # Holand-Netherlands only exists in test data

X = np.asarray(S).reshape((25000,15))
Y = X[:, -1].astype(np.float64)
X = X[:, :-1]

Xtest = np.asarray(val).reshape((23841,15))
ids = Xtest[:, 0].astype(np.int64)
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

gamma_0 = 0.0001
d = 500
v = 2
# model = logreg.MLEClassifier(schedule=lambda t: gamma_0 / (1 + gamma_0/d * t))
model = logreg.MAPClassifier(schedule=lambda t: gamma_0 / (1 + gamma_0/d * t), variance=v)

model.train(X, Y, threshold=1e-5)

predictions = [model.predict(input) for input in Xtest]
with open(f"kaggle-logreg-v{v}.csv", "w+", encoding="utf8") as log:
    log.write("ID,Prediction\n")
    for row_id, pred in zip(ids, predictions):
        log.write(f"{row_id},{pred}\n")
        if not row_id % 1000:
            print(f"{row_id},{pred}")

        # Row 17305 has value nonexistant in training data, interferes with one-hot encoding, "holand"
        if row_id == 17304:
            log.write("17305,0\n")
            row_id += 1
