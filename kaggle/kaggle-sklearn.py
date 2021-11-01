import datasets
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

S, val, attributes, labels = datasets.get_ilp_data(binarize=True)

X = np.asarray(S).reshape((25000,15))
Y = X[:, -1]
X = X[:, :-1]

Xtest = np.asarray(val).reshape((23842,15))
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
    steps=[("preprocessor", preprocessor), ("classifier", MLPClassifier(max_iter=5000, early_stopping=True))]
)

clf.fit(X, Y)

print(f"Training Accuracy: {clf.score(X, Y) :.3f}")

test_predictions = clf.predict(Xtest)
print(test_predictions)

test_predictions = ["0" if pred == "-1" else pred for pred in test_predictions]
with open("sklearn-neural-imputed-earlystop.csv", "a+", encoding="utf8") as log:
    log.write("ID,Prediction\n")
    id =1
    for pred in test_predictions:
        log.write(f"{id},{pred}\n")
        if not id % 1000:
            print(f"{id},{pred}")
        id += 1
