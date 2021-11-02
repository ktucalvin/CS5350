from pathlib import Path
import numpy as np

DATA_DIR = Path(__file__).parent

def get_lms_data(dataset):
    """Prepares dataset where all values are numeric. Also appends a leading 1 to account for bias term."""
    Xtrain = []
    Ytrain = []
    with (DATA_DIR / dataset / "train.csv").open() as file:
        for line in file:
            example = [1] + list(map(float, line.strip().split(',')))
            Xtrain.append(tuple(example[:-1]))
            Ytrain.append(example[-1])
    
    Xtest = []
    Ytest = []
    with (DATA_DIR / dataset / "test.csv").open() as file:
        for line in file:
            example = [1] + list(map(float, line.strip().split(',')))
            Xtest.append(tuple(example[:-1]))
            Ytest.append(example[-1])
    
    return Xtrain, Ytrain, Xtest, Ytest

def get_hw2_data():
    attributes = [
        "x1",
        "x2",
        "x3",
        "x4"
    ]
    
    return get_lms_data("hw2") + (attributes,)

def get_bank_note_data():
    attributes = [
        "variance",
        "skewness",
        "curtosis",
        "entropy"
    ]

    # Use numpy matrices instead of lists of lists
    Xtrain, Ytrain, Xtest, Ytest = get_lms_data("bank-note")
    Xtrain = np.array(Xtrain)
    Ytrain = np.array(Ytrain)
    Xtest = np.array(Xtest)
    Ytest = np.array(Ytest)

    # Use labels {-1, +1}
    Ytrain[Ytrain == 0] = -1
    Ytest[Ytest == 0] = -1

    Ytrain = Ytrain.reshape(-1,1)
    Ytest = Ytest.reshape(-1,1)

    return Xtrain, Ytrain, Xtest, Ytest, attributes

def get_concrete_data():
    attributes = [
        "Cement",
        "Slag",
        "Fly ash",
        "Water",
        "SP",
        "Coarse Aggr",
        "Fine Aggr",
        "SLUMP"
    ]
    
    return get_lms_data("concrete") + (attributes,)

def get_ilp_data(binarize=False):
    # :NOTE: test dataset does not have labels, but instead has additional "id" column
    attributes = {
        "age": [-1, 1], # continuous
        "workclass": ["?", "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"],
        "fnlwgt": [-1, 1], # continuous
        "education": ["?", "Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"],
        "education-num": [-1, 1], # continuous
        "marital-status": ["?", "Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"],
        "occupation": ["?", "Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"],
        "relationship": ["?", "Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"],
        "race": ["?", "White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"],
        "sex": ["?", "Female", "Male"],
        "capital-gain": [-1, 1], # continuous
        "capital-loss": [-1, 1], # continuous
        "hours-per-week": [-1, 1], # continuous
        "native-country": ["?", "United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"],
    }
    labels = [-1, 1]

    S = []
    with (DATA_DIR / "./ilp2021f/train_final.csv").open() as file:
        for line in file:
            example = line.strip().split(',')
            example[-1] = 1 if example[-1] == "1" else -1
            S.append(tuple(example))

    val = []
    with (DATA_DIR / "./ilp2021f/test_final.csv").open() as file:
        for line in file:
            val.append(tuple(line.strip().split(',')))
    
    # i.e. if using sklearn
    if not binarize:
        return S, val, attributes, labels

    continuous_features = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
    attr = list(attributes.keys())
    medians = []

    for feature in continuous_features:
        medians.append(np.median([float(x[attr.index(feature)]) for x in S]))

    # Replace numeric features with 1 or 0 if above or below median (or -1 if pdays)
    for index, example in enumerate(S):
        example = list(example)
        for median_index, feature in enumerate(continuous_features):
            feature_index = attr.index(feature)
            example[feature_index] = 1 if float(
                example[feature_index]) > medians[median_index] else -1
        
        S[index] = tuple(example)
    
    for index, example in enumerate(val):
        example = list(example)
        for median_index, feature in enumerate(continuous_features):
            feature_index = attr.index(feature) + 1 # shift for id column
            example[feature_index] = 1 if float(
                example[feature_index]) > medians[median_index] else -1
        
        val[index] = tuple(example)
    
    return S, val, attributes, labels

def get_credit_data():
    attributes = {
        "ID": [1, -1],
        "LIMIT_BAL": [1, -1],
        "SEX": [1, -1],
        "EDUCATION": [1, -1],
        "MARRIAGE": [1, -1],
        "AGE": [1, -1],
        "PAY_0": [1, -1],
        "PAY_2": [1, -1],
        "PAY_3": [1, -1],
        "PAY_4": [1, -1],
        "PAY_5": [1, -1],
        "PAY_6": [1, -1],
        "BILL_AMT1": [1, -1],
        "BILL_AMT2": [1, -1],
        "BILL_AMT3": [1, -1],
        "BILL_AMT4": [1, -1],
        "BILL_AMT5": [1, -1],
        "BILL_AMT6": [1, -1],
        "PAY_AMT1": [1, -1],
        "PAY_AMT2": [1, -1],
        "PAY_AMT3": [1, -1],
        "PAY_AMT4": [1, -1],
        "PAY_AMT5": [1, -1],
        "PAY_AMT6": [1, -1]
    }

    labels = [1, -1]

    with (DATA_DIR / "./credit/credit.csv").open() as file:
        X = []
        Y = []

        for line in file:
            example = [1] + list(map(float, line.strip().split(',')))[1:]
            X.append(tuple(example[:-1]))
            Y.append(1 if example[-1] == 1 else -1)
        
        averages = np.average(np.matrix(X), axis=0)
        for row,ex in enumerate(X):
            X[row] = np.ravel(np.array(np.array(ex) > averages, dtype=float))
        
        np.random.seed(1)
        train_indices = np.random.choice(len(X), 24000)
        np.random.seed(2)
        test_indices = np.random.choice(len(X), 6000)

        Xtrain = np.matrix(X)[train_indices]
        Ytrain = np.matrix(Y).T[train_indices]

        Xtest = np.matrix(X)[test_indices]
        Ytest = np.matrix(Y).T[test_indices]

        # adaboost needs labels to be -1,+1
        Ytrain[Ytrain == 0] = -1
        Ytest[Ytest == 0] = -1

        S = np.concatenate((Xtrain, Ytrain), axis=1)
        val = np.concatenate((Xtest, Ytest), axis=1)
        

        S = [tuple(example) for example in S.tolist()]
        val = [tuple(example) for example in val.tolist()]

        return S, val, attributes, labels

def preprocess_bank_data(refill_unknown=False, numeric_labels=False):
    S = []
    with (DATA_DIR / "./bank/train.csv").open() as file:
        for line in file:
            S.append(tuple(line.strip().split(',')))

    val = []
    with (DATA_DIR / "./bank/test.csv").open() as file:
        for line in file:
            val.append(tuple(line.strip().split(',')))

    # For numeric attributes, 1 = above median, 0 = below median
    attributes = {
        "age": ["0", "1"],
        "job": ["admin.", "unknown", "unemployed", "management", "housemaid", "entrepreneur",
                "student", "blue-collar", "self-employed", "retired", "technician", "services"],
        "marital": ["married", "divorced", "single"],
        "education": ["unknown", "secondary", "primary", "tertiary"],
        "default": ["yes", "no"],
        "balance": ["0", "1"],
        "housing": ["yes", "no"],
        "loan": ["yes", "no"],
        "contact": ["unknown", "telephone", "cellular"],
        "day": ["0", "1"],
        "month": ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
        "duration": ["0", "1"],
        "campaign": ["0", "1"],
        "pdays": ["-1", "0", "1"],
        "previous": ["0", "1"],
        "poutcome": ["unknown", "other", "failure", "success"]
    }
    labels = ["yes", "no"]

    if numeric_labels:
        labels = [1, -1]

    # preprocessing:
    # Compute median for age, balance, day, duration, campaign, pdays, previous
    attr = list(attributes.keys())
    numeric_features = ["age", "balance", "day",
                        "duration", "campaign", "pdays", "previous"]
    media = []
    for feature in numeric_features:
        # exclude rows where pdays = -1 from pdays median
        if feature == "pdays":
            media.append(np.median([float(x[-4]) for x in S if x[-4] != "-1"]))
        else:
            media.append(np.median([float(x[attr.index(feature)]) for x in S]))

    # Replace numeric features with 1 or 0 if above or below median (or -1 if pdays)
    for index, example in enumerate(S):
        example = list(example)
        for media_index, feature in enumerate(numeric_features):
            feature_index = attr.index(feature)
            if not (feature == "pdays" and example[-4] == "-1"):
                example[feature_index] = "1" if float(
                    example[feature_index]) > media[media_index] else "0"

        # Replace "no" with -1, and "yes" with 1
        if numeric_labels:
            if example[-1] == "yes":
                example[-1] = 1
            else:
                example[-1] = -1

        S[index] = tuple(example)

    for index, example in enumerate(val):
        example = list(example)
        for media_index, feature in enumerate(numeric_features):
            feature_index = attr.index(feature)
            if not (feature == "pdays" and example[-4] == "-1"):
                example[feature_index] = "1" if float(
                    example[feature_index]) > media[media_index] else "0"

        # Replace "no" with -1, and "yes" with 1
        if numeric_labels:
            if example[-1] == "yes":
                example[-1] = 1
            else:
                example[-1] = -1

        val[index] = tuple(example)

    # If treating "unknown" as a value, return here
    if not refill_unknown:
        return S, val, attributes, labels

    # Determine most common label for these features
    unknown_features = ["job", "education", "contact", "poutcome"]
    common_labels = []
    for feature in unknown_features:
        uniq_labels, count = np.unique([x[attr.index(feature)] for x in S if x[attr.index(feature)] != "unknown"], return_counts=True)
        common_labels.append(uniq_labels[np.argmax(count)])
    
    # For each example, replace each unknown feature with the most common label for that feature
    for index,example in enumerate(S):
        example = list(example)
        for label_index,feature in enumerate(unknown_features):
            feature_index = attr.index(feature)
            example[feature_index] = common_labels[label_index]
        S[index] = tuple(example)

    for index,example in enumerate(val):
        example = list(example)
        for label_index,feature in enumerate(unknown_features):
            feature_index = attr.index(feature)
            example[feature_index] = common_labels[label_index]
        val[index] = tuple(example)
    
    return S, val, attributes, labels
