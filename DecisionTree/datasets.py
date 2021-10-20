import numpy as np

def get_lms_data(dataset):
    """Prepares dataset where all values are numeric. Also appends a leading 1 to account for bias term."""
    Xtrain = []
    Ytrain = []
    with open(f"./datasets/{dataset}/train.csv") as file:
        for line in file:
            example = [1] + list(map(float, line.strip().split(',')))
            Xtrain.append(tuple(example[:-1]))
            Ytrain.append(example[-1])
    
    Xtest = []
    Ytest = []
    with open(f"./datasets/{dataset}/test.csv") as file:
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

def preprocess_bank_data(refill_unknown=False, numeric_labels=False):
    S = []
    with open("./datasets/bank/train.csv") as file:
        for line in file:
            S.append(tuple(line.strip().split(',')))

    val = []
    with open("./datasets/bank/test.csv") as file:
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
