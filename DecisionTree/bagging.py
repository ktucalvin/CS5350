import id3
from id3 import ID3
import numpy as np


def preprocess_bank_data(refill_unknown=False):
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
        if example[-1] == "yes":
            example[-1] = 1
        else:
            example[-1] = -1

        val[index] = tuple(example)

    # If treating "unknown" as a value, return here
    if not refill_unknown:
        return S, val, attributes, labels

def bag_predict(bag, input):
    # works because y_i in {-1, 1}
    total = sum([model.predict(input) for model in bag])
    if total > 0:
        return 1
    return -1

def compute_bagged_error(S, val, bag):
    # Predict on training set
    training_predictions = np.array([bag_predict(bag, input) for input in S])
    correct_labels = np.array([x[-1] for x in S])
    num_incorrect = np.sum(training_predictions != correct_labels)
    train_err = float(num_incorrect) / len(S)

    # Predict on test set
    test_predictions = np.array([bag_predict(bag, input) for input in val])
    correct_labels = np.array([x[-1] for x in val])
    num_incorrect = np.sum(test_predictions != correct_labels)
    test_err = float(num_incorrect) / len(val)

    return train_err, test_err

if __name__ == "__main__":
    S, val, attributes, labels = preprocess_bank_data()
    m = len(S)
    T = 500

    for t in range(1, T+1):
        bag = []
        # for t = 1,2,...,T
        for _ in range(t): # train `t` trees for bag
            # draw m' <= m samples uniformly with replacement
            mprime = np.random.randint(1, m + 1)
            indices = np.random.choice(mprime, mprime)
            samples = [S[i] for i in indices]

            # train a classifier c_t
            model = ID3()
            model.train(samples, attributes, labels)
            bag.append(model)

        # construct final classifier by taking votes from all c_t
        train_err, test_err = compute_bagged_error(S, val, bag)
        print(f"{t}\t{train_err}\t{test_err}")
        

