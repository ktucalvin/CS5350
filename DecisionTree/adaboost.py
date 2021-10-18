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
    labels = [1, -1]

    # preprocessing:
    # Compute median for age, balance, day, duration, campaign, pdays, previous
    attr = list(attributes.keys())
    numeric_features = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
    media = []
    for feature in numeric_features:
        # exclude rows where pdays = -1 from pdays median
        if feature == "pdays":
            media.append(np.median([float(x[-4]) for x in S if x[-4] != "-1"]))
        else:
            media.append(np.median([float(x[attr.index(feature)]) for x in S]))

    # Replace numeric features with 1 or 0 if above or below median (or -1 if pdays)
    for index,example in enumerate(S):
        example = list(example)
        for media_index,feature in enumerate(numeric_features):
            feature_index = attr.index(feature)
            if not (feature == "pdays" and example[-4] == "-1"):
                example[feature_index] = "1" if float(example[feature_index]) > media[media_index] else "0"

        # Replace "no" with -1, and "yes" with 1
        if example[-1] == "yes":
            example[-1] = 1
        else:
            example[-1] = -1

        S[index] = tuple(example)

    for index,example in enumerate(val):
        example = list(example)
        for media_index,feature in enumerate(numeric_features):
            feature_index = attr.index(feature)
            if not (feature == "pdays" and example[-4] == "-1"):
                example[feature_index] = "1" if float(example[feature_index]) > media[media_index] else "0"

        # Replace "no" with -1, and "yes" with 1
        if example[-1] == "yes":
            example[-1] = 1
        else:
            example[-1] = -1

        val[index] = tuple(example)

    # If treating "unknown" as a value, return here
    if not refill_unknown:
        return S, val, attributes, labels

def compute_error(S, val, model, D):
    """
    Compute error, training accuracy, test accuracy of `model`
    Additionally returns the training predictions for computing D_{t+1}
    S - training dataset
    val - test dataset
    model - a classifier model, something with a `predict` method
    D - the weights
    """
    # Predict on training set
    training_predictions = np.array([model.predict(input) for input in S])
    correct_labels = np.array([x[-1] for x in S])
    num_correct = np.sum(training_predictions == correct_labels)
    train_acc = float(num_correct) / len(S)

    # eps_t = sum of weights of incorrectly labeled
    incorrect = training_predictions != correct_labels
    eps_t = np.sum(np.array(D)[incorrect])

    # Predict on test set
    test_predictions = [model.predict(input) for input in val]
    correct_labels = [x[-1] for x in val]
    num_correct = np.sum(np.array(test_predictions) == np.array(correct_labels))
    test_acc = float(num_correct) / len(val)

    return eps_t, train_acc, test_acc, training_predictions

# TODO: need to compute ERROR not ACCURACY
def compute_ensemble_error(S, val, ensemble):
    # Predict on training set
    training_predictions = np.array([ensemble_predict(ensemble, input) for input in S])
    correct_labels = np.array([x[-1] for x in S])
    num_correct = np.sum(training_predictions == correct_labels)
    train_acc = float(num_correct) / len(S)

    # Predict on test set
    test_predictions = np.array([ensemble_predict(ensemble, input) for input in val])
    correct_labels = np.array([x[-1] for x in val])
    num_correct = np.sum(test_predictions == correct_labels)
    test_acc = float(num_correct) / len(val)

    return train_acc, test_acc

def ensemble_predict(ensemble, input):
    total = 0
    for tup in ensemble:
        alpha, model = tup
        total += alpha * model.predict(input)

    if total > 0:
        return 1
    return -1


if __name__ == "__main__":
    S, val, attributes, labels = preprocess_bank_data()
    Y = np.array([example[-1] for example in S])
    print(Y)

    # initialize D_1(i) = 1/m for i = 1,2,...,m
    m =  len(S)
    D = np.ones(m) / m
    T = 500
    h = [] # ensemble is list of tuples of (weight, classifier)

    # print(f"(t, \\epsilon_t ?, training accuracy, test accuracy)")

    # for t = 1,2,...,T:
    for t in range(1, T+1):
        # Find a classifier h_t whose weighted classification error is better than chance
        model = ID3()
        model.train(S, attributes, labels, weights=D, max_depth=1)

        # epsilon_t = error of hypothesis on training data
        eps_t, train_acc, test_acc, predictions = compute_error(S, val, model, D)

        # compute its vote as alpha_t = 1/2 * ln(1-epsilon_t / epsilon_t)
        alpha_t = 0.5 * np.log((1 - eps_t) / eps_t)

        # d_{t+1}(i) = D_t(i) / Z_t * exp(-alpha_t * y_i h_t(x_i))
        Dnext = np.multiply(D, np.exp(-alpha_t * np.multiply(Y, predictions)))
        Dnext /= sum(Dnext)
        # Update the values of the weights for the training example
        D = Dnext

        print(f"stump> ({t}, {eps_t}, {train_acc}, {test_acc})")

        # debug prints
        # print(f"D = {D}")
        # print(f"alpha_t = {alpha_t}")

        h.append((alpha_t, model))
        train_acc, test_acc = compute_ensemble_error(S, val, h)
        print(f"ensemble> ({t}, {train_acc}, {test_acc})")

        # print()
        # print()

    # return the final hypothesis H_final(x) = sgn(sum_t alpha_t h_t(x))
    # ^ would need to refactor above into adaboost class with train/predict
