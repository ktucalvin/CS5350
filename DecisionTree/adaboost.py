import datasets
import id3
from id3 import ID3
import numpy as np
import math

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
    training_predictions = np.array([model.predict(input) for input in S]) # model is failing to predict anything
    correct_labels = np.array([x[-1] for x in S])
    num_incorrect = np.sum(training_predictions != correct_labels)
    train_err = float(num_incorrect) / len(S)

    # eps_t = sum of weights of incorrectly labeled
    incorrect = training_predictions != correct_labels
    eps_t = np.sum(np.array(D)[incorrect])

    # Predict on test set
    test_predictions = [model.predict(input) for input in val]
    correct_labels = [x[-1] for x in val]
    num_incorrect = np.sum(np.array(test_predictions) != np.array(correct_labels))
    test_err = float(num_incorrect) / len(val)

    return eps_t, train_err, test_err, training_predictions

def compute_ensemble_error(S, val, ensemble):
    # Predict on training set
    training_predictions = np.array([ensemble_predict(ensemble, input) for input in S])
    correct_labels = np.array([x[-1] for x in S])
    num_incorrect = np.sum(training_predictions != correct_labels)
    train_err = float(num_incorrect) / len(S)

    # Predict on test set
    test_predictions = np.array([ensemble_predict(ensemble, input) for input in val])
    correct_labels = np.array([x[-1] for x in val])
    num_incorrect = np.sum(test_predictions != correct_labels)
    test_err = float(num_incorrect) / len(val)

    return train_err, test_err

def ensemble_predict(ensemble, input):
    total = 0
    for tup in ensemble:
        alpha, model = tup
        total += alpha * model.predict(input)

    if total > 0:
        return 1
    return -1


if __name__ == "__main__":
    S, val, attributes, labels = datasets.preprocess_bank_data(numeric_labels=True)
    Y = np.array([example[-1] for example in S])
    with open("adaboost-results.txt", "a", encoding="utf8") as log:
        # initialize D_1(i) = 1/m for i = 1,2,...,m
        m =  len(S)
        D = np.ones(m) / m
        T = 500
        h = [] # ensemble is list of tuples of (weight, classifier)

        # for t = 1,2,...,T:
        for t in range(1, T+1):
            # Find a classifier h_t whose weighted classification error is better than chance

            model = ID3()
            model.train(S, attributes, labels, weights=D, max_depth=1, measure=id3.entropy)

            # epsilon_t = error of hypothesis on training data
            eps_t, train_err, test_err, predictions = compute_error(S, val, model, D)

            # compute its vote as alpha_t = 1/2 * ln(1-epsilon_t / epsilon_t)
            alpha_t = 0.5 * np.log((1 - eps_t) / eps_t)

            # d_{t+1}(i) = D_t(i) / Z_t * exp(-alpha_t * y_i h_t(x_i))
            Dnext = D * np.exp(-alpha_t * Y * predictions)
            Dnext /= sum(Dnext)
            # Update the values of the weights for the training example
            D = Dnext

            print(f"stump{t}\t\t{train_err}\t{test_err}\t{eps_t}")
            log.write(f"stump{t}\t\t{train_err}\t{test_err}\t{eps_t}\n")

            h.append((alpha_t, model))
            train_err, test_err = compute_ensemble_error(S, val, h)
            print(f"ensemble{t}\t{train_err}\t{test_err}")
            log.write(f"ensemble{t}\t{train_err}\t{test_err}\n")

        # return the final hypothesis H_final(x) = sgn(sum_t alpha_t h_t(x))
        # ^ would need to refactor above into adaboost class with train/predict
