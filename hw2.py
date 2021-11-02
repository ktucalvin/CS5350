import datasets
import numpy as np
from LinearRegression.lms import *
from EnsembleLearning.adaboost import AdaboostClassifier
from EnsembleLearning.bagging import BaggedClassifier
from EnsembleLearning.random_forests import RandomForestClassifier

# ==== Adaboost
def compute_ensemble_error(S, val, model):
    # Predict on training set
    training_predictions = np.array([model.predict(input) for input in S])
    correct_labels = np.array([x[-1] for x in S])
    num_incorrect = np.sum(training_predictions != correct_labels)
    train_err = float(num_incorrect) / len(S)

    # Predict on test set
    test_predictions = np.array([model.predict(input) for input in val])
    correct_labels = np.array([x[-1] for x in val])
    num_incorrect = np.sum(test_predictions != correct_labels)
    test_err = float(num_incorrect) / len(val)

    return train_err, test_err

# ==== Bagged Stumps / Random Forest
def compute_bias_variance(val, model):
    Y = [x[-1] for x in val]
    predictions = np.array([model.predict(input, type="average") for input in val])
    
    # bias = average prediction, minus ground-truth, then squared, then average over all examples
    bias = (predictions - Y) ** 2
    bias = np.average(predictions)

    # variance = 1/(n-1) sum(xi - m)^2 where m = sample mean
    m = np.mean(predictions)
    variance = 1/(len(val) - 1) * np.sum((predictions - m) ** 2)

    return bias, variance

def compute_bagged_error(S, val, model):
    # Predict on training set
    training_predictions = np.array([model.predict(input) for input in S])
    correct_labels = np.array([x[-1] for x in S])
    num_incorrect = np.sum(training_predictions != correct_labels)
    train_err = float(num_incorrect) / len(S)

    # Predict on test set
    test_predictions = np.array([model.predict(input) for input in val])
    correct_labels = np.array([x[-1] for x in val])
    num_incorrect = np.sum(test_predictions != correct_labels)
    test_err = float(num_incorrect) / len(val)

    return train_err, test_err

if __name__ == "__main__":
    print("The commit that is set up to produce the data for hw2 is 5a8cc31.")
    print("The commit that is set up to produce for the credit dataset is 16cddf7.")
    print("Since hw2 requires rigging each model to print data during training, this file is only for demonstration.")
    # ===== Ensemble Models Demonstration
    S, val, attributes, labels = datasets.preprocess_bank_data(numeric_labels=True)
    # S, val, attributes, labels = datasets.get_credit_data()
    ensemble_models = [(AdaboostClassifier(), "Adaboost"), (BaggedClassifier(), "Bagged"), (RandomForestClassifier(), "Random Forests")]
    for model, name in ensemble_models:
        model.train(S, attributes, labels, epochs=100)
        predictions = np.array([model.predict(input[:-1]) for input in val])
        correct_labels = np.array([x[-1] for x in val])
        num_incorrect = np.sum(predictions != correct_labels)
        test_err = float(num_incorrect) / len(val)
        print(f"{name} error: {test_err}")

    # ===== Gradient descent classifiers
    Xtrain, Ytrain, Xtest, Ytest, attributes = datasets.get_concrete_data()
    Xtrain = np.matrix(Xtrain)
    Xtest = np.matrix(Xtest)
    Ytrain = np.array(Ytrain)
    Ytest = np.array(Ytest)

    # this implementation will perform poorly because the learning rate isn't adjustable
    bgd = BatchGradientDescentClassifier(2 ** -8)
    bgd.train(Xtrain, Ytrain, threshold=1e-6)

    print(f"overall bgd loss on test = {loss(bgd.w, Xtest, Ytest)}")

    sgd = StochasticGradientDescentClassifier(0.001)
    sgd.train(Xtrain, Ytrain, threshold=1e-6)

    print(f"overall sgd loss on test = {loss(sgd.w, Xtest, Ytest)}")
