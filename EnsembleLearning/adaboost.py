# Sibling imports https://stackoverflow.com/questions/6323860/sibling-package-imports
import sys
import os.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

from DecisionTree import id3
from DecisionTree.id3 import ID3
import numpy as np

class AdaboostClassifier:
    """Adaboost classifier based on id3 decision stumps."""
    def __init__(self):
        self.ensemble = [] # ensemble is list of tuples of (weight, classifier)
    
    def train(self, S, attributes, epochs=10):
        # initialize D_1(i) = 1/m for i = 1,2,...,m
        m =  len(S)
        D = np.ones(m) / m
        Y = np.array([example[-1] for example in S])

        # for t = 1,2,...,T:
        for t in range(1, epochs+1):
            # Find a classifier h_t whose weighted classification error is better than chance
            model = ID3()
            model.train(S, attributes, [-1, 1], weights=D, max_depth=1, measure=id3.entropy)

            # epsilon_t = error of hypothesis on training data
            # eps_t = sum of weights of incorrectly labeled
            training_predictions = np.array([model.predict(input) for input in S])
            correct_labels = np.array([x[-1] for x in S])
            incorrect = training_predictions != correct_labels
            eps_t = np.sum(np.array(D)[incorrect])

            # compute its vote as alpha_t = 1/2 * ln(1-epsilon_t / epsilon_t)
            alpha_t = 0.5 * np.log((1 - eps_t) / eps_t)

            # d_{t+1}(i) = D_t(i) / Z_t * exp(-alpha_t * y_i h_t(x_i))
            Dnext = D * np.exp(-alpha_t * Y * training_predictions)
            Dnext /= sum(Dnext)
            # Update the values of the weights for the training example
            D = Dnext

            # print(f"stump{t}\t\t{train_err}\t{test_err}\t{eps_t}")

            self.ensemble.append((alpha_t, model))

    def predict(self, x):
        # prediction = H_final(x) = sgn(sum_t alpha_t h_t(x))
        total = 0
        for tup in self.ensemble:
            alpha, model = tup
            total += alpha * model.predict(x)

        if total > 0:
            return 1
        return -1
