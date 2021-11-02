# Sibling imports https://stackoverflow.com/questions/6323860/sibling-package-imports
import sys
import os.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

from DecisionTree.id3 import ID3
import numpy as np

class BaggedClassifier:
    """Bagged classifier based on id3 decision stumps."""
    def __init__(self):
        self.bag = []
    
    def train(self, S, attributes, epochs=10):
        m = len(S)
        for t in range(1, epochs+1):
            # for t = 1,2,...,T
            # can put another nested loop here if we want `t` new trees for each bagged model
            # instead of just adding one more tree each time

            # draw m' <= m samples uniformly with replacement
            mprime = np.random.randint(1, m+1)
            indices = np.random.choice(len(S), mprime)
            samples = [S[i] for i in indices]

            # train a classifier c_t
            model = ID3()
            model.train(samples, attributes, [-1, 1])
            self.bag.append(model)

    def predict(self, x, type="majority"):
        # works because y_i in {-1, +1}, just returns majority vote
        if type == "majority":
            total = sum([model.predict(x) for model in self.bag])
            if total > 0:
                return 1
            return -1
        elif type == "average":
            return np.average([model.predict(x) for model in self.bag])
