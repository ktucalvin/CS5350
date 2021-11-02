# Sibling imports https://stackoverflow.com/questions/6323860/sibling-package-imports
import sys
import os.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

from DecisionTree.id3 import ID3
import copy
import numpy as np

class RandomID3(ID3):
    """Same as ID3, except selects a random subset of available attributes before each split."""
    def __init__(self, subset_size):
        self.tree = {}
        self.subset_size = subset_size

    def trainR(self, S, attributes, labels, measure, max_depth, depth=0):
        """Internal recursive training function"""
        # if all examples have same label
        #   return a leaf node with that label
        if len(set([x[-1] for x in S])) == 1:
            return S[0][-1]

        # if attributes is empty OR exceeded depth
        #   return a leaf node with the most common label
        if len(attributes.keys()) == 0 or depth >= max_depth:
            return self.most_common_label(S)

        # create root node for tree
        root = {}
        # A = attribute in attributes that best splits S
        attr = list(attributes.keys())

        # take random subset
        if len(attr) >= self.subset_size:
            subset_indices = np.random.choice(len(attr), self.subset_size, replace=False)
            attr = [attr[i] for i in subset_indices]

        gains = [self.gain(S, attributes[a], measure, labels,
                           self.order.index(a)) for a in attr]
        A = attr[np.argmax(gains)]
        indexA = self.order.index(A)

        # record in the tree which attribute we split on, for later traversal
        root[A] = {}

        # for each value v that A can take:
        for v in attributes[A]:
            #   add a new tree branch corresponding to A=v
            root[A][v] = {}
        #   let S_v be the subset of examples in S with A = v
            S_v = [example for example in S if example[indexA] == v]
        #   if S_v is empty:
        #       add leaf node with the most common label in S
            if len(S_v) == 0:
                root[A][v] = self.most_common_label(S)
        #   else:
        #       below this branch add the subtree ID3(S_v, attributes - {A}, label)
            else:
                newattr = copy.deepcopy(attributes)
                del newattr[A]
                root[A][v] = self.trainR(S_v, newattr, labels, measure, depth=(
                    depth + 1), max_depth=max_depth)
        # return root node
        return root

class RandomForestClassifier:
    """Random forest classifier based on id3 decision trees that use a random subset of attributes at each split."""
    def __init__(self):
        self.bag = []

    def train(self, S, attributes, epochs=10, subset_size=None):
        if not subset_size:
            subset_size = len(attributes.keys())

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
            model = RandomID3(subset_size)
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
