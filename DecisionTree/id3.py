import copy
import numpy as np

def entropy(S, labels):
    # - sum_i^K p_i log_2(p_i)
    weightedS = [example[0] for example in S]
    total = 0
    for lab in labels:
        examples = [example[0] for example in S if example[-1] == lab]
        if sum(weightedS) and sum(examples):
            p_i = sum(examples) / sum(weightedS)
            total += p_i * np.log2(p_i)
    return -total


# :NOTE: does not use fractional weights
def majority_error(S, labels):
    if len(S) == 0:
        return 0
    # determine majority label
    example_labels = [x[-1] for x in S]
    uniq_labels, count = np.unique(example_labels, return_counts=True)
    maj_label = uniq_labels[np.argmax(count)]

    # return count(minority labels) / count(labels)
    x = np.array(example_labels) != np.array([maj_label])
    mismatchedWeights = [float(example[0]) for example in np.asarray(S)[x]]
    return sum(mismatchedWeights) / sum([example[0] for example in S])


# :NOTE: does not use fractional weights
def gini_index(S, labels):
    total = 0
    for lab in labels:
        examples = [example[0] for example in S if example[-1] == lab]
        if len(S) and len(examples):
            p_i = sum(examples) / len(S)
            total += p_i ** 2
    # return 1 - sum_1^K p_k^2
    return 1 - total


class ID3:
    def __init__(self):
        self.tree = {}

    def most_common_label(self, S):
        # must use FRACTIONAL counts
        counts = {}
        for example in S:
            label = example[-1]
            if label not in counts:
                counts[label] = example[0]
            else:
                counts[label] += example[0]
        return max(counts, key=counts.get)
        

    def gain(self, S, values, measure, labels, index):
        # measure(S) - sum_{v in values(A)} (len(S_v) / len(S) measure(S_v))
        total = 0
        for v in values:
            S_v = [example for example in S if example[index] == v]
            weightedS = [ex[0] for ex in S]
            weightedSv = [ex[0] for ex in S_v]
            total += sum(weightedSv) / sum(weightedS) * measure(S_v, labels)
        return measure(S, labels) - total

    def train(self, S, attributes, labels, weights=None, measure=entropy, max_depth=float("inf")):
        # attach weights to each example
        self.order = [None] + list(attributes.keys())
        if weights is None:
            weights = np.ones(len(S))
        weightedS = []
        for index,weight in enumerate(weights):
            weightedS.append((weight,) + S[index])

        self.tree = self.trainR(weightedS, attributes, labels, measure, max_depth)

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

    def predict(self, input):
        """
            Predict label given input tuple
            input -- input tuple
            order -- the order of attributes in the tuple
        """
        # tree is striped where odd layers are attribute names to split on
        # and even layers are dicts of attribute value to the next layer
        # traversal ends when we reach something that is not a dict

        subtree = self.tree
        while isinstance(subtree, dict):
            attr = list(subtree.keys())[0]
            subtree = subtree[attr][input[self.order.index(attr) - 1]]

        return subtree

