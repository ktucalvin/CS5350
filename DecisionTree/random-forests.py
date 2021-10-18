from id3 import ID3
import datasets
import copy
import numpy as np

class RandomID3(ID3):
    def __init__(self, subset_size):
        self.tree = {}
        self.subset_size = subset_size

    """Same as ID3, except selects a random subset of available attributes before each split."""
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
        if len(attr) >= subset_size:
            subset_indices = np.random.choice(len(attr), subset_size, replace=False)
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
    with open("random-forests-results.txt", "a", encoding="utf8") as log:
        S, val, attributes, labels = datasets.preprocess_bank_data(numeric_labels=True)
        m = len(S)
        T = 500

        for subset_size in [2, 4, 6]:
            bag = []
            print(f"Subset size = {subset_size}")
            log.write(f"\nSubset size = {subset_size}\n")
            for t in range(1, T+1):
                # for t = 1,2,...,T
                # can put another nested loop here if we want `t` new trees for each bagged model
                # instead of just adding one more tree each time

                # draw m' <= m samples uniformly with replacement
                mprime = np.random.randint(1, m + 1)
                indices = np.random.choice(len(S), mprime)
                samples = [S[i] for i in indices]

                # train a classifier c_t
                model = RandomID3(subset_size)
                model.train(samples, attributes, labels)
                bag.append(model)

                # construct final classifier by taking votes from all c_t
                train_err, test_err = compute_bagged_error(S, val, bag)
                print(f"{t}\t{train_err}\t{test_err}")
                log.write(f"{t}\t{train_err}\t{test_err}\n")
