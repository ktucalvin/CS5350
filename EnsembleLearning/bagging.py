# Sibling imports https://stackoverflow.com/questions/6323860/sibling-package-imports
import sys
import os.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

import datasets
from DecisionTree.id3 import ID3
import numpy as np

def bag_predict_majority(bag, input):
    # works because y_i in {-1, +1}, just returns majority vote
    total = sum([model.predict(input) for model in bag])
    if total > 0:
        return 1
    return -1

def bag_predict_average(bag, input):
    # returns average prediction, which might not be in {-1, +1}
    return np.average([model.predict(input) for model in bag])

def compute_bias_variance(val, bag):
    Y = [x[-1] for x in val]
    predictions = np.array([bag_predict_average(bag, input) for input in val])
    
    # bias = average prediction, minus ground-truth, then squared, then average over all examples
    bias = (predictions - Y) ** 2
    bias = np.average(predictions)

    # variance = 1/(n-1) sum(xi - m)^2 where m = sample mean
    m = np.mean(predictions)
    variance = 1/(len(val) - 1) * np.sum((predictions - m) ** 2)

    return bias, variance

def compute_bagged_error(S, val, bag):
    # Predict on training set
    training_predictions = np.array([bag_predict_majority(bag, input) for input in S])
    correct_labels = np.array([x[-1] for x in S])
    num_incorrect = np.sum(training_predictions != correct_labels)
    train_err = float(num_incorrect) / len(S)

    # Predict on test set
    test_predictions = np.array([bag_predict_majority(bag, input) for input in val])
    correct_labels = np.array([x[-1] for x in val])
    num_incorrect = np.sum(test_predictions != correct_labels)
    test_err = float(num_incorrect) / len(val)

    return train_err, test_err

if __name__ == "__main__":
    S, val, attributes, labels = datasets.get_credit_data()
    with open("REVERSE-credit-bag-results.txt", "a", encoding="utf8") as log:
        m = len(S)
        T = 500

        bag = []
        for t in range(1, T+1):
            # for t = 1,2,...,T
            # can put another nested loop here if we want `t` new trees for each bagged model
            # instead of just adding one more tree each time

            # draw m' <= m samples uniformly with replacement
            mprime = np.random.randint(1, m+1)
            indices = np.random.choice(len(S), mprime)
            samples = [S[i] for i in indices]

            # train a classifier c_t
            model = ID3()
            model.train(samples, attributes, labels)
            bag.append(model)

            # construct final classifier by taking votes from all c_t
            train_err, test_err = compute_bagged_error(S, val, bag)
            print(f"{t}\t{train_err}\t{test_err}")
            log.write(f"{t}\t{train_err}\t{test_err}\n")
    exit()
    # part 2, q2c
    with open("bias-variance-bags-results.txt", "a", encoding="utf8") as log:
        predictors = [] # no BaggedPredictor class, only need bag_predict and list of bagged classifiers

        # repeat 100 times:
        for i in range(100):
            print(f"Training the {i}-th predictor")
            bag = []

            # sample 1000 examples uniformly WITHOUT replacement
            indices = np.random.choice(len(S), 1000, replace=False)
            samples = [S[i] for i in indices]

            # learn bagged predictor w/ 500 (or 100 if time constrained) trees
            for ti in range(500):
                # draw m' <= m samples uniformly with replacement
                indices = np.random.choice(len(samples), m)
                subsamples = [samples[i] for i in indices]

                model = ID3()
                model.train(subsamples, attributes, labels)
                bag.append(model)
                print(f"trained tree #{ti}")

            predictors.append(bag)
    
        # predictors is now a list of 100 lists, with each of those lists containing 500 ID3 trees
        print("(bias, variance)")
        for index,bag in enumerate(predictors):
            # compute variance and bias from 1st tree of the 100
            tree = bag[0]
            bias, variance = compute_bias_variance(val, [tree])
            print(f"tree{index}\t{bias}\t{variance}")
            log.write(f"tree{index}\t{bias}\t{variance}\n")

            # compute variance and bias from bagged predictor
            bias, variance = compute_bias_variance(val, bag)
            print(f"bag{index}\t{bias}\t{variance}")
            log.write(f"bag{index}\t{bias}\t{variance}\n")

        

