import datasets
from id3 import ID3
import numpy as np

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
    with open("bag-results.txt", "a", encoding="utf8") as log:
        S, val, attributes, labels = datasets.preprocess_bank_data(numeric_labels=True)
        m = len(S)
        T = 500

        bag = []
        for t in range(1, T+1):
            # for t = 1,2,...,T
            # can put another nested loop here if we want `t` new trees for each bagged model
            # instead of just adding one more tree each time

            # draw m' <= m samples uniformly with replacement
            mprime = np.random.randint(1, m + 1)
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
        

