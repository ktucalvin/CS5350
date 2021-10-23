from id3 import ID3
import numpy as np
import datasets

if __name__ == "__main__":
    model = ID3()
    S, val, attributes, labels = datasets.get_credit_data()

    model.train(S, attributes, labels)

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

    print(f"Training error: {train_err}")
    print(f"Test error: {test_err}")
    