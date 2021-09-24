from sklearn import tree
from sklearn import preprocessing
import numpy as np

if __name__ == "__main__":
    X = []
    Y = []
    valX = []
    valY = []
    with open("./datasets/car/train.csv") as file:
        for line in file:
            example = list(line.strip().split(','))
            X.append(example[:-1])
            Y.append(example[-1])

    with open("./datasets/car/test.csv") as file:
        for line in file:
            example = list(line.strip().split(','))
            valX.append(example[:-1])
            valY.append(example[-1])
    
    X = np.asarray(X).reshape((1000,6))
    Y = np.asarray(Y).reshape((1000,))
    valX = np.asarray(valX).reshape((728,6))
    valY = np.asarray(valY).reshape((728,))
    le = preprocessing.LabelEncoder()
    for i in range(6):
        X[:,i] = le.fit_transform(X[:,i])
        valX[:,i] = le.fit_transform(valX[:,i])

    model = tree.DecisionTreeClassifier(max_depth=1e6)
    model = model.fit(X, le.fit_transform(Y))

    training_predictions = le.inverse_transform(model.predict(X))
    test_predictions = le.inverse_transform(model.predict(valX))

    # print(predictions)

    correct_train = np.sum(np.array(training_predictions) == Y)
    correct_test = np.sum(np.array(test_predictions) == valY)
    print(f"Training accuracy: ({correct_train}/{len(Y)}) {float(correct_train) / len(Y) :.3f} ")
    print(f"Test accuracy: ({correct_test}/{len(valY)}) {float(correct_test) / len(valY) :.3f} ")
