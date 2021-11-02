This is a machine learning library developed by Calvin Tu for CS5350/6350 at the University of Utah.

# Perceptron Models (Standard, Voted, Averaged)

First ensure all data is numeric. The `concrete` and `bank-note` datasets are set up for Perceptron learning.

Basic usage:

```python
import numpy as np
import datasets
from Perceptron import perceptron

Xtrain, Ytrain, Xtest, Ytest, attributes = datasets.get_bank_note_data()

def evaluate(model, variant):
    predictions = np.array([model.predict(x) for x in Xtest]).reshape(-1,1)
    incorrect = np.sum(predictions != Ytest)
    print(f"{variant} perceptron error: {float(incorrect) / len(Ytest) :.3f}")
    
def vec2str(vector):
    return f"({','.join([f'{x :.5f}' for x in vector])})"

standard_model = perceptron.StandardPerceptron(0.1)
standard_model.train(Xtrain, Ytrain, epochs=10)
evaluate(standard_model, "Standard")
print(vec2str(standard_model.w))

voted_model = perceptron.VotedPerceptron(0.1)
voted_model.train(Xtrain, Ytrain, epochs=10)
evaluate(voted_model, "Voted")
print(f"Got {len(voted_model.weights)} weights")
for w,c in voted_model.weights:
    print(f"{c}\t{vec2str(w)}")

averaged_model = perceptron.AveragedPerceptron(0.1)
averaged_model.train(Xtrain, Ytrain, epochs=10)
evaluate(averaged_model, "Averaged")
print(vec2str(averaged_model.a))
```

Each perceptron variant's constuctor takes in a float parameter, the `learning_rate`.

Training each perceptron is straightforward: `model.train(X, Y, epochs=10)`, where X is a numpy ndarray of training
data, Y is a numpy array of labels, and epochs is the number of training epochs.

Prediction is also simple: `model.predict(x)` where x is a numpy array of the same dimensionality as the training data.

The weight vector for StandardPerceptron can be inspected through `StandardPerceptron.w`.

The counts and weight vectors for VotedPerceptron can be inspected through `VotedPerceptron.weights`; it is a list of
tuples (weight_vector, count).

The running-averaged weight vector for AveragedPerceptron can be inspected throuhg `AveragedPerceptron.a`.

# LMS Models (Batch/Stochastic Gradient Descent)

First ensure all data is numeric. The `concrete` and `bank-note` datasets are set up for LMS learning.

Basic usage:

```python
import datasets
import numpy as np
from LinearRegression.lms import *
Xtrain, Ytrain, Xtest, Ytest, attributes = datasets.get_concrete_data()
Xtrain = np.matrix(Xtrain)
Xtest = np.matrix(Xtest)
Ytrain = np.array(Ytrain)
Ytest = np.array(Ytest)

bgd = BatchGradientDescentClassifier(2 ** -8)
bgd.train(Xtrain, Ytrain, threshold=1e-6)

print(f"overall bgd loss on test = {loss(bgd.w, Xtest, Ytest)}")

sgd = StochasticGradientDescentClassifier(0.001)
sgd.train(Xtrain, Ytrain, threshold=1e-6)

print(f"overall sgd loss on test = {loss(sgd.w, Xtest, Ytest)}")
```

Both constructors, BatchGradientDescentClassifier() and StochasticGradientDescentClassifier() take in a float parameter,
which is the learning rate.

Both models have the same signature for training: `model.train(X, Y, threshold)`. Training stops when the norm of the
difference between updated weight vectors `w_{t+1} - w_t` is below the given threshold. X should be a numpy matrix of
training data and Y should be a numpy array of labels.

Prediction is done through `model.predict(x)` where x is a numpy array of the same dimensionality as the training data.

The weight vector of each classifier can be inspected at any time through the `w` property.

# Ensemble models (Adaboost, Bagging, Random Forests)

First ensure that data labels are {-1, 1}. The `bank` and `credit` datasets are set up for ensemble learning.

Basic usage:

```python
import datasets
import numpy as np
from EnsembleLearning.adaboost import AdaboostClassifier
from EnsembleLearning.bagging import BaggedClassifier
from EnsembleLearning.random_forests import RandomForestClassifier

# These datasets are set up for ensemble learning
S, val, attributes, labels = datasets.preprocess_bank_data(numeric_labels=True)
# S, val, attributes, labels = datasets.get_credit_data()

# Where attributes and labels look like this:
attributes = {
        "age": ["0", "1"],
        "job": ["admin.", "unknown", "unemployed", "management", "housemaid", "entrepreneur",
                "student", "blue-collar", "self-employed", "retired", "technician", "services"],
        "marital": ["married", "divorced", "single"],
        "education": ["unknown", "secondary", "primary", "tertiary"],
        "default": ["yes", "no"],
        "balance": ["0", "1"],
        "housing": ["yes", "no"],
        "loan": ["yes", "no"],
        "contact": ["unknown", "telephone", "cellular"],
        "day": ["0", "1"],
        "month": ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
        "duration": ["0", "1"],
        "campaign": ["0", "1"],
        "pdays": ["-1", "0", "1"],
        "previous": ["0", "1"],
        "poutcome": ["unknown", "other", "failure", "success"]
    }

ensemble_models = [(AdaboostClassifier(), "Adaboost"), (BaggedClassifier(), "Bagged"), (RandomForestClassifier(), "Random Forests")]
for model, name in ensemble_models:
    model.train(S, attributes)
    predictions = np.array([model.predict(input[:-1]) for input in val])
    correct_labels = np.array([x[-1] for x in val])
    num_incorrect = np.sum(predictions != correct_labels)
    test_err = float(num_incorrect) / len(val)
    print(f"{name} error: {test_err}")
```

The ensemble models are all trained by calling: `model.train(data, attributes, epochs=10)` where attributes is a dict
describing how the data is formatted, each attribute and its values. The attributes must be defined in the same order
as the columns in the CSV.

`epochs` is an optional parameter that changes how many stumps each ensemble will train.

Random forests has an additional optional parameter, `subset_size`, which controls how many the size of the subset of
attributes it will use at each split when training its decision trees. If not set, then it will default to
`len(attributes.keys())`.

Adaboost's ensemble can be read through AdaboostClassifier.ensemble. This is a list of tuples of (weight, classifier).

Bagged/RandomForestClassifier's list of stumps/trees can be read through the `bag` property.

# Decision Trees

Basic usage:

```python
import id3
from id3 import ID3

# an example from the raw csv:
# high,med,5more,4,small,low,unacc

# Load data from csv into training and validation arrays
with open(f"./datasets/{dataset}/train.csv") as file:
    for line in file:
        S.append(tuple(line.strip().split(',')))

val = []
with open(f"./datasets/{dataset}/test.csv") as file:
    for line in file:
        val.append(tuple(line.strip().split(',')))

# Describe how the data is formatted
# attributes represents each column and its values
# these MUST be defined in the same order as columns in the CSV!
# not including the final label
attributes = {
    "buying":   ["vhigh", "high", "med", "low"],
    "maint":    ["vhigh", "high", "med", "low"],
    "doors":    ["2", "3", "4", "5more"],
    "persons":  ["2", "4", "more"],
    "lug_boot": ["small", "med", "big"],
    "safety":   ["low", "med", "high"]
}

# final labels are assumed to be the last column
labels = ["unacc", "acc", "good", "vgood"]

# instantiate id3 model
model = ID3()

# measure can be entropy, majority_error, or gini_index
measure = id3.entropy

# training examples can be fractional/weighted
# weights should be an array of numbers the same length as S
# if None, then weights will be set to np.ones(len(S)) (i.e. all ones)
weights = None

# maximum depth of decision tree can be set, but defaults to infinity otherwise
maxdep = 6

# train the model
model.train(S, attributes, labels, weights, measure, max_depth=maxdep)

# collect predictions from validation dataset
predictions = [model.predict(input) for input in val]
```

The model's tree can be inspected by `model.tree`. The tree is striped, where odd layers
are the names of attributes to split on and even layers are dicts of attribute value
to the next layer. Traversal ends when we reach something that is not a dict.

Once predictions are collected as above, accuracy can be computed fairly straightforwardly.

```python
predictions = [model.predict(input) for input in S]
correct_labels = [x[-1] for x in S]
correct_train = np.sum(np.array(predictions) == np.array(correct_labels))

predictions = [model.predict(input) for input in val]
correct_labels = [x[-1] for x in val]
correct_test = np.sum(np.array(predictions) == np.array(correct_labels))

print("\n Tree: ")
print(model.tree)
print("\n Predictions: ")
print(predictions)
print(
    f"Training Accuracy: ({correct_train}/{len(S)}) {float(correct_train) / len(S) :.3f} ")
print(
    f"Test Accuracy: ({correct_test}/{len(val)}) {float(correct_test) / len(val) :.3f} ")
```
