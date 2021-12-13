This is a machine learning library developed by Calvin Tu for CS5350/6350 at the University of Utah.

## Logistic Regression

First ensure all data is numeric. The `concrete` and `bank-note` datasets are set up for LMS learning.

Basic usage;
```python
import numpy as np
import datasets
from LogisticRegression import logreg


Xtrain, Ytrain, Xtest, Ytest, attributes = datasets.get_bank_note_data()

def evaluate(model, label):
    predictions = np.array([model.predict(x) for x in Xtrain]).reshape(-1,1)
    incorrect_train = np.sum(predictions != Ytrain)
    predictions = np.array([model.predict(x) for x in Xtest]).reshape(-1,1)
    incorrect_test = np.sum(predictions != Ytest)
    print(f"{label}\nTrain Error: {float(incorrect_train) / len(Ytrain) :.7f}\nTest error: {float(incorrect_test) / len(Ytest) :.7f}")

gamma_0 = 0.01
d = 500
for v in [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]:
    map_model = logreg.MAPClassifier(schedule=lambda t: gamma_0 / (1 + gamma_0/d * t), variance=v)
    map_model.train(Xtrain, Ytrain, threshold=1e-6)
    evaluate(map_model, f"MAP Model, v = {v}")
    print()

# train MLE model, does not depend on variance
mle_model = logreg.MLEClassifier(schedule=lambda t: gamma_0 / (1 + gamma_0/d * t))
mle_model.train(Xtrain, Ytrain, threshold=1e-6)
evaluate(mle_model, "MLE Model")
```


For the MLEClassifer, usage is exactly the same as the LMS models.

For the MAPClassifier, usage is the same as the LMS models, except the constructor takes in a `variance` parameter. This
controls the balance between the likelihood and the prior when performing logistic regression. The prior is always
assumed to be from a gaussian distribution.

The training/prediction labels are {-1, +1}

## Neural Networks

First ensure all data is numeric. The `concrete` and `bank-note` datasets are set up for SVM learning.

Basic usage:
```python
import numpy as np
import datasets
from NeuralNetworks import nn

Xtrain, Ytrain, Xtest, Ytest, attributes = datasets.get_bank_note_data()

Ytrain[Ytrain == -1] = 0
Ytest[Ytest == -1] = 0

def evaluate(model, label):
    predictions = np.array([model.predict(x) for x in Xtrain]).reshape(-1,1)
    incorrect_train = np.sum(predictions != Ytrain)
    predictions = np.array([model.predict(x) for x in Xtest]).reshape(-1,1)
    incorrect_test = np.sum(predictions != Ytest)
    print(predictions)
    print(f"{label}\nTrain Error: {float(incorrect_train) / len(Ytrain) :.5f}\nTest error: {float(incorrect_test) / len(Ytest) :.5f}")

gamma_0 = 150
d = 1e-2
epochs = 1000
print(f"gamma_0={gamma_0}, d={d}, epochs={epochs}")

gauss_model = nn.NeuralNetworkClassifier(width, Xtrain.shape[1],
                                            schedule=lambda t: gamma_0 / (1 + gamma_0/d * t),
                                            get_weight=nn.WeightInitializer.gaussian)
gauss_model.train(Xtrain, Ytrain, epochs)

evaluate(gauss_model, f"NN, Gaussian weights, width={width}")
print()

zero_model = nn.NeuralNetworkClassifier(width, Xtrain.shape[1],
                                            schedule=lambda t: gamma_0 / (1 + gamma_0/d * t),
                                            get_weight=nn.WeightInitializer.gaussian)
zero_model.train(Xtrain, Ytrain, epochs)
evaluate(zero_model, f"NN, Zero weights, width={width}")
print()
```

Unfortunately, the neural network implementation does not work for more than one training example.

The constructor takes in these parameters:
* `layer_width`: the width of hidden layers 1 and 2.
* `dims`: the dimensionality of the training data (e.g. `Xtrain.shape[1]`).
* `schedule`: a function that takes in a time step `t` and returns the learning rate at that time step.
* `get_weight`: a function that returns a float. Used to initialize the weights of the network.

Training and prediction is the same as for SVMs. After training is complete, a plot of the loss history as well as the
weights value history will appear. If `train()` is called at least once, the gradients of the last epoch are available
for inspection through the `last_grad` property.

Keys of the form `n-m` represent the gradient of the node in layer `n` with id `m`. Keys of the form `L_n-m` represent
the gradient of the weight to layer `L` from the node with id `n` to the node with id `m`.

The graph is accessed through the `network` field. From here, you can access the various nodes of the network through
the `Node.inputs` and `Node.outputs` fields.


A PyTorch implementation is available through `torchnn.NeuralNetworkClassifier`. It works the same as any other PyTorch
model. The constructor takes the following:
* `depth`: Number of hidden layers
* `width`: Width of each hidden layer
* `dims`: Dimensionality of the input
* `activation`: What activation to use between hidden layers. The final layer is always `nn.Sigmoid`.

The weights of each linear hidden layer is initialized by calling `init_weights` with either the "he" or "xavier"
strategies.

## Support Vector Machines
First ensure all data is numeric. The `concrete` and `bank-note` datasets are set up for SVM learning. All data should
be augmented with a leading 1, to account for the bias; the Dual SVM will automatically strip it off in its handling of
the bias though.

Basic usage:
```python
import numpy as np
import datasets
from SVM import svm

Xtrain, Ytrain, Xtest, Ytest, attributes = datasets.get_bank_note_data()

def evaluate(model, variant):
    predictions = np.array([model.predict(x) for x in Xtrain]).reshape(-1,1)
    incorrect_train = np.sum(predictions != Ytrain)
    predictions = np.array([model.predict(x) for x in Xtest]).reshape(-1,1)
    incorrect_test = np.sum(predictions != Ytest)
    print(f"SVM, {variant}\nTest error: {float(incorrect_test) / len(Ytest) :.5f}\nTrain Error: {float(incorrect_train) / len(Ytrain) :.5f}")
    
def vec2str(vector):
    # augmented vector
    return f"w' = ({','.join([f'{x :.5f}' for x in vector])})"

gamma_0 = 1e-4

primal_simple_model = svm.PrimalSVMClassifier(C=100/873, schedule=lambda t: gamma_0 / (1 + t))
primal_simple_model.train(Xtrain, Ytrain, epochs=100)

evaluate(primal_simple_model, f"Primal simple schedule, C={c}")
print(vec2str(primal_simple_model.w))

dual_linear_model = svm.DualSVMClassifier(C=c, tol=1e-10, kernel="linear")
dual_linear_model.train(Xtrain, Ytrain)
evaluate(dual_linear_model, f"Dual linear kernel, C={c}")
print(f"Support vectors: {len(dual_linear_model.support_vectors)}")
print(vec2str(dual_linear_model.wstar))
print(f"b* = {dual_linear_model.bstar}")

dual_gaussian_model = svm.DualSVMClassifier(C=c, tol=1e-10, kernel="gaussian", gamma=gamma)
dual_gaussian_model.train(Xtrain, Ytrain)
evaluate(dual_gaussian_model, f"Dual gaussian kernel, gamma={gamma}, C={c}")
print(f"Support vectors: {len(dual_gaussian_model.support_vectors)}")
print(vec2str(dual_gaussian_model.wstar))
print(f"b* = {dual_gaussian_model.gauss_bstar}")
```

Both SVM implementations take in their constructor the regularization hyperparameter `C`.

The Primal SVM constructor additionally takes a schedule function, which should take in the current time step (int)
and return a step size (float).

Training a primal SVM is done by calling `model.train(X, Y, epochs)` where epochs is the number of training epochs.

Prediction is done by simply calling `model.predict(x)`.

The Dual SVM constructor takes a float parameter `tol`. Any Lagrange multipliers `alpha_i <= tol` are thrown out and not
considered support vectors. The constructor can also take `kernel="linear"` or `kernel="gaussian"` to select between the
linear and gaussian kernels, respectively. If `kernel="gaussian"`, then an additional constructor parameter, `gamma` can
be passed, which represents the gamma term in the gaussian kernel.

For a gaussian kernel model, the following model parameters can also be inspected:
```
model.support_labels    # y_i of support vectors
model.support_vectors   # x_i of support vectors
model.alpha_star        # vector of non-zero lagrange multipliers
model.gauss_bstar       # bias term for a gaussian model, only available after calling predict() at least once

model.wstar             # for a gaussian kernel model, will return a nonsensical vector, and should not be inspected
```

Training a dual SVM model is done by calling `model.train(X, Y)`.

Prediction is done by calling `model.predict(x)`, where `x` is augmented with a leading 1 (bank_note dataset already does
this).

## Perceptron Models (Standard, Voted, Averaged)

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

## LMS Models (Batch/Stochastic Gradient Descent)

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

## Ensemble models (Adaboost, Bagging, Random Forests)

First ensure that data labels are {-1, 1}. The `bank` and `credit` datasets are set up for ensemble learning.

Basic usage:

```python
import datasets
import numpy as np
from EnsembleLearning.adaboost import AdaboostClassifier
from EnsembleLearning.bagging import BaggedClassifier
from EnsembleLearning.random_forests import RandomForestClassifier

## These datasets are set up for ensemble learning
S, val, attributes, labels = datasets.preprocess_bank_data(numeric_labels=True)
## S, val, attributes, labels = datasets.get_credit_data()

## Where attributes and labels look like this:
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

## Decision Trees

Basic usage:

```python
import id3
from id3 import ID3

## an example from the raw csv:
## high,med,5more,4,small,low,unacc

## Load data from csv into training and validation arrays
with open(f"./datasets/{dataset}/train.csv") as file:
    for line in file:
        S.append(tuple(line.strip().split(',')))

val = []
with open(f"./datasets/{dataset}/test.csv") as file:
    for line in file:
        val.append(tuple(line.strip().split(',')))

## Describe how the data is formatted
## attributes represents each column and its values
## these MUST be defined in the same order as columns in the CSV!
## not including the final label
attributes = {
    "buying":   ["vhigh", "high", "med", "low"],
    "maint":    ["vhigh", "high", "med", "low"],
    "doors":    ["2", "3", "4", "5more"],
    "persons":  ["2", "4", "more"],
    "lug_boot": ["small", "med", "big"],
    "safety":   ["low", "med", "high"]
}

## final labels are assumed to be the last column
labels = ["unacc", "acc", "good", "vgood"]

## instantiate id3 model
model = ID3()

## measure can be entropy, majority_error, or gini_index
measure = id3.entropy

## training examples can be fractional/weighted
## weights should be an array of numbers the same length as S
## if None, then weights will be set to np.ones(len(S)) (i.e. all ones)
weights = None

## maximum depth of decision tree can be set, but defaults to infinity otherwise
maxdep = 6

## train the model
model.train(S, attributes, labels, weights, measure, max_depth=maxdep)

## collect predictions from validation dataset
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
