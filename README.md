This is a machine learning library developed by Calvin Tu for CS5350/6350 at the University of Utah.

# Decision Trees

Basic usage:

```python
import id3

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
