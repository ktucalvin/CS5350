import copy
import numpy as np

# :NOTE:
# p_i is probability of i-th label
# p_i = count(i-th label) / len(labels)
# ex: binary, p_+ and p_- are probability of positive and negative labels

def entropy(S, labels):
        # - sum_i^K p_i log_2(p_i)
        sum = 0
        for lab in labels:
            examples = [example for example in S if example[-1] == lab]
            if len(S) and len(examples):
                p_i = len(examples) / len(S)
                sum += p_i * np.log2(p_i)
        return -sum

def majority_error(S, labels):
    pass
    # determine majority label
    # return count(minority labels) / count(labels)

def gini_index(S, labels):
    pass
    # see top-level note for what p_k is
    # return 1 - sum_1^K p_k^2

class ID3:
    def __init__(self):
        self.tree = {}

    def most_common_label(self, S):
        uniq_labels, count = np.unique([x[-1] for x in S], return_counts=True)
        return uniq_labels[np.argmax(count)]

    def gain(self, S, values, measure, labels, index):
        # measure(S) - sum_{v in values(A)} (len(S_v) / len(S) measure(S_v))
        sum = 0
        for v in values:
            S_v = [example for example in S if example[index] == v]
            sum += len(S_v) / len(S) * measure(S_v, labels)
        return measure(S, labels) - sum

    def train(self, S, attributes, labels, measure=entropy, max_depth=float("inf")):
        self.tree = self.trainR(S, attributes, list(attributes.keys()), labels, measure, max_depth)

    def trainR(self, S, attributes, order, labels, measure=entropy, max_depth=float("inf"), depth=0):
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
        gains = [self.gain(S, attributes[a], measure, labels, attr.index(a)) for a in attr]
        A = attr[np.argmax(gains)]
        indexA = order.index(A)

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
                root[A][v] = self.trainR(S_v, newattr, order, labels, depth=(depth + 1))
        # return root node
        return root

    def predict(self, input, order):
        """
            Predict label given input tuple
            input -- input tuple
            order -- the order of attributes in the tuple
        """
        # tree is striped where odd layers are attribute names to split on
        # and even layers are dicts of attribute value to the next layer
        # traversal ends when we reach something that is not a dict

        subtree = self.tree
        i = 0
        while isinstance(subtree, dict) and i < len(order):
            attr = list(subtree.keys())[0]
            subtree = subtree[attr][input[order.index(attr)]]
            i += 1
        
        return subtree



def run_simple(dataset, attributes, labels, verbose=False):
    """Run ID3 on the shapes dataset"""
    model = ID3()
    S = []

    with open(f"./datasets/{dataset}/train.csv") as file:
        for line in file:
            S.append(tuple(line.strip().split(',')))

    val = []
    with open(f"./datasets/{dataset}/test.csv") as file:
        for line in file:
            val.append(tuple(line.strip().split(',')))

    model.train(S, attributes, labels, measure=entropy)

    predictions = [model.predict(input, list(attributes.keys())) for input in S]
    correct_labels = [x[-1] for x in S]
    correct_train = np.sum(np.array(predictions) == np.array(correct_labels))

    predictions = [model.predict(input, list(attributes.keys())) for input in val]
    correct_labels = [x[-1] for x in val]
    correct_test = np.sum(np.array(predictions) == np.array(correct_labels))

    if verbose:
        print("\n Tree: ")
        print(model.tree)
        print("\n Predictions: ")
        print(predictions)
    print(f"Training Accuracy: ({correct_train}/{len(S)}) {float(correct_train) / len(S) :.3f} ")
    print(f"Test Accuracy: ({correct_test}/{len(val)}) {float(correct_test) / len(val) :.3f} ")

def run_car():
    # data-desc is not standardized, data has to be pre-processed
    # Car example
    car_models = [ID3() for _ in range(6)]
    car_data = []
    car_attributes = {
        "buying":   ["vhigh", "high", "med", "low"],
        "maint":    ["vhigh", "high", "med", "low"],
        "doors":    ["2", "3", "4", "5more"],
        "persons":  ["2", "4", "more"],
        "lug_boot": ["small", "med", "big"],
        "safety":   ["low", "med", "high"]
    }
    labels = ["unacc", "acc", "good", "vgood"]
    with open("./datasets/car/train.csv") as file:
        for line in file:
            [buying, maint, doors, persons, lug_boot,
                safety, label] = line.strip().split(',')
            car_data.append(
                (buying, maint, doors, persons, lug_boot, safety, label))

    car_val = []
    car_predictions = []
    with open("./datasets/car/test.csv") as file:
        for line in file:
            [buying, maint, doors, persons, lug_boot,
                safety, label] = line.strip().split(',')
            car_val.append(
                (buying, maint, doors, persons, lug_boot, safety, label))

    for depth, model in enumerate(car_models, start=1):
        model.train(car_data, car_attributes, labels,
                    measure=ID3.entropy, max_depth=depth)

    for model in car_models:
        car_predictions.append([model.predict(input) for input in car_val])

    print(car_predictions)

if __name__ == "__main__":
    """ When run from CLI, just dump data necessary to do hw1 """

    # :NOTE: MUST declare attributes in same order as they are in the csv
    shapes_attributes = {
        "color":   ["red", "green", "blue"],
        "shape":    ["square", "circle", "triangle"]
    }
    shapes_labels = ["a", "b", "c"]
    run_simple("shapes", shapes_attributes, shapes_labels)

    tennis_attributes = {
        "outlook": ["sunny", "overcast", "rainy"],
        "temperature": ["hot", "medium", "cool"],
        "humidity": ["high", "normal", "low"],
        "wind": ["strong", "weak"]
    }
    tennis_labels = ["0", "1"]
    run_simple("tennis", tennis_attributes, tennis_labels)

    car_attributes = {
        "buying":   ["vhigh", "high", "med", "low"],
        "maint":    ["vhigh", "high", "med", "low"],
        "doors":    ["2", "3", "4", "5more"],
        "persons":  ["2", "4", "more"],
        "lug_boot": ["small", "med", "big"],
        "safety":   ["low", "med", "high"]
    }
    car_labels = ["unacc", "acc", "good", "vgood"]
    run_simple("car", car_attributes, car_labels)

