import id3
import numpy as np
from id3 import ID3

entropy = id3.entropy
majority_error = id3.majority_error
gini_index = id3.gini_index

def run_simple(dataset, attributes, labels, measure=entropy, verbose=False, max_depth=float("inf")):
    """Run ID3 on a dataset"""
    model = ID3()
    S = []

    with open(f"./datasets/{dataset}/train.csv") as file:
        for line in file:
            S.append(tuple(line.strip().split(',')))

    val = []
    with open(f"./datasets/{dataset}/test.csv") as file:
        for line in file:
            val.append(tuple(line.strip().split(',')))

    model.train(S, attributes, labels, None, measure, max_depth=max_depth)

    predictions = [model.predict(input) for input in S]
    correct_labels = [x[-1] for x in S]
    correct_train = np.sum(np.array(predictions) == np.array(correct_labels))

    predictions = [model.predict(input) for input in val]
    correct_labels = [x[-1] for x in val]
    correct_test = np.sum(np.array(predictions) == np.array(correct_labels))

    if verbose:
        print("\n Tree: ")
        print(model.tree)
        print("\n Predictions: ")
        print(predictions)
    print(
        f"Training Accuracy: ({correct_train}/{len(S)}) {float(correct_train) / len(S) :.3f} ")
    print(
        f"Test Accuracy: ({correct_test}/{len(val)}) {float(correct_test) / len(val) :.3f} ")


def run_debug_datasets(measure=entropy):
    # :NOTE: MUST declare attributes in same order as they are in the csv
    shapes_attributes = {
        "color":   ["red", "green", "blue"],
        "shape":    ["square", "circle", "triangle"]
    }
    shapes_labels = ["a", "b", "c"]

    tennis_attributes = {
        "outlook": ["sunny", "overcast", "rainy"],
        "temperature": ["hot", "medium", "cool"],
        "humidity": ["high", "normal", "low"],
        "wind": ["strong", "weak"]
    }
    tennis_labels = ["0", "1"]

    car_attributes = {
        "buying":   ["vhigh", "high", "med", "low"],
        "maint":    ["vhigh", "high", "med", "low"],
        "doors":    ["2", "3", "4", "5more"],
        "persons":  ["2", "4", "more"],
        "lug_boot": ["small", "med", "big"],
        "safety":   ["low", "med", "high"]
    }
    car_labels = ["unacc", "acc", "good", "vgood"]

    run_simple("shapes", shapes_attributes, shapes_labels, measure, verbose=True)
    run_simple("tennis", tennis_attributes, tennis_labels, measure)
    run_simple("car", car_attributes, car_labels, measure, max_depth=6)

def print_results(S, train_results, test_results, dataset):
    # Sort by measure function, then by depth
    train_results = sorted(train_results, key = lambda x: (x[1], x[0]))
    test_results = sorted(test_results, key = lambda x: (x[1], x[0]))

    measure_averages = {
        entropy.__name__: [],
        majority_error.__name__: [],
        gini_index.__name__: []
    }
    print(f"== {dataset} training data statistics")
    print("Measure & Depth & \# Correct & \# Incorrect & Accuracy")
    for (depth, measure, correct, accuracy) in train_results:
        measure_averages[measure].append(len(S) - correct)
        print(f"{measure} & {depth} & {correct} & {len(S) - correct} & {accuracy :.3f} \\\\")

    print("\nAverage errors per measure function")
    print("Measure & Average Errors")
    for measure,incorrect in measure_averages.items():
        print(f"{measure} & {sum(incorrect) / len(incorrect) :.3f} \\\\")
        measure_averages[measure] = []
    
    print()

    print(f"== {dataset} test data statistics")
    print("Measure & Depth & \# Correct & \# Incorrect & Accuracy")
    for (depth, measure, correct, accuracy) in test_results:
        measure_averages[measure].append(len(S) - correct)
        print(f"{measure} & {depth} & {correct} & {len(S) - correct} & {accuracy :.3f} \\\\")

    print("\nAverage errors per measure function")
    print("Measure & Average Errors")
    for measure,incorrect in measure_averages.items():
        print(f"{measure} & {sum(incorrect) / len(incorrect) :.3f} \\\\")
        measure_averages[measure] = []

def run_hw1_car():
    S = []

    with open("./datasets/car/train.csv") as file:
        for line in file:
            S.append(tuple(line.strip().split(',')))

    val = []
    with open("./datasets/car/test.csv") as file:
        for line in file:
            val.append(tuple(line.strip().split(',')))

    attributes = {
        "buying":   ["vhigh", "high", "med", "low"],
        "maint":    ["vhigh", "high", "med", "low"],
        "doors":    ["2", "3", "4", "5more"],
        "persons":  ["2", "4", "more"],
        "lug_boot": ["small", "med", "big"],
        "safety":   ["low", "med", "high"]
    }
    labels = ["unacc", "acc", "good", "vgood"]

    train_results = []
    test_results = []
    for depth in range(6):
        depth = depth + 1
        model = ID3()
        for measure in [entropy, majority_error, gini_index]:
            # Train model with set depth and measure
            model.train(S, attributes, labels, None, measure, max_depth=depth)

            # Predict on training set
            predictions = [model.predict(input) for input in S]
            correct_labels = [x[-1] for x in S]
            num_correct = np.sum(np.array(predictions) == np.array(correct_labels))
            train_results.append((depth, measure.__name__, num_correct, float(num_correct) / len(S)))

            # Predict on test set
            predictions = [model.predict(input) for input in val]
            correct_labels = [x[-1] for x in val]
            num_correct = np.sum(np.array(predictions) == np.array(correct_labels))
            test_results.append((depth, measure.__name__, num_correct, float(num_correct) / len(val)))
    
    print_results(S, train_results, test_results, "Car")

def preprocess_bank_data(refill_unknown=False):
    S = []

    with open("./datasets/bank/train.csv") as file:
        for line in file:
            S.append(tuple(line.strip().split(',')))

    val = []
    with open("./datasets/bank/test.csv") as file:
        for line in file:
            val.append(tuple(line.strip().split(',')))

    # For numeric attributes, 1 = above median, 0 = below median
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
    labels = ["yes", "no"]

    # preprocessing:
    # Compute median for age, balance, day, duration, campaign, pdays, previous
    attr = list(attributes.keys())
    numeric_features = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
    media = []
    for feature in numeric_features:
        # exclude rows where pdays = -1 from pdays median
        if feature == "pdays":
            media.append(np.median([float(x[-4]) for x in S if x[-4] != "-1"]))
        else:
            media.append(np.median([float(x[attr.index(feature)]) for x in S]))

    # Replace numeric features with 1 or 0 if above or below median (or -1 if pdays)
    for index,example in enumerate(S):
        example = list(example)
        for media_index,feature in enumerate(numeric_features):
            feature_index = attr.index(feature)
            if not (feature == "pdays" and example[-4] == "-1"):
                example[feature_index] = "1" if float(example[feature_index]) > media[media_index] else "0"
        S[index] = tuple(example)

    for index,example in enumerate(val):
        example = list(example)
        for media_index,feature in enumerate(numeric_features):
            feature_index = attr.index(feature)
            if not (feature == "pdays" and example[-4] == "-1"):
                example[feature_index] = "1" if float(example[feature_index]) > media[media_index] else "0"
        val[index] = tuple(example)

    # If treating "unknown" as a value, return here
    if not refill_unknown:
        return S, val, attributes, labels

    # Determine most common label for these features
    unknown_features = ["job", "education", "contact", "poutcome"]
    common_labels = []
    for feature in unknown_features:
        uniq_labels, count = np.unique([x[attr.index(feature)] for x in S if x[attr.index(feature)] != "unknown"], return_counts=True)
        common_labels.append(uniq_labels[np.argmax(count)])
    
    # For each example, replace each unknown feature with the most common label for that feature
    for index,example in enumerate(S):
        example = list(example)
        for label_index,feature in enumerate(unknown_features):
            feature_index = attr.index(feature)
            example[feature_index] = common_labels[label_index]
        S[index] = tuple(example)

    for index,example in enumerate(val):
        example = list(example)
        for label_index,feature in enumerate(unknown_features):
            feature_index = attr.index(feature)
            example[feature_index] = common_labels[label_index]
        val[index] = tuple(example)
    
    return S, val, attributes, labels

def run_hw1_bank():
    for b in [False, True]: # lmao boolean for loop
        print(f"\nRefilling unknown? {b}\n")
        S, val, attributes, labels = preprocess_bank_data(refill_unknown=b)
        
        train_results = []
        test_results = []
        for depth in range(16):
            depth = depth + 1
            model = ID3()
            for measure in [entropy, majority_error, gini_index]:
                # Train model with set depth and measure
                model.train(S, attributes, labels, None, measure, max_depth=depth)

                # Predict on training set
                predictions = [model.predict(input) for input in S]
                correct_labels = [x[-1] for x in S]
                num_correct = np.sum(np.array(predictions) == np.array(correct_labels))
                train_results.append((depth, measure.__name__, num_correct, float(num_correct) / len(S)))

                # Predict on test set
                predictions = [model.predict(input) for input in val]
                correct_labels = [x[-1] for x in val]
                num_correct = np.sum(np.array(predictions) == np.array(correct_labels))
                test_results.append((depth, measure.__name__, num_correct, float(num_correct) / len(val)))
        
        print_results(S, train_results, test_results, "Bank")
    

if __name__ == "__main__":
    for measure in [entropy, majority_error, gini_index]:
        run_debug_datasets(measure)
        print("==")
    
    print("========== CAR DATA ==========")
    run_hw1_car()
    
    print()

    print("========== BANK DATA ==========")
    run_hw1_bank()