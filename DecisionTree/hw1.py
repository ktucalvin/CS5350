import id3
import datasets
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

def run_hw1_bank():
    for b in [False, True]: # lmao boolean for loop
        print(f"\nRefilling unknown? {b}\n")
        S, val, attributes, labels = datasets.preprocess_bank_data(refill_unknown=b)
        
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