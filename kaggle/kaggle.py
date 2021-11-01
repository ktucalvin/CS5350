import datasets
import numpy as np
from DecisionTree.id3 import ID3

S, val, attributes, labels = datasets.get_ilp_data(binarize=True)

model = ID3()

model.train(S, attributes, labels)

predictions = [model.predict(input) for input in S]
correct_labels = [x[-1] for x in S]
correct_train = np.sum(np.array(predictions) == np.array(correct_labels))

print(
    f"Training Accuracy: ({correct_train}/{len(S)}) {float(correct_train) / len(S) :.3f} ")

with open("kaggle-predictions.csv", "a+", encoding="utf8") as log:
    log.write("ID,Prediction\n")
    for input in val:
        id = input[0]
        pred = model.predict(input[1:])
        log.write(f"{id},{pred}\n")
        print(f"{id},{pred}")
