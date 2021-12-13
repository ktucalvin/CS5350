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
    
def vec2str(vector, name="w'"):
    # augmented vector
    return f"{name} = ({','.join([f'{x :.7f}' for x in vector])})"

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
