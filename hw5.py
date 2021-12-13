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
    
def vec2str(vector, name="w'"):
    # augmented vector
    return f"{name} = ({','.join([f'{x :.5f}' for x in vector])})"

for width in [5, 10, 25, 50, 100]:
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