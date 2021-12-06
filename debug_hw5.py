import numpy as np
import datasets
from NeuralNetworks import nn

# prepend 1 for bias term
Xtrain = np.array([
    [1, 1, 1],
])
Ytrain = np.array([1]).reshape((-1,1))

Xtest = np.array([
    [1, 1, 1],
    [1, -1, -1]])
Ytest = np.array([1,-1]).reshape((-1,1))

def evaluate(model, label):
    predictions = np.array([model.predict(x) for x in Xtrain]).reshape(-1,1)
    incorrect_train = np.sum(predictions != Ytrain)
    # predictions = np.array([model.predict(x) for x in Xtest]).reshape(-1,1)
    # incorrect_test = np.sum(predictions != Ytest)
    print(predictions)
    print(f"{label}\nTrain Error: {float(incorrect_train) / len(Ytrain) :.5f}")
    
def vec2str(vector, name="w'"):
    # augmented vector
    return f"{name} = ({','.join([f'{x :.5f}' for x in vector])})"


width = 3
gamma_0 = 1e-4
model = nn.NeuralNetworkClassifier(width, Xtrain.shape[1],
                                   schedule=lambda t: gamma_0 / (1 + t),
                                   get_weight=nn.WeightInitializer.one)

model.train(Xtrain, Ytrain, epochs=1)
evaluate(model, "Debug model, weights = 1")
# Output should be ~2.8962, = 1 + 2s(1 + 2s(3))
