import numpy as np
import datasets
from NeuralNetworks import nn

# ===== :note: debug forward pass

# width = 3
# gamma_0 = 5
# model = nn.NeuralNetworkClassifier(width, Xtrain.shape[1],
#                                    schedule=lambda t: gamma_0 / (1 + t),
#                                    get_weight=nn.WeightInitializer.one)

# model.train(Xtrain, Ytrain, epochs=0)
# evaluate(model, "Debug model, weights = 1")

# print("Output should be ~2.8962, = 1 + 2s(1 + 2s(3))")
# pred = model.predict(np.array([1,1,1]))
# if abs(pred - 2.896200694978303) >= 1e-6:
#     print(f"ERROR: got {pred} instead")
# else:
#     print("Correct forward pass!")
# print()

# ===== :note: debug backprop

# width = 3
# gamma_0 = 1
# model = nn.NeuralNetworkClassifier(width, Xtrain.shape[1],
#                                    schedule=lambda t: gamma_0 / (1 + t),
#                                    get_weight=nn.WeightInitializer.one)

# # hack weights to match assignment's
# model.weights = {
#     "3_0-0": -1,
#     "3_1-0": 2,
#     "3_2-0": -1.5,
#     "2_0-1": -1,
#     "2_0-2": 1,
#     "2_1-1": -2,
#     "2_1-2": 2,
#     "2_2-1": -3,
#     "2_2-2": 3,
#     "1_0-1": -1,
#     "1_0-2": 1,
#     "1_1-1": -2,
#     "1_1-2": 2,
#     "1_2-1": -3,
#     "1_2-2": 3,
# }

# print(model.predict(np.array([1,1,1])))
# model.train(Xtrain, Ytrain, epochs=1) # breakpoint here

# model.train(Xtrain, Ytrain, epochs=100) # now real training, does it at least converge on 1?
# evaluate(model, "Debug model, weights = 1")

# ===== :note: debug convergence

# Xtrain, Ytrain, Xtest, Ytest, attributes = datasets.get_bank_note_data()
Xtrain = np.array([
    [1, 1, 1],
    [1, 1, -1],
    [1, -1, 1],
    [1, -1, -1],
    [-1, 1, 1],
    [-1, 1, -1],
    [-1, -1, 1],
    [-1, -1, -1],
])
# num negative odd = -1, even = 1
Ytrain = np.array([
    1,
    -1,
    -1,
    1,
    -1,
    1,
    1,
    -1
]).reshape((-1,1))

def sgn(x):
    if x > 0:
        return 1
    return -1

def evaluate(model, label):
    predictions = np.array([model.predict(x) for x in Xtrain]).reshape(-1,1)
    incorrect_train = np.sum(predictions != Ytrain)
    # predictions = np.array([model.predict(x) for x in Xtest]).reshape(-1,1)
    # incorrect_test = np.sum(predictions != Ytest)
    print(predictions)
    print(f"{label}\nTrain Error: {float(incorrect_train) / len(Ytrain) :.5f}")

# 150 / 1e-3 --> needs larger step size
epochs = 500
gamma_0 = 1
d = 1e-2
width = 5

gauss_model = nn.NeuralNetworkClassifier(width, Xtrain.shape[1],
                                            # schedule=lambda t: gamma_0 / (1 + gamma_0/d * t),
                                            schedule=lambda t: 0.1,
                                            get_weight=nn.WeightInitializer.gaussian)
gauss_model.train(Xtrain, Ytrain, epochs)
evaluate(gauss_model, f"Debug NN, Gaussian weights, width={width}, d={d}, gamma_0={gamma_0}, epochs={epochs}")
print()

