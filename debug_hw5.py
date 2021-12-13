import math
import numpy as np
import datasets
from NeuralNetworks import nn
import matplotlib.pyplot as plt

def evaluate(model, label):
    predictions = np.array([model.predict(x) for x in Xtrain]).reshape(-1,1)
    incorrect_train = np.sum(predictions != Ytrain)
    # predictions = np.array([model.predict(x) for x in Xtest]).reshape(-1,1)
    # incorrect_test = np.sum(predictions != Ytest)
    print(predictions)
    print(f"{label}\nTrain Error: {float(incorrect_train) / len(Ytrain) :.5f}")

Xtrain = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
])
# num negative odd = 0, even = 1
Ytrain = np.array([
    0,
    1,
    1,
    0
]).reshape((-1,1))

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

width = 3
gamma_0 = 0.01
d = 500
model = nn.NeuralNetworkClassifier(width, Xtrain.shape[1],
                                   schedule=lambda t: gamma_0 / (1 + gamma_0/d * t),
                                   get_weight=nn.WeightInitializer.gaussian)

# hack weights to match assignment's
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

model.train(Xtrain, Ytrain, epochs=1000) # now real training, does it converge
evaluate(model, "Debug model, weights = 1")

# ===== :note: debug convergence
exit()
# Xtrain, Ytrain, Xtest, Ytest, attributes = datasets.get_bank_note_data()
# Xtrain = np.array([
#     [1, 1, 1],
#     [1, 1, -1],
#     [1, -1, 1],
#     [1, -1, -1],
#     [-1, 1, 1],
#     [-1, 1, -1],
#     [-1, -1, 1],
#     [-1, -1, -1],
# ])
# # num negative odd = 0, even = 1
# Ytrain = np.array([
#     1,
#     0,
#     0,
#     0,
#     0,
#     0,
#     0,
#     0
# ]).reshape((-1,1))

# 150 / 1e-3 --> needs larger step size
epochs = 1000
gamma_0 = 1
d = 1e-2
width = 25

gauss_model = nn.NeuralNetworkClassifier(width, Xtrain.shape[1],
                                            schedule=lambda t: gamma_0 / (1 + gamma_0/d * t),
                                            # schedule=lambda t: 0.05,
                                            get_weight=nn.WeightInitializer.gaussian)
                                            # get_weight=nn.WeightInitializer.zero)

# see if weights converge
weights_history = {
    "3_0-0": [],
    "3_1-0": [],
    "3_2-0": [],
    "2_0-1": [],
    "2_0-2": [],
    "2_1-1": [],
    "2_1-2": [],
    "2_2-1": [],
    "2_2-2": [],
    "1_0-1": [],
    "1_0-2": [],
    "1_1-1": [],
    "1_1-2": [],
    "1_2-1": [],
    "1_2-2": []
}

epochs_axis = []
for i in range(epochs):
    model.train(Xtrain, Ytrain, epochs=1)
    if i % 100 == 0:
        epochs_axis.append(i)
        print(f"Epoch {i}")
        for key in weights_history.keys():
            weights_history[key].append(model.weights[key])

evaluate(model, f"Debug NN, Gaussian weights, width={width}, d={d}, gamma_0={gamma_0}, epochs={epochs}")

fig, axs = plt.subplots(15)
for i, key in enumerate(weights_history.keys()):
    axs[i].plot(epochs_axis, weights_history[key], '-o')
    axs[i].set_title(key)
plt.show()

print()

