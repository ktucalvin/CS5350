import numpy as np
import math
from queue import SimpleQueue
import matplotlib.pyplot as plt


class NNActivations():
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    def linear(x):
        return x

    # bias neurons always have feature value 1, to account for bias
    def bias(x):
        return 1


class WeightInitializer():
    def gaussian():
        return np.random.standard_normal()

    def one():
        return 1

    def zero():
        return 0


GRAD_CACHE = {}
LOSS_HISTORY = []


class Node:
    def __init__(self, layer, id, activation=NNActivations.sigmoid):
        self.layer = layer  # layer this neuron is on
        self.id = id  # id of this neuron on this layer
        self.activation = activation

        self.inputs = []
        self.outputs = []
        self.value = 0

    def forward(self, weights):
        # compute the output activation of this node
        # output is always activation(dot product, weights and input values)
        # weights are always on inbound edges
        val = 0
        for node in self.inputs:
            val += node.value * weights[f"{self.layer}_{node.id}-{self.id}"]
        self.value = self.activation(val)

    def backward(self, weights):
        # only need to compute gradient of this node and inbound weights if this node is not a bias node
        if self.activation == NNActivations.bias:
            return

        # backprop
        # gradient of this node depends on outbound edges
        grad = 0
        for node in self.outputs:
            weight = weights[f"{node.layer}_{self.id}-{node.id}"]
            node_grad = GRAD_CACHE[node.get_id()] * weight
            if node.activation == NNActivations.sigmoid:
                node_grad *= node.value * (1 - node.value)
            grad += node_grad
        GRAD_CACHE[self.get_id()] = grad

        # compute gradients of inbound weights
        for node in self.inputs:
            wstr = f"{self.layer}_{node.id}-{self.id}"
            weight = weights[wstr]
            GRAD_CACHE[wstr] = GRAD_CACHE[self.get_id()] * \
                self.value * (1 - self.value) * weight

    def get_id(self):
        return f"{self.layer}-{self.id}"

    def __repr__(self) -> str:
        return f"Node(layer={self.layer}, id={self.id}, activation={self.activation.__name__})"


class NeuralNetworkClassifier:
    """
    Three layer feed-forward neural network with variable layer width.
    Each layer uses the sigmoid activation function except for the output layer.
    The first node on each hidden layer is a bias node whose activation is always one.
    """

    def __init__(self, layer_width, dims, schedule, get_weight=WeightInitializer.gaussian):
        self.schedule = schedule

        # construct layers ; weights are stored in single map
        self.weights = {}
        self.network = Node(3, 0, activation=NNActivations.linear)
        layer2 = [Node(2, 0, activation=NNActivations.bias)] + \
                 [Node(2, i) for i in range(1, layer_width)]
        layer1 = [Node(1, 0, activation=NNActivations.bias)] + \
                 [Node(1, i) for i in range(1, layer_width)]
        self.inputs = [Node(0, i, activation=NNActivations.linear)
                       for i in range(dims)]

        # Connect layers to form a 3-layer feed-forward network w/ bias nodes

        # Layer 2 ---> Output layer
        self.network.inputs = layer2
        for node in layer2[1:]:
            node.outputs = [self.network]
            node.inputs = layer1
        layer2[0].outputs = [self.network]

        # Layer 1 ---> Layer 2
        for node in layer1[1:]:
            node.outputs = layer2[1:]
            node.inputs = self.inputs
        layer1[0].outputs = layer2[1:]

        # Inputs ---> Layer 1
        for node in self.inputs:
            node.outputs = layer1[1:]
        
        # Initialize weights, {toLayer}_{fromId}-{toId}
        for i in range(layer_width):
            self.weights[f"3_{i}-0"] = get_weight()
            for j in range(1, layer_width):
                self.weights[f"2_{i}-{j}"] = get_weight()
                self.weights[f"1_{i}-{j}"] = get_weight()
        
    def train(self, X, Y, epochs=10):
        Y = np.ravel(Y)
        # sgd-like + backprop
        # For epoch 1... T
        for epoch in range(epochs):
            # Shuffle training data S
            shuffled = np.random.permutation(X.shape[0])
            samplesX = X[shuffled]
            samplesY = Y[shuffled]
            t = 1
            loss = 0

            # For each xi, yi in S
            for xi, yi in zip(samplesX, samplesY):
                # compute gradient w/ backprop
                # compute dL/dy here, begin BFS on children
                gamma = self.schedule(t)

                pred = self.predict(xi)
                loss += 0.5 * (pred - yi) ** 2
                topgrad = pred - yi
                GRAD_CACHE[self.network.get_id()] = topgrad
                for node in self.network.inputs:
                    GRAD_CACHE[f"{self.network.layer}_{node.id}-{self.network.id}"] = topgrad * node.value
                visited = set()
                queue = SimpleQueue()
                for node in self.network.inputs:
                    queue.put(node)

                # Call backward() on each node in BFS order, on the reverse graph
                while not queue.empty():
                    curr = queue.get()
                    if curr.get_id() in visited or curr.layer == 0:
                        continue
                    curr.backward(self.weights)
                    visited.add(curr.get_id())
                    for node in curr.inputs:
                        queue.put(node)

                # BFS again to update weights
                visited.clear()
                for node in self.network.inputs:
                    queue.put(node)

                while not queue.empty():
                    curr = queue.get()
                    if curr.get_id() in visited:
                        continue

                    # update weight
                    for outnode in curr.outputs:
                        wstr = f"{outnode.layer}_{curr.id}-{outnode.id}"
                        self.weights[wstr] = self.weights[wstr] - gamma * GRAD_CACHE[wstr]

                    visited.add(curr.get_id())
                    for node in curr.inputs:
                        queue.put(node)

                t += 1
                GRAD_CACHE.clear()

            loss /= len(samplesX)
            LOSS_HISTORY.append(loss)
        plt.plot(np.arange(epochs), LOSS_HISTORY)
        plt.show()

    def predict(self, x):
        # plug input into network by rewriting input neurons
        # as a result, don't call forward() on input layer
        for node, component in zip(self.inputs, x):
            node.value = component

        # Forward pass:
        # Call forward() on each node in the network, in DFS postorder
        visited = set()
        stack = [self.network]
        while len(stack):
            curr = stack[-1]  # just want to peek at top of stack
            if curr.get_id() in visited:
                stack.pop()

            # check subset of curr.inputs to ensure iterative postorder DFS works
            descendants = [n for n in curr.inputs if n.get_id()
                           not in visited and n.layer > 0]
            if len(descendants):
                stack += descendants
            else:
                visited.add(curr.get_id())
                curr.forward(self.weights)
                stack.pop()

        return self.network.value
