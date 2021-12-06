import numpy as np


class NNActivations():
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

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


class Node:
    def __init__(self, layer, id, activation=NNActivations.sigmoid):
        self.layer = layer  # layer this neuron is on
        self.id = id  # id of this neuron on this layer
        self.activation = activation

        self.inputs = []
        self.outputs = []
        self.value = 0

    def forward(self):
        # compute the output activation of this node
        # output is always activation(dot product, weights and input values)
        # weights are always on inbound edges
        val = 0
        for node, weight in self.inputs:
            val += node.value * weight
        self.value = self.activation(val)

    def backward(self):
        # compute the gradient of this node, using the gradient cache (backprop)

        # DFS forward accumulating gradient? then apply gradients in train
        pass
    
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

        # construct layers
        # layer is list of tuples (Node, weight)
        self.network = Node(3, 0, activation=NNActivations.linear)
        layer2 = [(Node(2, 0, activation=NNActivations.bias), get_weight())] + \
                 [(Node(2, i), get_weight()) for i in range(1, layer_width)]
        layer1 = [(Node(1, 0, activation=NNActivations.bias), get_weight())] + \
                 [(Node(1, i), get_weight()) for i in range(1, layer_width)]
        self.inputs = [Node(0, i, activation=NNActivations.linear)
                       for i in range(dims)]

        # Connect layers, such that bias node has outputs to next layer but no inputs

        # Layer 2 ---> Output layer
        self.network.inputs = layer2
        for node, _ in layer2[1:]:
            node.outputs = [(self.network, get_weight())]
            node.inputs = layer1
        layer2[0][0].outputs = [(self.network, get_weight())]

        # Layer 1 ---> Layer 2
        for node, _ in layer1[1:]:
            node.outputs = layer2[1:]
            node.inputs = [(node, get_weight()) for node in self.inputs]
        layer1[0][0].outputs = layer2[1:]

        # Inputs ---> Layer 1
        for node in self.inputs:
            node.outputs = layer1[1:]

    def train(self, X, Y, epochs=10):
        # sgd-like + backprop
        # For epoch 1... T
        for epoch in range(epochs):
            # Shuffle training data S
            shuffled = np.random.permutation(N)
            samplesX = X[shuffled]
            samplesY = Y[shuffled]
            t = 1
            # For each xi, yi in S
            for xi, yi in zip(samplesX, samplesY):
                # compute gradient w/ backprop
                # call backward() on each node in BFS order, on the reverse graph starting from output node
                # update weights
                gamma = self.schedule(t)
                t += 1

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
            curr = stack[-1] # just want to peek at top of stack
            if curr.get_id() in visited:
                stack.pop()

            # check subset of curr.inputs to ensure iterative postorder DFS works
            descendants = [n for n, _ in curr.inputs if n.get_id() not in visited and n.layer > 0]
            if len(descendants):
                stack += descendants
            else:
                visited.add(curr.get_id())
                curr.forward()
                stack.pop()
        
        return self.network.value
