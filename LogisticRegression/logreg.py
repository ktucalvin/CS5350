import numpy as np
import math

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class MLEClassifier():
    def __init__(self, schedule):
        self.schedule = schedule

    def train(self, X, Y, threshold=1e-6):
        # Initialize w_0
        self.w = np.zeros(X.shape[1])
        norm = float("inf")
        N = X.shape[0]

        # For t = 0, 1, 2, ... (until below threshold)
        t = 1
        while norm > threshold:
            # shuffle examples
            shuffled = np.random.permutation(N)
            X = X[shuffled]
            Y = Y[shuffled]

            # attempt at vectorization
            gamma = self.schedule(t)
            err = (sigmoid(Y.T * (self.w @ X.T)) - 1).reshape((-1,1))
            grad = np.ravel(np.sum(err * X * Y, axis=0))
            wnext = self.w - gamma * grad
            norm = np.linalg.norm(wnext - self.w)
            self.w = np.ravel(wnext)
            t += 1

            # incomplete online learning
            # For each training example (x_i, y_i)
            # for xi, yi in zip(samplesX, samplesY):
            #     # gamma = self.schedule(t)
            #     gamma = 0.1
            #     # w^{t+1} = w^t - gamma * (sigmoid(yi * w @ x) - 1)(xi*yi)
            #     grad = N * (sigmoid(yi * (self.w @ xi.T)) - 1) * yi * xi
            #     wnext = self.w - gamma * grad
            #     t += 1
    
    def predict(self, input):
        if sigmoid(self.w @ input) > 0.5:
            return 1
        return -1

class MAPClassifier():
    def __init__(self, schedule, variance):
        self.schedule = schedule
        self.v = variance

    def train(self, X, Y, threshold=1e-6):
        # Initialize w_0
        self.w = np.zeros(X.shape[1])
        norm = float("inf")
        N = X.shape[0]

        # For t = 0, 1, 2, ... (until below threshold)
        t = 1
        while norm > threshold:
            # shuffle examples
            shuffled = np.random.permutation(N)
            X = X[shuffled]
            Y = Y[shuffled]

            # attempt at vectorization
            gamma = self.schedule(t)
            err = (sigmoid(Y.T * (self.w @ X.T)) - 1).reshape((-1,1))
            grad = np.ravel(np.sum(err * X * Y, axis=0)) + 2/self.v * self.w
            wnext = self.w - gamma * grad
            norm = np.linalg.norm(wnext - self.w)
            self.w = np.ravel(wnext)
            t += 1
    
    def predict(self, input):
        if sigmoid(self.w @ input) > 0.5:
            return 1
        return -1
