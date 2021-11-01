import numpy as np

def sgn(num):
    if num > 0:
        return 1
    return -1 # just treat 0 as a negative example

# Standard Perceptron
class StandardPerceptron:
    def __init__(self, learning_rate):
        self.r = learning_rate

    def train(self, X, Y, epochs=10):
        # initialize w_0 = zero vector in R^n
        self.w = np.zeros(X.shape[1])
        # for epoch = 1 ... T, T is a hyperparameter
        for epoch in range(epochs):
            # shuffle the data
            shuffled = np.random.permutation(X.shape[0])
            samplesX = X[shuffled]
            samplesY = Y[shuffled]
            # for each training example (xi, yi), where x is a real vector and yi in {-1, +1}
            for xi, yi in zip(samplesX, samplesY):
                # if yi * w^\top @ xi <= 0
                if yi * self.w.T @ xi <= 0:
                    # Update w <-- w + r*yi*xi
                    self.w += self.r * yi * xi

    def predict(self, x):
        # Prediction: sgn(w^T @ x)
        return sgn(self.w.T @ x)


# Voted Perceptron
class VotedPerceptron:
    def __init__(self, learning_rate):
        self.r = learning_rate
        self.weights = []
    
    def train(self, X, Y, epochs=10):
        # Initialize w_0 = zero vector in R^n, m = 0
        # Final form will be [(w_1, c_1), ... (w_k, c_k)]
        self.weights = []
        w = np.zeros(X.shape[1])
        C = 1
        # don't really need to track m, since can just use append()

        # For epoch = 1 ... T
        for epoch in range(epochs):
            # shuffle the data
            shuffled = np.random.permutation(X.shape[0])
            samplesX = X[shuffled]
            samplesY = Y[shuffled]
            # For each training example (xi, yi)
            for xi, yi in zip(samplesX, samplesY):
                # if yi * w^T @ xi <= 0
                if (yi * w.T @ xi) <= 0:
                    # Update w_{m+1} <-- w_m + r * yi * xi
                    wnext = w + self.r * yi * xi
                    self.weights.append((wnext, C))
                    # m = m + 1
                    w = wnext
                    # C_m = 1
                    C = 1
                else:
                    # C_m is roughly the number of correct predictions
                    # C_m = C_m + 1
                    C += 1
            # Return the list (w_1, c_1), ... (w_k, c_k)

    def predict(self, x):
        # Prediction: sgn(sum_{1,k} c_i * sgn(w_i^T @ x))
        total = 0
        for w,c in self.weights:
            total += c * sgn(w.T @ x)
        return sgn(total)

# Averaged Perceptron
class AveragedPerceptron:
    def __init__(self, learning_rate):
        self.r = learning_rate
    
    def train(self, X, Y, epochs=10):
        # Initialize w, a = zero vectors in R^n
        w = np.zeros(X.shape[1])
        self.a = np.zeros(X.shape[1])
        # For epoch = 1 ... T
        for epoch in range(epochs):
            # shuffle the data
            shuffled = np.random.permutation(X.shape[0])
            samplesX = X[shuffled]
            samplesY = Y[shuffled]
            # For each training example (xi, yi)
            for xi, yi in zip(samplesX, samplesY):
                # If yi * w^T @ xi <= 0
                if yi * w.T @ xi <= 0:
                    # Update w <-- w + r * yi * xi
                    w += self.r * yi * xi
                # a <-- a + w
                self.a += w
        # Return a

    def predict(self, x):
        # Prediction: sgn(a^T @ x)
        return sgn(self.a.T @ x)
