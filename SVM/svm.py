import numpy as np

def sgn(num):
    if num > 0:
        return 1
    return -1 # just treat 0 as a negative example

class PrimalSVMClassifier:
    def __init__(self, schedule, C):
        """
        Initialize SVM with regularization parameter C and given learning rate schedule.
        SVM is in the primal domain and solved using SGD.
        The schedule should be a function that takes the current timestep and returns a step size.
        """
        self.C = C
        self.schedule = schedule

    def train(self, X, Y, epochs=10):
        # initialize w^0 = zero vector in R^n
        # this implementation will fold bias into self.w
        # though it must be unfolded for each update
        self.w = np.zeros(X.shape[1])
        N = X.shape[0]

        # For epoch 1... T
        for epoch in range(epochs):
            # Shuffle training data S
            shuffled = np.random.permutation(N)
            samplesX = X[shuffled]
            samplesY = Y[shuffled]
            t = 1
            # For each xi, yi in S
            for xi, yi in zip(samplesX, samplesY):
                gamma = self.schedule(t)
                # if yiw^Txi <= 1
                if yi * self.w.T @ xi <= 1:
                    # print("bad prediction, update weights")
                    # w <-- w - gamma_t[0, w_0] + gamma_t * C * N * yi * xi
                    w0 = np.copy(self.w)
                    w0[0] = 0
                    self.w = self.w - (gamma * w0) + (gamma * self.C * N * yi * xi)
                else:
                    # print("good prediction, increase strength of existing weights")
                    # w0 <-- (1-gamma_t)w_0
                    # update everything except bias
                    # unlike lecture slides, bias is the first term
                    bias = self.w[0]
                    self.w = (1 - gamma) * self.w
                    self.w[0] = bias
                t += 1
            # Return w

    def predict(self, x):
        # Prediction: sgn(w^T @ x)
        return sgn(self.w.T @ x)
