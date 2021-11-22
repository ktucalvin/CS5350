from math import exp
import numpy as np
from scipy.optimize import minimize, LinearConstraint
from scipy.spatial.distance import pdist, squareform

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
                    # w <-- w - gamma_t[0, w_0] + gamma_t * C * N * yi * xi
                    w0 = np.copy(self.w)
                    w0[0] = 0
                    self.w = self.w - (gamma * w0) + (gamma * self.C * N * yi * xi)
                else:
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

class DualSVMClassifier:
    def __init__(self, C, tol=1e-6, kernel="linear", gamma=0.1):
        self.C = C
        self.tol = tol
        self.kernel = kernel
        self.gamma = gamma
        self.gauss_bstar = None

    def train(self, X, Y):
        # unaugment training data, dual formulation will handle bias separately
        X = X[:,1:]
        Y = Y.reshape((-1))
        N = X.shape[0]
        svm_constraints ={ "type": "eq", "fun": lambda A: A @ Y } # A @ Y = 0
        svm_bounds = [(0, self.C) for _ in range(N)] # limit each component 0 <= alpha_i <= C

        def svm_objective(A):
            if self.kernel == "linear":
                return 0.5 * np.sum(np.outer(Y*A, Y*A) * (X @ X.T).T) - np.sum(A)
            
            # pairwise distance
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html 
            dists = squareform(pdist(X, "euclidean"))
            dists = np.exp(-1/self.gamma * dists ** 2)
            return 0.5 * np.sum(np.outer(Y*A, Y*A) * dists.T) - np.sum(A)

        alpha_star = minimize(fun=svm_objective, x0=np.zeros(N), method="SLSQP",
                              constraints=svm_constraints, bounds=svm_bounds)
        
        if alpha_star.status:
            print("Optimization for alpha* failed. Aborting.")
            print(alpha_star)
            exit()

        nonzero_indices = abs(alpha_star.x) >= self.tol
        self.alpha = alpha_star.x[nonzero_indices]
        self.support_vectors = X[nonzero_indices]
        self.support_labels = Y[nonzero_indices]
        self.recover_wb(alpha_star)

    def recover_wb(self, alpha_star):
        if not len(self.support_vectors):
            print("No support vectors found, try turning down the tolerance")
            print(alpha_star)
            exit()

        self.wstar = np.zeros_like(self.support_vectors[0], dtype='float64')
        for ai, yi, xi in zip(self.alpha, self.support_labels, self.support_vectors):
            self.wstar += ai * yi * xi

        # alpha_i already within (0, c) interval, so b* is just average y_j - w* @ xj
        b = 0
        for yj, xj in zip(self.support_labels, self.support_vectors):
            b += yj - self.wstar @ xj
        
        self.bstar = b / len(self.support_labels)

    def predict(self, x):
        """Predict given an *augmented* input. Prediction will strip off the first term."""
        # unaugment input
        x = x[1:]
        if self.kernel == "linear":
            return sgn(self.wstar.T @ x)
        
        # Compute b* along with prediction, cache b* for later
        if self.gauss_bstar is None:
            b = 0
            total = 0
            for ai, yi, xi in zip(self.alpha, self.support_labels, self.support_vectors):
                for xj in self.support_vectors:
                    b += ai * yi * self.rbf(xi, xj)
                total += ai * yi * self.rbf(xi, x)
            b /= len(self.support_vectors) ** 2
            self.gauss_bstar = b
            return sgn(total + b)

        # Use cached b* for prediction
        total = 0
        xj = self.support_vectors[0]
        for ai, yi, xi in zip(self.alpha, self.support_labels, self.support_vectors):
            total += ai * yi * self.rbf(xi, x)
        return sgn(total + self.gauss_bstar)
        

    def rbf(self, x, z):
        return np.exp(-1/self.gamma * np.linalg.norm(x - z) ** 2)
