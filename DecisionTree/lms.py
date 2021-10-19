import datasets
import numpy as np

# :NOTE: should use hw2 part 1 q5 as debug dataset

def sgn(num):
    if num > 0:
        return 1
    return -1

class BatchGradientDescentClassifier():
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def train(self, X, Y, threshold=1e-6):
        # Initialize w_0
        self.w = np.zeros(X.shape[1])
        norm = float("inf")
        
        # For t = 0, 1, 2, ... (until convergence)
        while norm > threshold:
            # Compute gradient of J(w) at (w), call it grad
            # count errors through all examples, then scale each x_i by errors and sum

            err = np.sum(Y - self.w.dot(X.T))
            grad = np.ravel(np.sum(-err * X, axis=0))

            # Update w_{t+1} = w_t - r*grad
            wnext = np.ravel(self.w - self.learning_rate*grad)
            norm = np.linalg.norm(wnext - self.w)
            self.w = wnext
    
    def predict(self, input):
        return self.w @ input

def StochasticGradientDescentClassifier():
    def train(self, X, Y, threshold=1e-6):
        # Initialize w_0
        self.w = np.zeros(len(X))
        err = float("inf")
        # For t = 0, 1, 2, ... (until below threshold)
        while err > threshold:
            # For each training example (x_i, y_i)
            for xi, yi in zip(X, Y):
                pass
                # w_j^{t+1} = w_j^t + r(y_i - w^Tx_i)x_ij
                # w^{t+1} = w^t + r(y_i - w^T x_i)x_i
    
    def predict(self, input):
        pass

if __name__ == "__main__":
    Xtrain, Ytrain, Xtest, Ytest, attributes = datasets.get_concrete_data()

    bgd = BatchGradientDescentClassifier(2 ** -8)
    bgd.train(np.matrix(Xtrain), Ytrain, threshold=1e-12)

    predictions = [bgd.predict(input) for input in Xtest]
    incorrect = np.sum(np.array(predictions) != np.array(Ytest))
    print(f"incorrect: {incorrect}")
    print(predictions)
    print(Ytest)

