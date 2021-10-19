import datasets
import numpy as np

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
            # error is PER ROW
            # grad = weighted sum of each example, where weights are each row's error

            err = Y - self.w.dot(X.T)
            grad = np.ravel(np.sum(-err * X, axis=0))

            # Update w_{t+1} = w_t - r*grad
            wnext = self.w - self.learning_rate*grad
            norm = np.linalg.norm(wnext - self.w)
            self.w = np.ravel(wnext)
    
    def predict(self, input):
        return self.w @ input

class StochasticGradientDescentClassifier():
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def train(self, X, Y, threshold=1e-6):
        # Initialize w_0
        self.w = np.zeros(X.shape[1])
        norm = float("inf")

        # For t = 0, 1, 2, ... (until below threshold)
        t = 0
        while norm > threshold:
            # shuffle examples
            shuffled = np.random.permutation(X.shape[1])
            samplesX = X[shuffled]
            samplesY = Y[shuffled]

            # For each training example (x_i, y_i)
            for xi, yi in zip(samplesX, samplesY):
                t += 1
                # w^{t+1} = w^t + r(y_i - w^T x_i)x_i
                wnext = self.w + self.learning_rate * (yi - self.w.dot(xi.T)) * xi
                norm = np.linalg.norm(wnext - self.w)
                self.w = np.ravel(wnext)
    
    def predict(self, input):
        return self.w @ input

if __name__ == "__main__":
    # Xtrain, Ytrain, Xtest, Ytest, attributes = datasets.get_hw2_data()
    Xtrain, Ytrain, Xtest, Ytest, attributes = datasets.get_concrete_data()

    # this implementation will perform poorly because the learning rate isn't adjustable

    bgd = BatchGradientDescentClassifier(2 ** -8)
    bgd.train(np.matrix(Xtrain), np.array(Ytrain), threshold=1e-6)

    predictions = [bgd.predict(input) for input in Xtrain]
    incorrect = len(Ytrain) - np.sum(np.isclose(np.array(predictions), np.array(Ytrain), atol=1e-6))
    print(f"BGD Training error: {incorrect/len(Ytrain)}")

    # predictions = [bgd.predict(input) for input in Xtest]
    # incorrect = len(Ytest) - np.sum(np.isclose(np.array(predictions), np.array(Ytest), atol=1e-6))
    # print(f"incorrect: {incorrect/len(Ytest)}")
    # print(f"predictions: {predictions}")
    # print(f"correct labels: {Ytest}")
    # print(f"bgd w = {bgd.w}")

    print()

    sgd = StochasticGradientDescentClassifier(0.001)
    sgd.train(np.matrix(Xtrain), np.array(Ytrain), threshold=1e-6)

    predictions = [sgd.predict(input) for input in Xtrain]
    incorrect = len(Ytrain) - np.sum(np.isclose(np.array(predictions), np.array(Ytrain), atol=1e-6))
    print(f"incorrect: {incorrect/len(Ytrain)}")
    print(f"predictions: {predictions}")
    print(f"correct labels: {Ytest}")
    print(f"sgd w = {sgd.w}")

