import datasets
import numpy as np

def loss(w, X, Y):
    sse = np.sum(np.power(2, Y - w.dot(X.T)))
    return 0.5 * sse

class BatchGradientDescentClassifier():
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def train(self, X, Y, threshold=1e-6):
        with open ("bgd-results.txt", "a", encoding="utf8") as log:
            # Initialize w_0
            self.w = np.zeros(X.shape[1])
            norm = float("inf")
            
            # For t = 0, 1, 2, ... (until convergence)
            t = 0
            while norm > threshold:
                t += 1
                # Compute gradient of J(w) at (w), call it grad
                # error is PER ROW
                # grad = weighted sum of each example, where weights are each row's error

                cost = loss(self.w, X, Y)
                print(f"{t}\t{cost}")
                log.write(f"{t}\t{cost}\n")

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
        with open("sgd-results-auto-converge.txt", "a", encoding="utf8") as log:
            # Initialize w_0
            self.w = np.zeros(X.shape[1])
            norm = float("inf")

            # For t = 0, 1, 2, ... (until below threshold)
            t = 0
            while norm > threshold:
                # shuffle examples
                shuffled = np.random.permutation(X.shape[0])
                samplesX = X[shuffled]
                samplesY = Y[shuffled]

                # For each training example (x_i, y_i)
                for xi, yi in zip(samplesX, samplesY):
                    t += 1
                    # w^{t+1} = w^t + r(y_i - w^T x_i)x_i
                    cost = loss(self.w, X, Y)
                    print(f"{t}\t{cost}\t{self.w}")
                    stringw = str(self.w).replace('\n', '')
                    log.write(f"{t}\t{cost}\t{stringw}\n")
                    wnext = self.w + self.learning_rate * (yi - self.w.dot(xi.T)) * xi
                    norm = np.linalg.norm(self.w - wnext)
                    self.w = np.ravel(wnext)

    
    def predict(self, input):
        return self.w @ input

if __name__ == "__main__":
    # Xtrain, Ytrain, Xtest, Ytest, attributes = datasets.get_hw2_data()
    Xtrain, Ytrain, Xtest, Ytest, attributes = datasets.get_concrete_data()
    Xtrain = np.matrix(Xtrain)
    Xtest = np.matrix(Xtest)
    Ytrain = np.array(Ytrain)
    Ytest = np.array(Ytest)

    # this implementation will perform poorly because the learning rate isn't adjustable

    bgd = BatchGradientDescentClassifier(2 ** -8)
    bgd.train(Xtrain, Ytrain, threshold=1e-6)

    print(f"overall bgd loss on test = {loss(bgd.w, Xtest, Ytest)}")

    sgd = StochasticGradientDescentClassifier(0.001)
    sgd.train(Xtrain, Ytrain, threshold=1e-6)

    print(f"overall sgd loss on test = {loss(sgd.w, Xtest, Ytest)}")

