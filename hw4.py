import numpy as np
import datasets
from SVM import svm

Xtrain, Ytrain, Xtest, Ytest, attributes = datasets.get_bank_note_data()

#     |    + - +
#     | 
#=================
#     |
# -   |


# prepend 1 for bias term
# Xtrain = np.array([
#     [1, 1, 1],
#     [1,-1,-1],
#     [1, 2, 1],
#     [1, 1.5, 1]])
# Ytrain = np.array([1,-1, 1, -1]).reshape((-1,1))

# Xtest = np.array([
#     [1, 1, 1],
#     [1, -1, -1]])
# Ytest = np.array([1,-1]).reshape((-1,1))

def evaluate(model, variant):
    predictions = np.array([model.predict(x) for x in Xtrain]).reshape(-1,1)
    incorrect_train = np.sum(predictions != Ytrain)
    predictions = np.array([model.predict(x) for x in Xtest]).reshape(-1,1)
    incorrect_test = np.sum(predictions != Ytest)
    print(f"SVM, {variant}\nTest error: {float(incorrect_test) / len(Ytest) :.5f}\nTrain Error: {float(incorrect_train) / len(Ytrain) :.5f}")
    
def vec2str(vector):
    # augmented vector
    return f"w' = ({','.join([f'{x :.5f}' for x in vector])})"

for c in [100/873, 500/873, 700/873]:
    gamma_0 = 1e-4
    a = 150

    primal_simple_model = svm.PrimalSVMClassifier(C=c, schedule=lambda t: gamma_0 / (1 + t))
    primal_simple_model.train(Xtrain, Ytrain, epochs=100)

    evaluate(primal_simple_model, f"Primal simple schedule, C={c}")
    print(vec2str(primal_simple_model.w))
    print()

    primal_multiplied_model = svm.PrimalSVMClassifier(C=c, schedule=lambda t: gamma_0 / (1 + gamma_0/a * t))
    primal_multiplied_model.train(Xtrain, Ytrain, epochs=100)
    evaluate(primal_multiplied_model, f"Primal multiplied schedule, C={c}")
    print(vec2str(primal_simple_model.w))
    print()

    dual_linear_model = svm.DualSVMClassifier(C=c, tol=1e-6)
    dual_linear_model.train(Xtrain, Ytrain, kernel="linear")
    evaluate(dual_linear_model, f"Dual linear kernel, C={c}")
    print(f"Support vectors: {len(dual_linear_model.support_vectors)}")
    print()

    for gamma in [0.1, 0.5, 1, 5, 100]:
        dual_gaussian_model = svm.DualSVMClassifier(C=c, tol=1e-6)
        dual_gaussian_model.train(Xtrain, Ytrain, gamma=gamma, kernel="gaussian")
        evaluate(dual_gaussian_model, f"Dual gaussian kernel, gamma={gamma}, C={c}")
        print(f"Support vectors: {len(dual_gaussian_model.support_vectors)}")
        print()
        # Analyze overlap later
        with open(f"hw_data/hw4/g{gamma}-c{c}", "w+", encoding="utf8") as log:
            for vec in dual_gaussian_model.support_vectors:
                log.write(vec2str(vec))
