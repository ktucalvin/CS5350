import numpy as np
import datasets
from SVM import svm

Xtrain, Ytrain, Xtest, Ytest, attributes = datasets.get_bank_note_data()

#     |    +   +
#     | 
#=================
#     |
# -   |


# prepend 1 for bias term
# Xtrain = np.array([[1,1,1], [1,-1,-1], [1,2,1]])
# Ytrain = np.array([1,-1,1]).reshape((-1,1))
# Xtest = np.array([[1,1,1], [1,-1,-1]])
# Ytest = np.array([1,-1]).reshape((-1,1))

def evaluate(model, variant):
    predictions = np.array([model.predict(x) for x in Xtest]).reshape(-1,1)
    incorrect = np.sum(predictions != Ytest)
    print(f"{variant} SVM Error: {float(incorrect) / len(Ytest) :.3f}")
    
def vec2str(vector):
    return f"({','.join([f'{x :.5f}' for x in vector])})"

for c in [100/873, 500/873, 700/873]:
    gamma_0 = 1e-4
    a = 150

    standard_model = svm.PrimalSVMClassifier(C=c, schedule=lambda t: gamma_0 / (1 + t))
    standard_model.train(Xtrain, Ytrain, epochs=100)

    evaluate(standard_model, "Primal simple schedule")
    # print(vec2str(standard_model.w))

    standard_model = svm.PrimalSVMClassifier(C=c, schedule=lambda t: gamma_0 / (1 + gamma_0/a * t))
    standard_model.train(Xtrain, Ytrain, epochs=100)
    evaluate(standard_model, "Primal multiplied schedule")
    # print(vec2str(standard_model.w))

