import numpy as np
import datasets
from Perceptron import perceptron

Xtrain, Ytrain, Xtest, Ytest, attributes = datasets.get_bank_note_data()

def evaluate(model, variant):
    predictions = np.array([model.predict(x) for x in Xtest]).reshape(-1,1)
    incorrect = np.sum(predictions != Ytest)
    print(f"{variant} perceptron error: {float(incorrect) / len(Ytest) :.3f}")
    
def vec2str(vector):
    return f"({','.join([f'{x :.5f}' for x in vector])})"

standard_model = perceptron.StandardPerceptron(0.1)
standard_model.train(Xtrain, Ytrain, epochs=10)
evaluate(standard_model, "Standard")
print(vec2str(standard_model.w))

voted_model = perceptron.VotedPerceptron(0.1)
voted_model.train(Xtrain, Ytrain, epochs=10)
evaluate(voted_model, "Voted")
print(f"Got {len(voted_model.weights)} weights")
for w,c in voted_model.weights:
    print(f"{c}\t{vec2str(w)}")

averaged_model = perceptron.AveragedPerceptron(0.1)
averaged_model.train(Xtrain, Ytrain, epochs=10)
evaluate(averaged_model, "Averaged")
print(vec2str(averaged_model.a))
