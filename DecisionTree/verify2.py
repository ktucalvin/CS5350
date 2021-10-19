import datasets
import numpy as np
from sklearn.linear_model import LinearRegression

Xtrain, Ytrain, Xtest, Ytest, attributes = datasets.get_concrete_data()
linreg = LinearRegression()

linreg.fit(Xtrain, Ytrain)
print(linreg.coef_)

predictions = linreg.predict(Xtest)
incorrect = len(Ytest) - np.sum(np.isclose(np.array(predictions), np.array(Ytest), atol=1e-6))
print(f"predictions:\n{predictions}")
print(f"correct labels:\n{Ytest}")
print(f"test error: {incorrect/len(Ytrain)}")
