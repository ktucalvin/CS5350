import datasets
from math import exp
import numpy as np
Xtrain, Ytrain, Xtest, Ytest, attributes = datasets.get_bank_note_data()

X = Xtrain[:,1:]
Y = Ytrain
# gamma = 1

# xsum = 0
# for xi in X:
#     for xj in X:
#         xsum += exp(- (np.linalg.norm(xi - xj) ** 2) / gamma)

# print(f"Manual rbf = {xsum}")

# from scipy.spatial.distance import pdist, squareform
#   # this is an NxD matrix, where N is number of items and D its dimensionalites
# pairwise_dists = squareform(pdist(X, 'euclidean'))
# K = np.exp(-pairwise_dists ** 2 / gamma)
# print(f"vecsum = {np.sum(K)}")

print(X.shape)
N = X.shape[0]
A = np.random.rand(N)
Y = Y.reshape((-1))

total = 0
for i in range(N):
    for j in range(N):
        total += Y[i] * Y[j] * A[i] * A[j] * X[i].T @ X[j]
total *= 0.5
total -= np.sum(A)

print(f"looped sum: {total}")
# print(f"squared dot product sum: {0.5 * (np.sum(Y @ A) ** 2) + np.sum(X @ X.T) - np.sum(A)}")
print(f"transposed dot product sum: {0.5 * np.sum(np.sum(np.outer(Y, Y.T) * np.outer(A, A.T)) * X @ X.T) - np.sum(A)}")

outsum = np.outer(A * Y * X.T, (A * Y * X.T).T)
# outsum = (A * Y * X.T) @ (A * Y * X.T).T

print(f"outer product sum: {0.5 * np.sum(outsum) - np.sum(A)}")
