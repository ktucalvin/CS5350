import datasets
from math import exp
import numpy as np
from scipy.spatial.distance import pdist, squareform
Xtrain, Ytrain, Xtest, Ytest, attributes = datasets.get_bank_note_data()

X = Xtrain[:,1:]
Y = Ytrain.reshape((-1))

N = X.shape[0]
# N = 10
# d = 5
# X = np.random.rand(N, d)
# Y = np.random.rand(N)
A = np.random.rand(N)
# Y = Y.reshape((-1))

# X = np.array([
#     [1,2],
#     [1,2],
# ])
# Y = np.array([1,2])
# A = np.array([1,2])
# N = X.shape[0]

# ===== Trying to vectorize linear kernel
print("Linear kernel")
# total = 0
# for i in range(N):
#     for j in range(N):
#         total += Y[i] * Y[j] * A[i] * A[j] * X[i].T @ X[j]
# total *= 0.5
# total -= np.sum(A)
# print(f"index-looped sum: {total}")

# total = 0
# for i in range(N):
#     for j in range(N):
#         total += A[i] * A[j] * X[i].T @ X[j]
# total *= np.sum(Y) ** 2
# total *= 0.5
# total -= np.sum(A)
# print(f"vectorized sum?: {0.5 * np.sum(np.outer(Y*A, Y*A) * (X @ X.T).T) - np.sum(A)}")

# not working attempts:
# total = 0
# for yi in Y:
#     for yj in Y:
#         for ai in A:
#             for aj in A:
#                 for xi in X:
#                     for xj in X:
#                         total += yi * yj * ai * aj * xi.T @ xj
# total *= 0.5
# total -= np.sum(A)
# print(f"iterator-looped sum: {total}")
# print(f"sum outer product sum: {0.5 * np.sum(np.outer(A, A)) * np.sum(np.outer(Y, Y)) * np.sum(X @ X.T) - np.sum(A)}")
# sum_i ai*aj*yi*yj*xi@xj = sum_i [ai*yi*xi] * sum_j [aj*yj*xj]
# outsum = np.outer(A * Y * X.T, A * Y * X.T)
# outsum = (A * Y * X.T) @ (A * Y * X.T).T
# print(f"outer product sum: {0.5 * np.sum(outsum) - np.sum(A)}")
# print(f"a-y split w/ x: {0.5 * np.sum(np.outer(A * Y, A * Y) * np.sum(X @ X.T)) - np.sum(A)}")
# print(f"square sum: {0.5 * (np.sum(np.outer(A, A)) ** 2) * (np.sum(np.outer(Y, Y)) ** 2) * np.sum(X @ X.T) - np.sum(A)}")


# ====== Trying to vectorize Gaussian kernel
print("Gauss kernel")
gamma = 0.1
def rbf(x, z):
    retval = np.exp(-1/gamma * np.linalg.norm(x - z) ** 2)
    return retval
total = 0
for i in range(N):
    for j in range(N):
        total += Y[i] * Y[j] * A[i] * A[j] * rbf(X[i], X[j])
total *= 0.5
total -= np.sum(A)
print(f"index-looped sum: {total}")

dists = squareform(pdist(X, "euclidean"))
dists = np.exp(-1/gamma * dists ** 2)
print(f"Vectorized sum?: {0.5 * np.sum(np.outer(Y*A, Y*A) * dists.T) - np.sum(A)}")
