import numpy as np
from numpy import linalg as LA

X = np.matrix([
  [1,  1, -1,  2],
  [1,  1,  1,  3],
  [1, -1,  1,  0],
  [1,  1,  2, -4],
  [1,  3, -1, -1],
])

Y = np.asarray([1, 4, -1, -2, 0])

# w = np.matrix([-1, 1, -1])
# b = -1

b = -1
w = np.matrix([b, 1, 1, 1])

dw = 0
db = 0
lms = 0

for xi,yi in zip(X,Y):
  xi = np.ravel(xi)
  yi = np.ravel(yi)
  # print sum (yi - w.T*x)(-x) = -sum (yi-w.T*x)(x) = sum (w.T*x - yi)(x)
  print(f"[-{yi[0]} + b + w_1({xi[0]}) + w_2({xi[1]}) + w_3({xi[2]})]({xi})")
  
  # stuff regarding derivative
  # print(f"x_i = {xi}")
  # print(f"y_i = {yi}")
  # print(f" + {np.ravel(-yi + w @ xi + b)}")
  # dw += (yi - w @ xi - b)*(-xi)
  # db += (-yi + w @ xi + b)

  lms += (yi - w @ xi)**2

lms *= 0.5

# print(f"dw: {np.ravel(dw)}")
# print(f"db: {np.ravel(db)}")
print(f"lms: {np.ravel(lms)[0]}")



# d x m
X = np.matrix([
  [1, 1, 1, 1, 1],
  [1, 1, -1, 1, 3],
  [-1, 1, 1, 2, -1],
  [2, 3, 0, -4, -1]
])

# m x 1
Y = np.matrix([[ 1],
               [ 4],
               [-1],
               [-2],
               [ 0]])

wstar = LA.inv(X*X.T)*X*Y
print(f"w* = {np.ravel(wstar)}")
