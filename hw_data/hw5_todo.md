## Problem 2

a) Implement backpropagation, verify against paper problem 3

b) Implement SGD to learn the weights of the NN

  - Use learning rate schedule gamma_t = gamma_0 / (1 + gamma_0/d * t)
  - Initialize edge weights with random numbers from normal distribution
  - Vary layer width from [5, 10, 25, 50, 100]
  - Tune gamma_0 and d to ensure convergence
  - Shuffle when performing SGD
  - Report train/test error for each layer width

c) Same as above but with weights initialized to zero

d) Compare empirically the NN, Logistic Regression, and NN performance

e) Use PyTorch to train a neural network

  - Use tanh+Xavier and RELU+He activations and initialization strategies
  - Vary depth from 3, 5, 9
  - Vary width from [5, 10, 25, 50, 100]
  - Use Adam optimizer with initial learning rate 10e-3
  - Report train/test error

## Problem 3

a) Obtain MAP estimation

  - Try prior variance v from [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]
  - Use learning rate schedule gamma_t = gamma_0 / (1 + gamma_0/d * t)
  - Report train/test error for each setting of variance v

b) Obtain the ML estimation

  - Don't assume a prior
  - Use the same learning rate schedule
  - Report train/test error for each setting of variance v

c) Compare train/test performance between MAP estimation and MLE. What do you think of *v* as compared to the
hyperparameter *C* used in SVMs

## Problem 4

How do you like the ML library?

