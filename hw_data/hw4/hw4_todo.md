## Problem 2

2a) use multiplied learning rate, for each C report train + test error

2b) use simple schedule, for each C report train + test error

2c) For each C, report the difference beteen the model parameters learned and the differences between the training/test errors.

## Problem 3

3a) Run dual SVM learning with C in 100/873, 500/873, 700/873. Recover the feature weights w and the bias b. Compare with
parameters learned in 2

3b) Use Gaussian kernel. Use gamma from 0.1, 0.5, 1, 5, 100 and hyperparameter C from 100/873, 500/873, 700/873
List the train/test errors for all combinations of gamma and C. What is the best combination?
Compare to linear SVM with the same settings of C

3c) For each setting of gamma and C, list the number of support vectors. When C = 500/873, report the number of overlapped
support vectors between consecutive values of gamma (i.e. how many support vectors are the same for each gamma).
What do you conclude?

SVM, Primal simple schedule, C=0.1145475372279496
Test error: 0.01000
Train Error: 0.00917
w' = (0.69610,-0.67586,-0.39682,-0.47408,-0.07268)

SVM, Primal multiplied schedule, C=0.1145475372279496
Test error: 0.01200
Train Error: 0.01261
w' = (0.69610,-0.67586,-0.39682,-0.47408,-0.07268)

SVM, Dual linear kernel, C=0.1145475372279496
Test error: 0.27600
Train Error: 0.27523
Support vectors: 872

C:\Users\Khang\anaconda3\envs\py39\lib\site-packages\scipy\optimize\optimize.py:282: RuntimeWarning: Values in x were outside bounds during a minimize step, clipping to bounds
  warnings.warn("Values in x were outside bounds during a "
SVM, Dual gaussian kernel, gamma=0.1, C=0.1145475372279496
Test error: 0.27200
Train Error: 0.27867
Support vectors: 872

SVM, Dual gaussian kernel, gamma=0.5, C=0.1145475372279496
Test error: 0.27000
Train Error: 0.27179
Support vectors: 872

SVM, Dual gaussian kernel, gamma=1, C=0.1145475372279496
Test error: 0.27600
Train Error: 0.27523
Support vectors: 872

SVM, Dual gaussian kernel, gamma=5, C=0.1145475372279496
Test error: 0.27000
Train Error: 0.26835
Support vectors: 872

SVM, Dual gaussian kernel, gamma=100, C=0.1145475372279496
Test error: 0.27800
Train Error: 0.27752
Support vectors: 872

SVM, Primal simple schedule, C=0.572737686139748
Test error: 0.00800
Train Error: 0.00917
w' = (1.23904,-1.01246,-0.68632,-0.76845,-0.07852)

SVM, Primal multiplied schedule, C=0.572737686139748
Test error: 0.00800
Train Error: 0.00803
w' = (1.23904,-1.01246,-0.68632,-0.76845,-0.07852)

SVM, Dual linear kernel, C=0.572737686139748
Test error: 0.26000
Train Error: 0.26376
Support vectors: 872

SVM, Dual gaussian kernel, gamma=0.1, C=0.572737686139748
Test error: 0.26000
Train Error: 0.26950
Support vectors: 872

SVM, Dual gaussian kernel, gamma=0.5, C=0.572737686139748
Test error: 0.26000
Train Error: 0.27064
Support vectors: 872

SVM, Dual gaussian kernel, gamma=1, C=0.572737686139748
Test error: 0.28400
Train Error: 0.28211
Support vectors: 872

SVM, Dual gaussian kernel, gamma=5, C=0.572737686139748
Test error: 0.27000
Train Error: 0.27408
Support vectors: 872

SVM, Dual gaussian kernel, gamma=100, C=0.572737686139748
Test error: 0.27400
Train Error: 0.28096
Support vectors: 872

SVM, Primal simple schedule, C=0.8018327605956472
Test error: 0.02800
Train Error: 0.02867
w' = (1.40872,-1.16576,-0.86407,-0.67149,-0.14216)

SVM, Primal multiplied schedule, C=0.8018327605956472
Test error: 0.01000
Train Error: 0.00917
w' = (1.40872,-1.16576,-0.86407,-0.67149,-0.14216)

SVM, Dual linear kernel, C=0.8018327605956472
Test error: 0.27800
Train Error: 0.27752
Support vectors: 872

SVM, Dual gaussian kernel, gamma=0.1, C=0.8018327605956472
Test error: 0.25600
Train Error: 0.26261
Support vectors: 872

SVM, Dual gaussian kernel, gamma=0.5, C=0.8018327605956472
Test error: 0.27200
Train Error: 0.27638
Support vectors: 872

SVM, Dual gaussian kernel, gamma=1, C=0.8018327605956472
Test error: 0.26800
Train Error: 0.27294
Support vectors: 872

SVM, Dual gaussian kernel, gamma=5, C=0.8018327605956472
Test error: 0.27200
Train Error: 0.27064
Support vectors: 872

SVM, Dual gaussian kernel, gamma=100, C=0.8018327605956472
Test error: 0.27400
Train Error: 0.27408
Support vectors: 872
