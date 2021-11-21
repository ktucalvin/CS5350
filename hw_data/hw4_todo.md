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
Test error: 0.01200
Train Error: 0.01147
w' = (0.68366,-0.66992,-0.44366,-0.46223,-0.08843)

SVM, Primal multiplied schedule, C=0.1145475372279496
Test error: 0.00800
Train Error: 0.00688
w' = (0.68366,-0.66992,-0.44366,-0.46223,-0.08843)

SVM, Dual linear kernel, C=0.1145475372279496
Test error: 0.27000
Train Error: 0.27408

SVM, Dual gaussian kernel, gamma=0.1, C=0.1145475372279496
Test error: 0.27000
Train Error: 0.27294

SVM, Dual gaussian kernel, gamma=0.5, C=0.1145475372279496
Test error: 0.27800
Train Error: 0.28326

SVM, Dual gaussian kernel, gamma=1, C=0.1145475372279496
Test error: 0.27600
Train Error: 0.27638

SVM, Dual gaussian kernel, gamma=5, C=0.1145475372279496
Test error: 0.26800
Train Error: 0.27523

SVM, Dual gaussian kernel, gamma=100, C=0.1145475372279496
Test error: 0.26800
Train Error: 0.27408

SVM, Primal simple schedule, C={c}
Test error: 0.01000
Train Error: 0.00917
w' = (1.28573,-0.96950,-0.73823,-0.76964,-0.06994)

SVM, Primal multiplied schedule, C={c}
Test error: 0.01200
Train Error: 0.00917
w' = (1.28573,-0.96950,-0.73823,-0.76964,-0.06994)

SVM, Dual linear kernel, C=0.572737686139748
Test error: 0.27200
Train Error: 0.27638

SVM, Dual gaussian kernel, gamma=0.1, C=0.572737686139748
Test error: 0.27200
Train Error: 0.27982

SVM, Dual gaussian kernel, gamma=0.5, C=0.572737686139748
Test error: 0.27600
Train Error: 0.28555

SVM, Dual gaussian kernel, gamma=1, C=0.572737686139748
Test error: 0.27200
Train Error: 0.28096

SVM, Dual gaussian kernel, gamma=5, C=0.572737686139748
Test error: 0.27000
Train Error: 0.27408

SVM, Dual gaussian kernel, gamma=100, C=0.572737686139748
Test error: 0.26200
Train Error: 0.26835

SVM, Primal simple schedule, C={c}
Test error: 0.00800
Train Error: 0.00803
w' = (1.39806,-1.06157,-0.76550,-0.81796,-0.17224)

SVM, Primal multiplied schedule, C={c}
Test error: 0.02200
Train Error: 0.02523
w' = (1.39806,-1.06157,-0.76550,-0.81796,-0.17224)

SVM, Dual linear kernel, C=0.8018327605956472
Test error: 0.26800
Train Error: 0.27294

SVM, Dual gaussian kernel, gamma=0.1, C=0.8018327605956472
Test error: 0.26800
Train Error: 0.27523

SVM, Dual gaussian kernel, gamma=0.5, C=0.8018327605956472
Test error: 0.27000
Train Error: 0.27294

SVM, Dual gaussian kernel, gamma=1, C=0.8018327605956472
Test error: 0.27200
Train Error: 0.27982

SVM, Dual gaussian kernel, gamma=5, C=0.8018327605956472
Test error: 0.27000
Train Error: 0.27294

SVM, Dual gaussian kernel, gamma=100, C=0.8018327605956472
Test error: 0.25800
Train Error: 0.26720
