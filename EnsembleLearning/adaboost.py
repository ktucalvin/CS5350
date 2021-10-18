# initialize D_1(i) = 1/m for i = 1,2,...,m
# for t = 1,2,...,T:
  # Find a classifier h_t whose weighted classification error is better than chance
  # compute its vote as
    # alpha_t = 1/2 * ln(1-epsilon_t / epsilon_t)
  # Update the values of the weights for the training example
    # d_{t+1}(i) = D_t(i) / Z_t * exp(-alpha_t * y_i h_t(x_i))
# return the final hypothesis H_final(x) = sgn(sum_t alpha_t h_t(x))