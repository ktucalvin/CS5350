# :NOTE: should use hw2 part 1 q5 as debug dataset


# === BATCH GRADIENT DESCENT

# Initialize w_0
# For t = 0, 1, 2, ... (until below threshold)
  # Compute gradient of J(w) at (w), call it dJdw
    # count errors through all examples, then scale  each x_i by errors and sum
    # ^ sounds like perceptron
  # Update w_{t+1} = w_t - r*dJdw

# === STOCHASTIC GRADIENT DESCENT

# Initialize w_0
# For t = 0, 1, 2, ... (until below threshold)
  # For each training example (x_i, y_i)
  # w_j^{t+1} = w_j^t + r(y_i - w^Tx_i)x_ij
  # w^{t+1} = w^t + r(y_i - w^T x_i)x_i