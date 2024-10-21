import numpy as np

# Example values for the inputs
n_features = 4
n_actions = 2
n_samples = 5

# Create random feature vectors for 3 samples (n_samples, n_features)
phi = np.random.randn(n_samples, n_features)
print(phi.shape)

# Initialize random weights (n_features, n_actions)
weights = np.random.randn(n_features, n_actions)

# Actions for 3 samples (n_samples, n_actions)
actions = np.random.randn(n_samples, n_actions)
print(actions.shape)

# Standard deviation (scalar for all actions)
sigma = 1.0

# Define the dlog_gaussian_probs function
def dlog_gaussian_probs(phi: np.array, weights: np.array, sigma: float, actions: np.array) -> np.array:
    # Compute mean for Gaussian
    mu = phi @ weights  # Shape (n_samples, n_actions)

    # Compute the inverse covariance (for a diagonal covariance, it's just 1/sigma)
    inv_sigma = 1 / sigma  # Scalar value

    # Reshape to align dimensions for broadcasting
    phi = phi[:, :, np.newaxis]  # Now (n_samples, n_features, 1)
    actions = actions[:, np.newaxis, :]  # Now (n_samples, 1, n_actions)
    mu = mu[:, np.newaxis, :]  # Shape (n_samples, 1, n_actions) for consistency

    # Calculate the gradient of log probability
    # Broadcasting shapes:
    # - (actions - mu) is (n_samples, 1, n_actions)
    # - phi is (n_samples, n_features, 1)
    # - Output will be (n_samples, n_features, n_actions)
    dlog_pi = inv_sigma * (actions - mu) * phi  # Element-wise multiplication

    return dlog_pi  # Shape: (n_samples, n_features, n_actions)

# Call the function with the sample data
dlog_pi = dlog_gaussian_probs(phi, weights, sigma, actions)
print(dlog_pi.shape)
# Compute gradients with returns
T = n_samples
G = np.random.randn(T)
gradient = np.zeros_like(dlog_pi)  # Shape: (T, n_features, n_actions)
for t in range(T):
    G_t = float(G[t])  # Ensure G[t] is a scalar
    gradient[t] = dlog_pi[t] * G_t  # Element-wise product

weights = np.zeros((phi.shape[1], n_actions))
print(weights)
gradient_mean = gradient.mean(0)

# Print the result
# print("Gradient of log-probabilities (dlog_pi):")
# print(dlog_pi)

# print(gradient_mean)
alpha = 0.1
weights += alpha * gradient_mean
print(weights)

weights += alpha * gradient_mean
print(weights)







# def reinforce(baseline="none"):
#     weights = np.zeros((phi_dummy.shape[1], action_dim))
#     sigma = 1.0  # for Gaussian
#     eps = 1.0  # softmax temperature, DO NOT DECAY
#     tot_steps = 0
#     exp_return_history = np.zeros(max_steps)
#     exp_return = expected_return(env_eval, weights, gamma, episodes_eval)
#     pbar = tqdm(total=max_steps)

#     while tot_steps < max_steps:
#         # Collect data
#         data = collect_data(env, weights, sigma, n_episodes=10)  # Collect data from the environment
#         r = np.vstack(data["r"])  # Stack returns
#         G = np.zeros(r.shape[0])  # Initialize the MC returns
#         T = len(r)  # T is the total number of steps collected across all episodes

#         # Initialize dlog_pi
#         dlog_pi = np.zeros((T, len(weights)))  # Initialize gradients array
#         baseline_value = 0

#         # Update weights with baseline
#         if baseline == "none":
#             baseline_value = 0
#         elif baseline == "mean_return":
#             baseline_value = G.mean()  # Compute constant baseline
#         elif baseline == "optimal":
#             # Compute the optimal baseline here
#             numerator = np.sum((dlog_pi ** 2) * G[:, np.newaxis], axis=0)  # Sum over time
#             denominator = np.sum(dlog_pi ** 2, axis=0)  # Sum of squared gradients
            
#             # Compute the optimal baseline, avoiding division by zero
#             baseline_value = numerator / denominator if np.sum(denominator) != 0 else 0

#         # Normalize returns
#         G -= baseline_value

#         # Compute MC return (Monte Carlo return)
#         cumulative_sum = 0
#         for t in reversed(range(T)):
#             if data["done"][t]:  # Check if the episode is done
#                 cumulative_sum = 0  # Reset cumulative sum at episode end
#             cumulative_sum = gamma * cumulative_sum + r[t].item()  # Ensuring r[t] is scalar
#             # print(cumulative_sum)
#             G[t] = cumulative_sum  # Ensuring cumulative_sum is scalar
        
#         # Compute gradient of all samples (with/without baseline)
#         dlog_pi = np.zeros((T, len(weights)))  # Initialize gradients
#         for t in range(T):
#             phi = data["phi"][t]  # Get feature vector
#             temp = dlog_gaussian_probs(phi, weights, sigma, data["a"][t])  # Compute gradient
#             dlog_pi[t] = temp.ravel()  # Flatten the (343, 1) array to (343,)
        
#         # Compute gradients with returns
#         # gradient = np.zeros_like(weights)
#         gradient = np.zeros(weights.shape[0])  # Change gradient to be a 1D array with shape (343,)
#         # gradient = np.zeros((T, len(weights)))
#         # gradient = np.zeros((T, phi.shape[1], action_dim))  # Shape: (T, n_features, n_actions)


#         for t in range(T):
#             G_t = float(G[t])  # Ensure G[t] is a scalar
#             temp2 = dlog_pi[t] * G_t
#             gradient += temp2  # Accumulate gradient

#         # Average gradient over all samples
#         # gradient /= T
        
#         # Update weights
#         # weights += alpha * gradient
#         # weights += alpha * gradient[:, np.newaxis]  # Reshape gradient to (343, 1) for updating
#         weights += alpha * gradient.mean(0)

#         # Update return history and total steps
#         exp_return_history[tot_steps : tot_steps + T] = exp_return 
#         tot_steps += T

#         # Evaluate the policy
#         exp_return = expected_return(env_eval, weights, gamma, episodes_eval)

#         # Decay sigma
#         sigma = max(sigma - T / max_steps, 0.1)

#         # Progress bar update
#         pbar.set_description(f"G: {exp_return:.3f}")
#         pbar.update(T)

#     pbar.close()
#     return exp_return_history




# phi_1 = np.zeros((1,10))
# phi_1[0,0] = 1.0
# weights_1 = np.zeros((10,5))
# act_1 = np.array([[1]])
# dlog_softmax_probs(phi_1, weights_1, 1.0, act_1)
# print(dlog_softmax_probs)








# --------------------------------------------------------------------

# def dlog_gaussian_probs(phi, weights, sigma, action: np.array):
#     # implement log-derivative of pi with respect to the mean only

# def dlog_gaussian_probs(phi: np.array, weights: np.array, sigma: float, action: np.array) -> np.array:
    # # implement log-derivative of pi for Gaussian policy
    # # For Gaussian policy: π(a|s) = N(μ(s), σ²)
    # # where μ(s) = φ(s)ᵀw
    
    # # Calculate mean of the Gaussian (μ = φᵀw)
    # mu = np.dot(phi, weights)
    
    # # For Gaussian policy, the log derivative is:
    # # ∇w log π(a|s) = (a - μ)φ / σ²
    
    # # Calculate the difference between action and mean
    # diff = action - mu
    
    # # Calculate gradient
    # # Note: phi is the feature vector, and we multiply it by diff/sigma²
    # # This gives us the gradient with respect to the weights
    # grad = phi * (diff / (sigma**2))
    
    # return grad

# def dlog_softmax_probs(phi, weights, eps, act):
#     # implement log-derivative of pi

# def dlog_softmax_probs(phi, weights, eps, act):
    # # implement log-derivative of pi for softmax policy
    # # weights shape: (n_features, n_actions)
    # # phi shape: (n_features,)
    # # act: scalar (action index)
    
    # # Calculate logits (θᵀφ)
    # logits = np.dot(phi, weights)  # shape: (n_actions,)
    
    # # Calculate softmax probabilities
    # # pi(a|s) = exp(θᵀφ) / Σ exp(θᵀφ)
    # exp_logits = np.exp(logits - np.max(logits))  # subtract max for numerical stability
    # probs = exp_logits / np.sum(exp_logits)
    
    # # Calculate gradient of log pi(a|s)
    # # ∇θ log π(a|s) = φ(1[a=a'] - π(a'|s))
    # grad = np.zeros_like(weights)  # shape: (n_features, n_actions)
    
    # # For each feature
    # for i in range(len(phi)):
    #     # Gradient is feature * (1 for chosen action - probability)
    #     grad[i] = phi[i] * (-probs)  # Fill with -probs first
    #     grad[i, act] += phi[i]  # Add 1 for the chosen action
    
    # return grad



# def reinforce(baseline="none"):
#     weights = np.zeros((phi_dummy.shape[1], action_dim))
#     sigma = 1.0  # for Gaussian
#     eps = 1.0  # softmax temperature, DO NOT DECAY
#     tot_steps = 0
#     exp_return_history = np.zeros(max_steps)
#     exp_return = expected_return(env_eval, weights, gamma, episodes_eval)
#     pbar = tqdm(total=max_steps)

#     while tot_steps < max_steps:
#         # collect data
#         data = collect_data(env, weights, sigma, n_episodes=10)  # Collect data from the environment
#         r = np.vstack(data["r"])  # Stack returns
#         G = np.zeros(r.shape[0])  # Initialize the MC returns
#         T = len(r)  # T is the total number of steps collected across all episodes


#         # Update weights with baseline
#         if baseline == "none":
#             baseline_value = 0
#         elif baseline == "average":
#             baseline_value = G.mean()
#         elif baseline == "optimal":
#             baseline_value = ...  # Compute optimal baseline here

#         # Normalize returns
#         G -= baseline_value

#         # compute MC return
#         cumulative_sum = 0
#         for t in reversed(range(len(r))):
#             if data["done"][t]:  # Check if the episode is done
#                 cumulative_sum = 0  # Reset cumulative sum at episode end
#             cumulative_sum = gamma * cumulative_sum + r[t].item()  # Ensuring r[t] is scalar
#             G[t] = cumulative_sum.item()  # Ensuring cumulative_sum is scalar
        
#         # compute gradient of all samples (with/without baseline)
#         dlog_pi = np.zeros((len(r), len(weights)))  # Initialize gradients
#         for t in range(len(r)):
#             phi = data["phi"][t]  # Get feature vector
#             dlog_pi[t] = dlog_gaussian_probs(phi, weights, sigma, data["a"][t])  # Compute gradient
        
#         # Ensure G is scalar by extracting its first element (if necessary)
#         for t in range(len(r)):
#             G_t = float(G[t])  # Ensure G[t] is a scalar
#             gradient += dlog_pi[t] * G_t  # Compute the gradient sum

#         # average gradient over all samples
        
#         # update weights

#         # Finally, update weights
#         weights += alpha * gradient / len(r)



#         T = ... # steps taken while collecting data
#         exp_return_history[tot_steps : tot_steps + T] = exp_return 
#         tot_steps += T
#         exp_return = expected_return(env_eval, weights, gamma, episodes_eval)
#         sigma = max(sigma - T / max_steps, 0.1)

#         pbar.set_description(
#             f"G: {exp_return:.3f}"
#         )
#         pbar.update(T)

#     pbar.close()
#     return exp_return_history
