import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import PolynomialFeatures
import tensorflow as tf

np.set_printoptions(precision=3, suppress=True)

# Feature Approximation Methods
def poly_features(state: np.array, degree: int) -> np.array:
    poly = PolynomialFeatures(degree)
    return poly.fit_transform(state)

def rbf_features(state, centers, sigmas):
    """
    Returns the RBF feature representation for a given state.
    `state`: (N, 2) where N is the number of data points and 2 corresponds to the 2D state.
    `centers`: (M, 2) where M is the number of RBF centers.
    `sigmas`: (2,) or (M, 2) array specifying the widths of the RBFs for each dimension.
    """
    dists = np.linalg.norm(state[:, None, :] - centers[None, :, :], axis=-1)  # (N, M)
    if sigmas.shape == (2,):
        sigmas = np.ones_like(dists) * sigmas.mean()  # Example case, adjust if needed
    return np.exp(-0.5 * (dists / sigmas)**2)

# Neural network for Q-function approximation
def create_q_network(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)  # Output is the Q-value for the continuous action
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Q-learning update for continuous actions
def update_q_function(state, action, reward, next_state, model, gamma):
    action = np.reshape(action, (1,))  # Ensure action is 1D array

    # Get the current Q-value
    q_values = model.predict(np.array([np.concatenate((state, action))]))

    # Compute target Q-value using TD update
    next_q_values = model.predict(np.array([np.concatenate((next_state, action))]))
    target_q = reward + gamma * np.max(next_q_values)  # Max over next state's actions

    # Update the model
    q_values[0] = target_q  # Update the Q-value for the given state-action pair
    model.fit(np.array([np.concatenate((state, action))]), q_values, verbose=0)

# Load the data
data = np.load("a6_gridworld.npz")
s = data["s"]  # Current states
a = data["a"]  # Actions taken (note: actions must be continuous)
r = data["r"]  # Rewards received
s_next = data["s_next"]  # Next states
Q = data["Q"]  # True Q-values (from the dataset)
term = data["term"]  # Terminal state indicator
n = s.shape[0]
n_actions = 5  # Assuming 5 actions, adjust if needed
gamma = 0.99

# Set parameters
max_iter = 10000
alpha = 1e-3
thresh = 1e-8

degree = 2  # Example polynomial degree, adjust as necessary
centers = np.array([[i, j] for i in range(9) for j in range(9)])  # RBF centers
sigmas = np.array([1.0, 1.0])  # RBF sigmas
widths = np.array([1.0, 1.0])  # Coarse coding widths
offsets = [0]  # Tile coding offsets

# Choose feature approximation method
# Uncomment one of the following to test different feature approximations
name, get_phi = "Poly", lambda state: poly_features(state, degree)
name, get_phi = "RBFs", lambda state: rbf_features(state, centers, sigmas)
# name, get_phi = "Coarse", lambda state: coarse_features(state, centers, widths, offsets)
# name, get_phi = "Tiles", lambda state: tile_features(state, centers, widths, offsets)
# name, get_phi = "Aggreg.", lambda state: aggregation_features(state, centers)

# Initialize features
phi = get_phi(s)            # shape: (n_samples, n_features)

# Create Q-network
input_shape = (phi.shape[1] + 1,)  # State features + action dimension
q_model = create_q_network(input_shape)

# Semi-gradient TD learning loop for Q-function approximation
pbar = tqdm(total=max_iter)
for iter in range(max_iter):
    for i in range(n):  # Iterate through each sample
        # Get the current state, action, reward, and next state
        state = s[i]
        action = a[i]  # Continuous action
        reward = r[i]
        next_state = s_next[i]

        # Update Q-function
        update_q_function(state, action, reward, next_state, q_model, gamma)

    # Compute MSE between prediction and true Q-values
    q_pred = np.array([q_model.predict(np.array([np.concatenate((s[j], a[j]))])) for j in range(n)])
    mse = np.mean((q_pred.flatten() - Q.flatten()) ** 2)
    pbar.set_description(f"MSE: {mse:.6f}")
    pbar.update()

    # Early stopping if MSE converges
    if mse < thresh:
        break

# Display results
print(f"Iterations: {iter}, MSE: {mse:.6f}")

# Visualizing the learned Q-function vs the true Q-function for each action
fig, axs = plt.subplots(2, n_actions, figsize=(15, 10))
for i in range(n_actions):
    # For continuous actions, visualization may require specific adjustments based on your application
    axs[0, i].imshow(Q[:, i].reshape(9, 9))  # True Q-values (reshaped)
    axs[0, i].set_title(f"True Q {i}")
    axs[1, i].imshow(q_pred.reshape(9, 9))  # Predicted Q-values (reshaped)
    axs[1, i].set_title(f"Approx. Q {i} (MSE {mse:.3f})")

plt.show()
