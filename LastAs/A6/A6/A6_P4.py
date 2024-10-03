import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import PolynomialFeatures

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

def tile_features(state: np.array, centers: np.array, widths: np.array, offsets: list = [0]) -> np.array:
    activations = np.zeros((state.shape[0], centers.shape[0]))
    for offset in offsets:
        shifted_centers = centers + offset
        in_tile = np.all(np.abs(state[:, np.newaxis] - shifted_centers) <= widths / 2, axis=2)
        activations += in_tile
    return activations / len(offsets)

def coarse_features(state, centers, widths, offsets):
    """
    Returns the coarse coding feature representation for a given state.
    `state`: (N, 2) where N is the number of data points and 2 corresponds to the 2D state.
    `centers`: (M, 2) where M is the number of coarse coding centers.
    `widths`: (2,) array specifying the widths of the receptive fields for each dimension.
    `offsets`: (M,) array specifying the offsets for each feature center.
    """
    dists = np.abs(state[:, None, :] - centers[None, :, :])  # (N, M, 2)
    widths = widths.reshape(1, 1, -1)  # Reshape to broadcast correctly
    in_field = np.all(dists <= widths, axis=-1)  # (N, M)
    return in_field.astype(float)

def aggregation_features(state: np.array, centers: np.array) -> np.array:
    dists = np.linalg.norm(state[:, np.newaxis] - centers, axis=2)
    closest = np.argmin(dists, axis=1)
    activations = np.zeros((state.shape[0], centers.shape[0]))
    activations[np.arange(state.shape[0]), closest] = 1
    return activations

# Load the data
data = np.load("a6_gridworld.npz")
s = data["s"]  # Current states
a = data["a"]  # Actions taken
r = data["r"]  # Rewards received
s_next = data["s_next"]  # Next states
Q = data["Q"]  # True Q-values (from the dataset)
term = data["term"]  # Terminal state indicator
n = s.shape[0]
n_actions = 5
gamma = 0.99

# Display true Q-functions for each action as heatmaps
s_idx = np.ravel_multi_index(s.T, (9, 9))
unique_s_idx = np.unique(s_idx, return_index=True)[1]

# Display the true Q-function as heatmaps for each action
# fig, axs = plt.subplots(1, n_actions, figsize=(15, 5))
# for i, action in enumerate(["LEFT", "DOWN", "RIGHT", "UP", "STAY"]):
#     axs[i].imshow(Q[unique_s_idx, i].reshape(9, 9))
#     axs[i].set_title(f"True Q {action}")
# plt.show()

# Set parameters
max_iter = 10000
alpha = 1
thresh = 1e-8

degree = 2  # Example polynomial degree, adjust as necessary
centers = np.array([[i, j] for i in range(9) for j in range(9)])  
sigmas = np.array([1.0, 1.0])  
widths = np.array([1.0, 1.0])  
offsets = [0]  

# Pick one feature approximation method
# Uncomment one of the following to test different feature approximations
name, get_phi = "Poly", lambda state: poly_features(state, degree)
name, get_phi = "RBFs", lambda state: rbf_features(state, centers, sigmas)
name, get_phi = "Coarse", lambda state: coarse_features(state, centers, widths, offsets)
# name, get_phi = "Tiles", lambda state: tile_features(state, centers, widths, offsets)
# name, get_phi = "Aggreg.", lambda state: aggregation_features(state, centers)

if name == "Poly":
    alpha = 1e-6
    max_iter = 100000

# Initialize features and weights
phi = get_phi(s)            # shape: (n_samples, n_features)
phi_next = get_phi(s_next)  # shape: (n_samples, n_features)

# Initialize weights with correct shape
weights = np.zeros((phi.shape[1], n_actions))  

# Semi-gradient TD learning loop for Q-function approximation
pbar = tqdm(total=max_iter)
for iter in range(max_iter):
    V_s_next = np.max(np.dot(phi_next, weights), axis=1) * (1 - term)     
    Q_sa = np.dot(phi, weights)  # Shape: (n_samples, n_actions)    
    Q_sa = Q_sa[np.arange(len(a)), a]  # Shape: (n_samples,)
    
    # TD error calculation
    td_error = r + gamma * V_s_next - Q_sa
    
    # Update weights for each action
    for i in range(n_actions):
        mask = (a == i)  # Only update for actions taken
        weights[:, i] += alpha * np.sum(td_error[mask][:, None] * phi[mask], axis=0) / np.sum(mask)

    # Compute MSE between prediction and true Q-values
    q_pred = np.dot(phi, weights)
    mse = np.mean((q_pred - Q) ** 2)
    pbar.set_description(f"TDE: {td_error.mean():.4f}, MSE: {mse:.6f}")
    pbar.update()

    # Early stopping if MSE converges
    if mse < thresh:
        break

# Display results
print(f"Iterations: {iter}, MSE: {mse:.6f}, N. of Features {weights.shape[0]}")

# Visualizing the learned Q-function vs the true Q-function for each action
fig, axs = plt.subplots(2, n_actions, figsize=(15, 10))
td_prediction = np.dot(phi, weights)

for i, action in enumerate(["LEFT", "DOWN", "RIGHT", "UP", "STAY"]):
    axs[0][i].imshow(Q[unique_s_idx, i].reshape(9, 9))
    axs[0][i].set_title(f"True Q {action}")
    axs[1][i].imshow(td_prediction[unique_s_idx, i].reshape(9, 9))
    axs[1][i].set_title(f"Approx. Q {action} (MSE {mse:.3f})")

plt.show()