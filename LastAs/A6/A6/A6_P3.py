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
    # Calculate the distance between the state and the centers
    dists = np.linalg.norm(state[:, None, :] - centers[None, :, :], axis=-1)  # (N, M)

    # Apply sigmas element-wise
    if sigmas.shape == (2,):
        # Apply different sigmas for each dimension
        sigmas = np.ones_like(dists) * sigmas.mean()  # Example case, adjust if needed
    
    # Compute RBF features
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
    # Calculate the distances between the state and the centers
    dists = np.abs(state[:, None, :] - centers[None, :, :])  # (N, M, 2)

    # Apply different widths for each dimension
    widths = widths.reshape(1, 1, -1)  # Reshape to broadcast correctly
    in_field = np.all(dists <= widths, axis=-1)  # (N, M)

    # Return features as binary indicators (1 if within the receptive field, else 0)
    return in_field.astype(float)

def aggregation_features(state: np.array, centers: np.array) -> np.array:
    dists = np.linalg.norm(state[:, np.newaxis] - centers, axis=2)
    closest = np.argmin(dists, axis=1)
    activations = np.zeros((state.shape[0], centers.shape[0]))
    activations[np.arange(state.shape[0]), closest] = 1
    return activations


# Semi-gradient TD Prediction with a Feature Approximation
# Load the data
data = np.load("a6_gridworld.npz")
s = data["s"]
a = data["a"]
r = data["r"]
s_next = data["s_next"]
Q = data["Q"]
V = data["Q"].max(-1)  # value of the greedy policy
term = data["term"]
n = s.shape[0]
n_states = 81
n_actions = 5
gamma = 0.99

# Display the true V-function as a heatmap
s_idx = np.ravel_multi_index(s.T, (9, 9))
unique_s_idx = np.unique(s_idx, return_index=True)[1]

# fig, axs = plt.subplots(1, 1)
# surf = axs.imshow(V[unique_s_idx].reshape(9, 9))
# plt.colorbar(surf)
# plt.show()

# Set parameters
max_iter = 30000
alpha = 1e0
thresh = 1e-8

degree = 4  # Example polynomial degree, adjust as necessary
centers = np.array([[i, j] for i in range(9) for j in range(9)])  # RBF and Tiles and Aggreg centers
sigmas = np.array([1.0, 1.0])  # RBF sigmas
widths = np.array([1.0, 1.0])  # Coarse coding widths
offsets = [0]  # Tile coding offsets

# Pick one Feature Approximation method
# Uncomment one of the following to test different feature approximations
name, get_phi = "Poly", lambda state: poly_features(state, degree)
name, get_phi = "RBFs", lambda state: rbf_features(state, centers, sigmas)
name, get_phi = "Coarse", lambda state: coarse_features(state, centers, widths, offsets)
# name, get_phi = "Tiles", lambda state: tile_features(state, centers, widths, offsets)
# name, get_phi = "Aggreg.", lambda state: aggregation_features(state, centers)

# Initialize features and weights
phi = get_phi(s)
phi_next = get_phi(s_next)
weights = np.zeros(phi.shape[-1])
if name == "Poly":
    alpha = 1e-6
    max_iter = 100000

# Semi-gradient TD learning loop
pbar = tqdm(total=max_iter)
for iter in range(max_iter):
    # TD Error: δ = r + γ * V(s') - V(s)
    V_s = np.dot(phi, weights)
    V_s_next = np.dot(phi_next, weights) * (1 - term)  # If terminal, don't consider next state
    td_error = r + gamma * V_s_next - V_s
    
    # Gradient update
    weights += alpha * np.dot(td_error, phi) / n
    
    # Compute MSE between prediction and true value function
    mse = np.mean((np.dot(phi, weights) - V) ** 2)
    pbar.set_description(f"TDE: {td_error.mean():.4f}, MSE: {mse:.6f}")
    pbar.update()

    # Early stopping if MSE converges
    if mse < thresh:
        break

# Display results
print(f"Iterations: {iter}, MSE: {mse:.6f}, N. of Features {len(weights)}")

# Plot the true V-function vs the approximation
td_prediction = np.dot(phi, weights)
fig, axs = plt.subplots(1, 2)
axs[0].imshow(V[unique_s_idx].reshape(9, 9))
axs[0].set_title("True V-function")
axs[1].imshow(td_prediction[unique_s_idx].reshape(9, 9))
axs[1].set_title(f"Approx. with {name} (MSE {mse:.3f})")
plt.show()

