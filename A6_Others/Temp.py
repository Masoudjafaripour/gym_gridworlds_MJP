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
fig, axs = plt.subplots(1, n_actions, figsize=(15, 5))
for i, action in enumerate(["LEFT", "DOWN", "RIGHT", "UP", "STAY"]):
    axs[i].imshow(Q[unique_s_idx, i].reshape(9, 9))
    axs[i].set_title(f"True Q {action}")
plt.show()

# Set parameters
max_iter = 1000
alpha = 0.01
thresh = 1e-8

degree = 2  # Example polynomial degree, adjust as necessary
centers = np.array([[i, j] for i in range(9) for j in range(9)])  # RBF centers
sigmas = np.array([1.0, 1.0])  # RBF sigmas
widths = np.array([1.0, 1.0])  # Coarse coding widths
offsets = [0]  # Tile coding offsets

# Pick one feature approximation method
# Uncomment one of the following to test different feature approximations
name, get_phi = "Poly", lambda state: poly_features(state, degree)
name, get_phi = "RBFs", lambda state: rbf_features(state, centers, sigmas)
# name, get_phi = "Coarse", lambda state: coarse_features(state, centers, widths, offsets)
# name, get_phi = "Tiles", lambda state: tile_features(state, centers, widths, offsets)
# name, get_phi = "Aggreg.", lambda state: aggregation_features(state, centers)

# Initialize features and weights
phi = get_phi(s)            # shape: (n_samples, n_features)
phi_next = get_phi(s_next)  # shape: (n_samples, n_features)

# Initialize weights with correct shape
weights = np.zeros((phi.shape[1], n_actions))  # weights should match the feature dimensions

# Semi-gradient TD learning loop for Q-function approximation
pbar = tqdm(total=max_iter)
for iter in range(max_iter):
    # TD Error for Q: δ = r + γ * max_a' Q(s', a') - Q(s, a)
    V_s_next = np.max(np.dot(phi_next, weights), axis=1) * (1 - term)  # Max over actions

    # Compute Q(s, a) using dot product for all actions
    Q_sa = np.dot(phi, weights)  # Shape: (n_samples, n_actions)
    # Select Q-values corresponding to actions taken
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














# import numpy as np
# import gymnasium
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# from sklearn.preprocessing import PolynomialFeatures

# # Set print options for better readability of numpy arrays
# np.set_printoptions(precision=3, suppress=True)

# # Notation for array sizes:
# # - S: state dimensionality
# # - D: features dimensionality
# # - N: number of samples
# #
# # N is always the first dimension, meaning that states come in arrays of shape (N, S)
# # and features in arrays of shape (N, D).
# # We recommend to implement the functions below assuming that the input has
# # always shape (N, S) and the output (N, D), even when N = 1.

# def poly_features(state: np.array, degree: int) -> np.array:
#     """
#     Compute polynomial features. For example, if state = (s1, s2) and degree = 2,
#     the output must be 1 + s1 + s2 + s1*s2 + s1**2 + s2**2.
#     """
#     return PolynomialFeatures(degree, include_bias=True).fit_transform(state)

# def rbf_features(state: np.array, centers: np.array, sigmas: np.array) -> np.array:
#     """
#     Compute radial basis functions features: exp(- ||s - c||**2 / w**2)
#     where s is the state, c are the centers, and w are the widths (sigmas) of the Gaussians.
#     Assume sigma is the same for all dimensions.
#     """
#     # Compute the squared distances from each state to each center
#     distances = np.linalg.norm(state[:, np.newaxis] - centers, axis=2)**2
#     # Compute the RBF features
#     return np.exp(-distances / (sigmas**2))

# def tile_features(state: np.array, centers: np.array, widths: np.array) -> np.array:
#     """
#     Compute tile coding features: given centers and widths, the output will be an
#     array of 0/1, with 1s corresponding to tiles the state belongs to.
#     Assume width is the same for all dimensions.
#     """
#     n_samples = state.shape[0]
#     n_tiles = centers.shape[0]
#     features = np.zeros((n_samples, n_tiles))

#     for i in range(n_samples):
#         for j in range(n_tiles):
#             # Calculate the boundaries of the tile
#             lower_bound = centers[j] - (widths[j] / 2)
#             upper_bound = centers[j] + (widths[j] / 2)

#             # Check if the state falls within the bounds of the tile
#             if np.all(state[i] >= lower_bound) and np.all(state[i] <= upper_bound):
#                 features[i, j] = 1  # Assign tile feature
    
#     return features

# def aggregation_features(state: np.array, centers: np.array) -> np.array:
#     """
#     Compute tile coding features: given centers and widths, the output will be an
#     array of 0s and one 1 corresponding to the closest tile the state belongs to.
#     This is basically tile coding with non-overlapping tiles (a state belongs to one
#     tile only).
#     Note that we can turn this into a discrete (finite) representation of the state,
#     because we will have as many feature representations as centers.
#     """
#     n_samples = state.shape[0]
#     n_centers = centers.shape[0]
#     features = np.zeros((n_samples, n_centers))
    
#     for i in range(n_samples):
#         distances = np.linalg.norm(state[i] - centers, axis=1)
#         closest_tile_index = np.argmin(distances)  # Get index of closest center
#         features[i, closest_tile_index] = 1  # Assign feature to closest tile
    
#     return features

# # Main execution
# if __name__ == "__main__":
#     state_size = 2
#     n_samples = 10
#     n_centers = 10  # Adjust as necessary for hyperparameter tuning
#     state = np.random.rand(n_samples, state_size)  # Random states in [0, 1]

#     # Create centers for the feature functions
#     centers = np.array(
#         np.meshgrid(np.linspace(-0.2, 1.0, n_centers), np.linspace(-0.2, 1.0, n_centers))
#     ).reshape(2, -1).T

#     # Hyperparameters for RBF and tile features
#     sigmas = np.ones(n_centers * n_centers) * 0.2
#     widths = np.ones(n_centers * n_centers) * 0.2  # Initial width

#     # Compute features
#     poly = poly_features(state, 2)
#     aggr = aggregation_features(state, centers)
#     rbf = rbf_features(state, centers, sigmas)
#     tile = tile_features(state, centers, widths)

#     # Plotting the features
#     fig, axs = plt.subplots(1, 3)

#     axs[0].tricontourf(centers[:, 0], centers[:, 1], rbf[0])
#     axs[1].tricontourf(centers[:, 0], centers[:, 1], tile[0])
#     axs[2].tricontourf(centers[:, 0], centers[:, 1], aggr[0])

#     for ax, title in zip(axs, ["RBFs", "Tile", "Aggreg."]):
#         ax.plot(state[0][0], state[0][1], marker="+", markersize=12, color="red")
#         ax.set_title(title + f" for {state[0]}")

#     plt.show()

#     #################### PART 1
#     # Submit your heatmaps.
#     # What are the hyperparameters of each FA and how do they affect the shape of
#     # the function they can approximate?
#     # - In RBFs the hyperparameter(s) is/are ... More/less ... will affect ...,
#     #   while narrower/wider ... will affect ...
#     # - In tile coding the hyperparameter(s) is/are ...
#     # - In polynomials the hyperparameter(s) is/are ...
#     # - In state aggregation the hyperparameter(s) is/are ...

# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm

# # Function to fit
# def fit_function_1(x):
#     return np.sin(x) + x**2 - 0.5 * x**3 + np.log(np.abs(x))

# def fit_function_2(x):
#     y = np.zeros(x.shape)
#     y[0:10] = x[0:10]**3 / 3.0
#     y[10:20] = np.exp(x[25:35])
#     y[20:30] = -x[0:10]**3 / 2.0
#     y[30:60] = 100.0
#     y[60:70] = 0.0
#     y[70:100] = np.cos(x[70:100]) * 100.0
#     return y

# # Function for gradient descent
# def gradient_descent(phi, y, alpha, max_iter, thresh):
#     weights = np.zeros(phi.shape[1])
#     pbar = tqdm(total=max_iter)
    
#     for iter in range(max_iter):
#         # Linear prediction
#         y_hat = phi @ weights
        
#         # Calculate the Mean Squared Error (MSE)
#         mse = np.mean((y - y_hat) ** 2)
        
#         # Gradient calculation
#         gradient = -2 * phi.T @ (y - y_hat) / y.size
        
#         # Update weights
#         weights -= alpha * gradient
        
#         pbar.set_description(f"MSE: {mse:.4f}")
#         pbar.update()
        
#         if mse < thresh:
#             break
    
#     return weights, mse, iter

# # Main execution for fitting the first function
# x = np.linspace(-10, 10, 100)
# y1 = fit_function_1(x)

# # Fit the first function using different feature approximations
# max_iter = 10000
# thresh = 1e-8
# alpha = 0.01  # Learning rate

# for name, get_phi in zip(["Poly", "RBFs", "Tiles", "Coarse", "Aggreg."], [
#         lambda state: poly_features(state, 3),  # Degree 3 for poly features
#         lambda state: rbf_features(state, centers, sigmas),  # Define `centers` and `sigmas`
#         lambda state: tile_features(state, centers, widths),  # Define `centers` and `widths`
#         lambda state: coarse_features(state, centers),  # Define `centers`
#         lambda state: aggregation_features(state, centers),  # Define `centers`
#     ]):
    
#     phi = get_phi(x[..., None])  # Reshape x for the feature functions
#     weights, mse, iter = gradient_descent(phi, y1, alpha, max_iter, thresh)
    
#     # Plotting results
#     y_hat = phi @ weights
#     fig, axs = plt.subplots(1, 2)
#     axs[0].plot(x, y1, label='True Function')
#     axs[1].plot(x, y_hat, label='Approximation', color='orange')
#     axs[0].set_title("True Function")
#     axs[1].set_title(f"Approximation with {name} (MSE {mse:.3f})")
#     plt.show()
#     print(f"Iterations: {iter}, MSE: {mse:.4f}, N. of Features: {len(weights)}")

# # Main execution for fitting the second function
# x = np.linspace(-10, 10, 100)
# y2 = fit_function_2(x)

# for name, get_phi in zip(["Poly", "RBFs", "Tiles", "Coarse", "Aggreg."], [
#         lambda state: poly_features(state, 3),  # Degree 3 for poly features
#         lambda state: rbf_features(state, centers, sigmas),  # Define `centers` and `sigmas`
#         lambda state: tile_features(state, centers, widths),  # Define `centers` and `widths`
#         lambda state: coarse_features(state, centers),  # Define `centers`
#         lambda state: aggregation_features(state, centers),  # Define `centers`
#     ]):
    
#     phi = get_phi(x[..., None])  # Reshape x for the feature functions
#     weights, mse, iter = gradient_descent(phi, y2, alpha, max_iter, thresh)
    
#     # Plotting results
#     y_hat = phi @ weights
#     fig, axs = plt.subplots(1, 2)
#     axs[0].plot(x, y2, label='True Function')
#     axs[1].plot(x, y_hat, label='Approximation', color='orange')
#     axs[0].set_title("True Function")
#     axs[1].set_title(f"Approximation with {name} (MSE {mse:.3f})")
#     plt.show()
#     print(f"Iterations: {iter}, MSE: {mse:.4f}, N. of Features: {len(weights)}")
