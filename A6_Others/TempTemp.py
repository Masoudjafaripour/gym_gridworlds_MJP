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
s = data["s"]
a = data["a"]
r = data["r"]
s_next = data["s_next"]
Q = data["Q"]  # true Q-values (from the dataset)
term = data["term"]
n = s.shape[0]
n_states = 81
n_actions = 5
gamma = 0.99

# Display true Q-functions for each action as heatmaps
s_idx = np.ravel_multi_index(s.T, (9, 9))
unique_s_idx = np.unique(s_idx, return_index=True)[1]

# Displaying the true Q-function as heatmaps for each action
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
name, get_phi = "Poly", lambda state: poly_features(state, degree)
name, get_phi = "RBFs", lambda state: rbf_features(state, centers, sigmas)
# name, get_phi = "Coarse", lambda state: coarse_features(state, centers, widths, offsets)
# name, get_phi = "Tiles", lambda state: tile_features(state, centers, widths, offsets)
# name, get_phi = "Aggreg.", lambda state: aggregation_features(state, centers)

# Initialize features and weights
phi = get_phi(s)            # shape: (n_samples, n_features)
phi_next = get_phi(s_next)   # shape: (n_samples, n_features)

# Initialize weights correctly
weights = np.zeros((phi.shape[1], n_actions))  # weights should match (n_features, n_actions)

# Semi-gradient TD learning loop for Q-function approximation
pbar = tqdm(total=max_iter)
for iter in range(max_iter):
    # TD Error for Q: δ = r + γ * max_a' Q(s', a') - Q(s, a)
    V_s_next = np.max(np.dot(phi_next, weights), axis=1) * (1 - term)  # Max over actions
    Q_sa = np.dot(phi, weights[:, a])  # Q(s, a) for the current action
    td_error = r + gamma * V_s_next - Q_sa

    # Gradient update: Update weights only for the actions taken
    for i in range(n_actions):
        mask = (a == i)  # Only update for actions taken
        # Fixing the dimension mismatch with element-wise multiplication
        weights[:, i] += alpha * np.sum(td_error[mask] * phi[mask], axis=0) / n

    
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
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import PolynomialFeatures
# from tqdm import tqdm

# # Function to be approximated
# x = np.linspace(-10, 10, 100)
# y = np.sin(x) + x**2 - 0.5 * x**3 + np.log(np.abs(x))

# # Feature transformation (polynomial)
# def poly_features(state: np.array, degree: int) -> np.array:
#     poly = PolynomialFeatures(degree)
#     return poly.fit_transform(state)

# # Parameters
# degree = 3  # You can try different degrees
# max_iter = 10000
# thresh = 1e-8
# alpha = 1e-6  # Smaller learning rate to prevent overflow

# # Gradient descent
# for name, get_phi in zip(["Poly"], [lambda state: poly_features(state, degree)]):
#     phi = get_phi(x[..., None])  # from (N,) to (N, S) with S = 1
#     weights = np.random.randn(phi.shape[-1]) * 0.01  # Small random initialization
#     pbar = tqdm(total=max_iter)
    
#     for iter in range(max_iter):
#         # Compute predictions
#         y_hat = phi @ weights
        
#         # Compute Mean Squared Error (MSE)
#         mse = np.mean((y - y_hat)**2)
        
#         # Update weights using gradient descent
#         grad = -2 * phi.T @ (y - y_hat) / len(y)
#         weights -= alpha * grad
        
#         # Check for convergence
#         pbar.set_description(f"MSE: {mse:.8f}")
#         pbar.update()
#         if mse < thresh or np.isnan(mse):  # Stop if MSE is NaN
#             break

#     pbar.close()
    
#     print(f"Iterations: {iter}, MSE: {mse:.8f}, N. of Features {len(weights)}")

#     # Plot the true function vs approximation
#     fig, axs = plt.subplots(1, 2, figsize=(12, 4))
#     axs[0].plot(x, y, label="True Function")
#     axs[1].plot(x, y_hat, label=f"Approximation (MSE {mse:.3f})", color="orange")
#     axs[0].set_title("True Function")
#     axs[1].set_title(f"Approximation with {name} (MSE {mse:.3f})")
#     plt.show()






# # import numpy as np
# # import matplotlib.pyplot as plt
# # from tqdm import tqdm

# # # Define feature approximations
# # def poly_features(state, degree=2):
# #     """Polynomial features up to the given degree."""
# #     x, y = state[:, 0], state[:, 1]
# #     return np.column_stack([x**i * y**j for i in range(degree + 1) for j in range(degree + 1 - i)])

# # def rbf_features(state, centers, sigmas):
# #     """Radial Basis Function features."""
# #     dists = np.linalg.norm(state[:, None, :] - centers[None, :, :], axis=-1)
# #     return np.exp(-0.5 * (dists / sigmas)**2)

# # def tile_features(state, centers, widths, offsets):
# #     """Tile coding features."""
# #     dists = np.abs(state[:, None, :] - centers[None, :, :])
# #     widths = widths.reshape(1, 1, -1)
# #     in_field = np.all(dists <= widths, axis=-1)
# #     return in_field.astype(float)

# # def coarse_features(state, centers, widths, offsets):
# #     """Coarse coding features."""
# #     dists = np.abs(state[:, None, :] - centers[None, :, :])
# #     widths = widths.reshape(1, 1, -1)
# #     in_field = np.all(dists <= widths, axis=-1)
# #     return in_field.astype(float)

# # def aggregation_features(state, grid_size=(10, 10)):
# #     """Aggregation features based on discretization."""
# #     x, y = state[:, 0], state[:, 1]
    
# #     # Normalize x and y values between 0 and 1 if not already
# #     x_norm = np.clip(x, 0, 1)
# #     y_norm = np.clip(y, 0, 1)
    
# #     # Create bins for x and y coordinates
# #     x_bins = np.digitize(x_norm, np.linspace(0, 1, grid_size[0] + 1)) - 1
# #     y_bins = np.digitize(y_norm, np.linspace(0, 1, grid_size[1] + 1)) - 1
    
# #     # Ensure indices are within the valid range
# #     x_bins = np.clip(x_bins, 0, grid_size[0] - 1)
# #     y_bins = np.clip(y_bins, 0, grid_size[1] - 1)
    
# #     # Create features array (grid_size[0] * grid_size[1] features)
# #     features = np.zeros((len(state), grid_size[0] * grid_size[1]))
    
# #     # Map the binned (x, y) values to feature indices
# #     indices = x_bins * grid_size[1] + y_bins
# #     features[np.arange(len(state)), indices] = 1
    
# #     return features


# # # Load dataset
# # data = np.load("a6_gridworld.npz")
# # s = data["s"]
# # a = data["a"]
# # r = data["r"]
# # s_next = data["s_next"]
# # Q = data["Q"]
# # V = data["Q"].max(-1)  # value of the greedy policy
# # term = data["term"]
# # n = s.shape[0]
# # gamma = 0.99

# # # Visualize the true V-function
# # fig, axs = plt.subplots(1, 1)
# # axs.tricontourf(s[:, 0], s[:, 1], V, levels=100)
# # plt.show()

# # # Set hyperparameters
# # max_iter = 20000
# # alpha = 0.01
# # thresh = 1e-8

# # # Choose a Feature Approximation method
# # # Uncomment one of the following
# # # name, get_phi = "Poly", lambda state : poly_features(state, degree=3)
# # # name, get_phi = "RBFs", lambda state : rbf_features(state, centers=np.random.rand(100, 2), sigmas=0.1)
# # # name, get_phi = "Tiles", lambda state : tile_features(state, centers=np.random.rand(100, 2), widths=np.array([0.1, 0.1]), offsets=np.zeros(100))
# # # name, get_phi = "Coarse", lambda state : coarse_features(state, centers=np.random.rand(100, 2), widths=np.array([0.2, 0.2]), offsets=np.zeros(100))
# # name, get_phi = "Aggreg.", lambda state: aggregation_features(state, grid_size=(10, 10))

# # # Initialize features and weights
# # phi = get_phi(s)
# # phi_next = get_phi(s_next)
# # weights = np.zeros(phi.shape[-1])

# # # Run semi-gradient TD(0) batch prediction
# # pbar = tqdm(total=max_iter)
# # for iter in range(max_iter):
# #     # TD error
# #     td_prediction = phi @ weights
# #     td_target = r + gamma * (1 - term) * (phi_next @ weights)
# #     td_error = td_target - td_prediction

# #     # Update weights using semi-gradient
# #     weights += alpha * phi.T @ td_error / n

# #     # Compute MSE
# #     mse = np.mean((td_prediction - V)**2)

# #     pbar.set_description(f"TDE: {np.mean(np.abs(td_error)):.5f}, MSE: {mse:.5f}")
# #     pbar.update()

# #     if mse < thresh:
# #         break

# # pbar.close()

# # print(f"Iterations: {iter}, MSE: {mse}, N. of Features {len(weights)}")

# # # Plot true and approximated value functions
# # fig, axs = plt.subplots(1, 2)
# # axs[0].tricontourf(s[:, 0], s[:, 1], V, levels=100)
# # axs[1].tricontourf(s[:, 0], s[:, 1], phi @ weights, levels=100)
# # axs[0].set_title("True Function")
# # axs[1].set_title(f"Approximation with {name} (MSE {mse:.3f})")
# # plt.show()

# # # TD Approximation of Q-function
# # max_iter = 1000
# # alpha = 0.01
# # thresh = 1e-8
# # n_actions = 5  # assuming 5 actions in this example

# # phi = get_phi(s)
# # phi_next = get_phi(s_next)

# # # Weights for each action: shape = (n_features, n_actions)
# # weights = np.zeros((phi.shape[-1], n_actions))

# # pbar = tqdm(total=max_iter)
# # for iter in range(max_iter):
# #     # Calculate the current Q-values using the feature matrix and weights
# #     q_pred = phi @ weights  # shape: (n_samples, n_actions)
    
# #     # TD error for each action taken
# #     q_next = phi_next @ weights  # predicted Q-values for next state
# #     td_target = r + gamma * (1 - term) * q_next.max(axis=1)
# #     td_error = td_target - q_pred[np.arange(n), a]
    
# #     # Check for NaN or Inf values
# #     if np.any(np.isnan(td_error)) or np.any(np.isinf(td_error)):
# #         print(f"NaN or Inf detected in TD error at iteration {iter}")
# #         break

# #     # Update weights using semi-gradient TD update
# #     for i in range(n):  # iterate over each sample
# #         weights[:, a[i]] += alpha * td_error[i] * phi[i]
    
# #     # Mean squared error (MSE) for monitoring progress
# #     mse = np.mean((q_pred[np.arange(n), a] - Q[np.arange(n), a])**2)
    
# #     # Check for NaN or Inf in q_pred
# #     if np.any(np.isnan(q_pred)) or np.any(np.isinf(q_pred)):
# #         print(f"NaN or Inf detected in Q-prediction at iteration {iter}")
# #         break
    
# #     pbar.set_description(f"TDE: {np.mean(td_error):.4f}, MSE: {mse:.4f}")
# #     pbar.update()
    
# #     if mse < thresh:
# #         break

# # print(f"Iterations: {iter}, MSE: {mse:.4f}, N. of Features {weights.shape[0] * weights.shape[1]}")

# # # Plotting the Q-function and its approximation
# # fig, axs = plt.subplots(5, 2, figsize=(10, 20))
# # for i in range(5):  # for each action
# #     axs[i][0].tricontourf(s[:, 0], s[:, 1], Q[:, i], levels=100)
# #     axs[i][1].tricontourf(s[:, 0], s[:, 1], q_pred[:, i], levels=100)  # use the predicted Q-values
# #     axs[i][0].set_title(f"True Q-function for action {i}")
# #     axs[i][1].set_title(f"Approximated Q-function (MSE {mse:.3f})")
# # plt.tight_layout()
# # plt.show()
