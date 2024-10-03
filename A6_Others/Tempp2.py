# P4 -- 12:35 pm not working

# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# from sklearn.preprocessing import PolynomialFeatures

# np.set_printoptions(precision=3, suppress=True)
# # Feature Approximation Methods
# def poly_features(state: np.array, degree: int) -> np.array:
#     poly = PolynomialFeatures(degree)
#     return poly.fit_transform(state)

# def rbf_features(state, centers, sigmas):
#     """
#     Returns the RBF feature representation for a given state.
#     `state`: (N, 2) where N is the number of data points and 2 corresponds to the 2D state.
#     `centers`: (M, 2) where M is the number of RBF centers.
#     `sigmas`: (2,) or (M, 2) array specifying the widths of the RBFs for each dimension.
#     """
#     # Calculate the distance between the state and the centers
#     dists = np.linalg.norm(state[:, None, :] - centers[None, :, :], axis=-1)  # (N, M)

#     # Apply sigmas element-wise
#     if sigmas.shape == (2,):
#         # Apply different sigmas for each dimension
#         sigmas = np.ones_like(dists) * sigmas.mean()  # Example case, adjust if needed
    
#     # Compute RBF features
#     return np.exp(-0.5 * (dists / sigmas)**2)

# def tile_features(state: np.array, centers: np.array, widths: np.array, offsets: list = [0]) -> np.array:
#     activations = np.zeros((state.shape[0], centers.shape[0]))
#     for offset in offsets:
#         shifted_centers = centers + offset
#         in_tile = np.all(np.abs(state[:, np.newaxis] - shifted_centers) <= widths / 2, axis=2)
#         activations += in_tile
#     return activations / len(offsets)

# def coarse_features(state, centers, widths, offsets):
#     """
#     Returns the coarse coding feature representation for a given state.
#     `state`: (N, 2) where N is the number of data points and 2 corresponds to the 2D state.
#     `centers`: (M, 2) where M is the number of coarse coding centers.
#     `widths`: (2,) array specifying the widths of the receptive fields for each dimension.
#     `offsets`: (M,) array specifying the offsets for each feature center.
#     """
#     # Calculate the distances between the state and the centers
#     dists = np.abs(state[:, None, :] - centers[None, :, :])  # (N, M, 2)

#     # Apply different widths for each dimension
#     widths = widths.reshape(1, 1, -1)  # Reshape to broadcast correctly
#     in_field = np.all(dists <= widths, axis=-1)  # (N, M)

#     # Return features as binary indicators (1 if within the receptive field, else 0)
#     return in_field.astype(float)

# def aggregation_features(state: np.array, centers: np.array) -> np.array:
#     dists = np.linalg.norm(state[:, np.newaxis] - centers, axis=2)
#     closest = np.argmin(dists, axis=1)
#     activations = np.zeros((state.shape[0], centers.shape[0]))
#     activations[np.arange(state.shape[0]), closest] = 1
#     return activations

# # Load the data
# data = np.load("a6_gridworld.npz")
# s = data["s"]
# a = data["a"]
# r = data["r"]
# s_next = data["s_next"]
# Q = data["Q"]  # true Q-values (from the dataset)
# term = data["term"]
# n = s.shape[0]
# n_states = 81
# n_actions = 5
# gamma = 0.99

# # Display true Q-functions for each action as heatmaps
# s_idx = np.ravel_multi_index(s.T, (9, 9))
# unique_s_idx = np.unique(s_idx, return_index=True)[1]

# # Displaying the true Q-function as heatmaps for each action
# fig, axs = plt.subplots(1, n_actions, figsize=(15, 5))
# for i, action in enumerate(["LEFT", "DOWN", "RIGHT", "UP", "STAY"]):
#     axs[i].imshow(Q[unique_s_idx, i].reshape(9, 9))
#     axs[i].set_title(f"True Q {action}")
# plt.show()

# # Set parameters
# max_iter = 1000
# alpha = 0.01
# thresh = 1e-8

# degree = 2  # Example polynomial degree, adjust as necessary
# centers = np.array([[i, j] for i in range(9) for j in range(9)])  # RBF centers
# sigmas = np.array([1.0, 1.0])  # RBF sigmas
# widths = np.array([1.0, 1.0])  # Coarse coding widths
# offsets = [0]  # Tile coding offsets

# # Pick one feature approximation method
# # Uncomment one of the following to test different feature approximations
# name, get_phi = "Poly", lambda state: poly_features(state, degree)
# # name, get_phi = "RBFs", lambda state: rbf_features(state, centers, sigmas)
# # name, get_phi = "Coarse", lambda state: coarse_features(state, centers, widths, offsets)
# # name, get_phi = "Tiles", lambda state: tile_features(state, centers, widths, offsets)
# # name, get_phi = "Aggreg.", lambda state: aggregation_features(state, centers)

# # Initialize features and weights
# phi = get_phi(s)            # shape: (n_samples, n_features)
# phi_next = get_phi(s_next)   # shape: (n_samples, n_features)

# # Initialize weights with correct shape
# weights = np.zeros((phi.shape[1], n_actions))  # weights should match the feature dimensions

# # Semi-gradient TD learning loop for Q-function approximation
# pbar = tqdm(total=max_iter)
# for iter in range(max_iter):
#     # TD Error for Q: δ = r + γ * max_a' Q(s', a') - Q(s, a)
#     V_s_next = np.max(np.dot(phi_next, weights), axis=1) * (1 - term)  # Max over actions

#     # Compute Q(s, a) using dot product
#     Q_sa = np.sum(phi * weights[:, a], axis=1)  # Q(s, a) for the current action
#     td_error = r + gamma * V_s_next - Q_sa
    
#     # Gradient update: Update weights only for the actions taken
#     for i in range(n_actions):
#         mask = (a == i)  # Only update for actions taken
#         # Fixing the dimension mismatch with element-wise multiplication
#         weights[:, i] += alpha * np.sum(td_error[mask][:, None] * phi[mask], axis=0) / n
#     weights[:, i] += alpha * gradient[a==i].mean()
#     # Compute MSE between prediction and true Q-values
#     q_pred = np.dot(phi, weights)
#     mse = np.mean((q_pred - Q) ** 2)
#     pbar.set_description(f"TDE: {td_error.mean():.4f}, MSE: {mse:.6f}")
#     pbar.update()

#     # Early stopping if MSE converges
#     if mse < thresh:
#         break

# # Display results
# print(f"Iterations: {iter}, MSE: {mse:.6f}, N. of Features {weights.shape[0]}")

# # Visualizing the learned Q-function vs the true Q-function for each action
# fig, axs = plt.subplots(2, n_actions, figsize=(15, 10))
# td_prediction = np.dot(phi, weights)

# for i, action in enumerate(["LEFT", "DOWN", "RIGHT", "UP", "STAY"]):
#     axs[0][i].imshow(Q[unique_s_idx, i].reshape(9, 9))
#     axs[0][i].set_title(f"True Q {action}")
#     axs[1][i].imshow(td_prediction[unique_s_idx, i].reshape(9, 9))
#     axs[1][i].set_title(f"Approx. Q {action} (MSE {mse:.3f})")

# plt.show()



# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm

# # Define feature functions
# def poly_features(state, degree=3):
#     """Polynomial features up to the given degree."""
#     return np.array([state**d for d in range(degree + 1)]).T

# def rbf_features(state, centers, sigma):
#     """Radial basis function features."""
#     return np.exp(-np.square(np.subtract(state[:, None], centers)) / (2 * sigma**2))

# def tile_features(state, num_tilings, tiles_per_dim):
#     """Tile coding features."""
#     tiles = np.zeros((num_tilings, tiles_per_dim))
#     for i in range(num_tilings):
#         offset = i / num_tilings
#         tiled = (state + offset) % 1 * tiles_per_dim
#         tiles[i, np.floor(tiled).astype(int)] = 1
#     return tiles.flatten()

# def coarse_features(state, centers, width):
#     """Coarse coding features."""
#     return np.array([1 if np.abs(state - center) < width else 0 for center in centers]).T

# def aggregation_features(state, features):
#     """Aggregate features (just a placeholder for demonstration)."""
#     return np.sum(features, axis=0)

# # Gradient descent function
# def gradient_descent(phi, y, alpha, max_iter, thresh):
#     weights = np.zeros(phi.shape[1])  # Adjust weights to match the number of features
#     pbar = tqdm(total=max_iter)
    
#     for iter in range(max_iter):
#         y_hat = phi @ weights
#         mse = np.mean((y - y_hat) ** 2)
#         gradient = -2 * phi.T @ (y - y_hat) / y.size
#         weights -= alpha * gradient
        
#         pbar.set_description(f"MSE: {mse:.4f}")
#         pbar.update()
        
#         if mse < thresh:
#             break
    
#     return weights, mse, iter

# # Main code
# max_iter = 10000
# thresh = 1e-8
# alpha = 0.01

# # Define x and true function y for the first part
# x = np.linspace(-10, 10, 100)
# y1 = np.sin(x) + x**2 - 0.5 * x**3 + np.log(np.abs(x))

# # Plot the true function
# plt.figure()
# plt.plot(x, y1)
# plt.title("True Function y1")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.grid()
# plt.show()

# # Feature approximators to test
# for name, get_phi in zip(
#     ["Poly", "RBFs", "Tiles", "Coarse", "Aggreg."],
#     [
#         lambda state: poly_features(state, degree=3),
#         lambda state: rbf_features(state, centers=np.linspace(-10, 10, 10), sigma=1.0),
#         lambda state: tile_features(state, num_tilings=5, tiles_per_dim=10),
#         lambda state: coarse_features(state, centers=np.linspace(-10, 10, 10), width=1.0),
#         lambda state: aggregation_features(state, features=[poly_features(state, degree=3), rbf_features(state, centers=np.linspace(-10, 10, 10), sigma=1.0)])
#     ]
# ):
#     phi = get_phi(x[..., None])  # Reshape x for the feature functions
#     weights, mse, iter = gradient_descent(phi, y1, alpha, max_iter, thresh)

#     print(f"{name} - Iterations: {iter}, MSE: {mse:.4f}, N. of Features: {len(weights)}")
#     y_hat = phi @ weights
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.plot(x, y1, label='True Function')
#     plt.title("True Function y1")
#     plt.xlabel("x")
#     plt.ylabel("y")
#     plt.grid()
    
#     plt.subplot(1, 2, 2)
#     plt.plot(x, y_hat, label='Approximation', color='orange')
#     plt.title(f"Approximation with {name} (MSE {mse:.4f})")
#     plt.xlabel("x")
#     plt.ylabel("y_hat")
#     plt.grid()
#     plt.tight_layout()
#     plt.show()

# # Now repeat for the second function
# y2 = np.zeros(x.shape)
# y2[0:10] = x[0:10]**3 / 3.0
# y2[10:20] = np.exp(x[10:20])
# y2[20:30] = -x[20:30]**3 / 2.0
# y2[30:60] = 100.0
# y2[60:70] = 0.0
# y2[70:100] = np.cos(x[70:100]) * 100.0

# # Plot the second true function
# plt.figure()
# plt.plot(x, y2)
# plt.title("True Function y2")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.grid()
# plt.show()

# for name, get_phi in zip(
#     ["Poly", "RBFs", "Tiles", "Coarse", "Aggreg."],
#     [
#         lambda state: poly_features(state, degree=3),
#         lambda state: rbf_features(state, centers=np.linspace(-10, 10, 10), sigma=1.0),
#         lambda state: tile_features(state, num_tilings=5, tiles_per_dim=10),
#         lambda state: coarse_features(state, centers=np.linspace(-10, 10, 10), width=1.0),
#         lambda state: aggregation_features(state, features=[poly_features(state, degree=3), rbf_features(state, centers=np.linspace(-10, 10, 10), sigma=1.0)])
#     ]
# ):
#     phi = get_phi(x[..., None])  # Reshape x for the feature functions
#     weights, mse, iter = gradient_descent(phi, y2, alpha, max_iter, thresh)

#     print(f"{name} - Iterations: {iter}, MSE: {mse:.4f}, N. of Features: {len(weights)}")
#     y_hat = phi @ weights
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.plot(x, y2, label='True Function')
#     plt.title("True Function y2")
#     plt.xlabel("x")
#     plt.ylabel("y")
#     plt.grid()
    
#     plt.subplot(1, 2, 2)
#     plt.plot(x, y_hat, label='Approximation', color='orange')
#     plt.title(f"Approximation with {name} (MSE {mse:.4f})")
#     plt.xlabel("x")
#     plt.ylabel("y_hat")
#     plt.grid()
#     plt.tight_layout()
#     plt.show()
