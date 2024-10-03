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

# Gradient Descent Function for SL fitting
def gradient_descent(phi, y, max_iter=10000, alpha=1.0, thresh=1e-8):
    weights = np.zeros(phi.shape[-1])
    pbar = tqdm(total=max_iter)
    for iter in range(max_iter):
        y_hat = phi @ weights        
        error = y - y_hat        
        mse = np.mean(error**2)
        grad = -2 * phi.T @ error / len(y)
        weights -= alpha * grad
        pbar.set_description(f"MSE: {mse}")
        pbar.update()
        if mse < thresh:
            break
    
    pbar.close()
    return weights, mse

# Part 2 Example Functions

# Function 1: y = sin(x) + x^2 - 0.5 * x^3 + log(|x|)
x = np.linspace(-10, 10, 100)
y = np.sin(x) + x**2 - 0.5 * x**3 + np.log(np.abs(x))

# fig, axs = plt.subplots(1, 1)
# axs.plot(x, y)
# plt.title("Main Function")
# plt.show()

# 2D state: s in [-10, 10] x [0, 1000]
state = np.column_stack([x, np.linspace(0, 1000, len(x))])

# Centers for 2D RBF, Tile, Coarse
n_centers_1 = 20 #10
n_centers_2 = 20 #10
state_1_centers = np.linspace(-10, 10, n_centers_1)
state_2_centers = np.linspace(0, 1000, n_centers_2)
centers = np.array(np.meshgrid(state_1_centers, state_2_centers)).reshape(2, -1).T
sigmas = np.array([1.0, 15.0]) # 15.0 best #np.array([1.0, 50.0]) # Different sigmas for each dimension
widths = np.array([2.0, 100.0])
offsets = [(0, 0), (1, 50), (-1, -50), (1, -50), (-1, 50)]

# Train and visualize using each feature approximation
thresh = 1e-8

# Define different learning rates (alpha) and max_iter for each feature
alphas = [1e-6, 0.08, 1, 0.5, 1]  # Learning rates for each feature type
max_iters = [30000, 200000, 10000, 10000, 10000]  # Maximum iterations for each feature type

# for name, get_phi, alpha, max_iter in zip(
#     ["Poly", "RBFs", "Tiles", "Coarse", "Aggreg."], 
#     [
#         lambda state: poly_features(state, 3),        
#         lambda state: rbf_features(state, centers, sigmas),
#         lambda state: tile_features(state, centers, widths, offsets),
#         lambda state: coarse_features(state, centers, widths, offsets),
#         lambda state: aggregation_features(state, centers),
#     ],
#     alphas,
#     max_iters):
    
#     phi = get_phi(x[..., None])  # from (N,) to (N, S) with S = 1
#     weights, mse = gradient_descent(phi, y, max_iter=max_iter, alpha=alpha, thresh=thresh)
#     y_hat = phi @ weights

#     print(f"Feature Approximation: {name}")
#     print(f"Iterations: {max_iter}, Final MSE: {mse}, Number of Features: {phi.shape[1]}")

#     # Plotting the results
#     fig, axs = plt.subplots(1, 2)
#     axs[0].plot(x, y)
#     axs[1].plot(x, y_hat)
#     axs[0].set_title("True Function")
#     axs[1].set_title(f"Approximation with {name} (MSE: {mse:.3f})")
#     plt.show()


# Now repeat the experiment but fit the following function y.
# Submit your plots and discuss your results, paying attention to the
# non-smoothness of the new target function.
# - How did you change your hyperparameters? Did you use more/less wider/narrower features?
# - Consider the number of features. How would it change if your state would be 2-dimensional?
# Discuss each bullet point in at most two sentences.



# ---------------------------------------------------------------------------------------------------
# Function 2: Non-smooth target function
x = np.linspace(-10, 10, 100)
y = np.zeros(x.shape)
y[0:10] = x[0:10]**3 / 3.0
y[10:20] = np.exp(x[25:35])
y[20:30] = -x[0:10]**3 / 2.0
y[30:60] = 100.0
y[60:70] = 0.0
y[70:100] = np.cos(x[70:100]) * 100.0

# fig, axs = plt.subplots(1, 1)
# axs.plot(x, y)
# plt.show()

# Centers for 2D RBF, Tile, Coarse
n_centers_1 = 30 #10
n_centers_2 = 30 #10
state_1_centers = np.linspace(-10, 10, n_centers_1)
state_2_centers = np.linspace(0, 1000, n_centers_2)
centers = np.array(np.meshgrid(state_1_centers, state_2_centers)).reshape(2, -1).T
sigmas = np.array([1.0, 50.0]) #np.array([1.0, 50.0]) # Different sigmas for each dimension
widths = np.array([2.0, 50.0])
offsets = [(0, 0), (1, 50), (-1, -50), (1, -50), (-1, 50)]

# Define different learning rates (alpha) and max_iter for each feature
alphas = [1e-6, 0.01, 1, 0.1, 1]  # Learning rates for each feature type
max_iters = [10000, 15000, 30000, 25000, 10000]  # Maximum iterations for each feature type

# Train and visualize using each feature approximation for non-smooth function
for name, get_phi, alpha, max_iter in zip(
    ["Poly", "RBFs", "Tiles", "Coarse", "Aggreg."], 
    [
        lambda state: poly_features(state, 3),        
        lambda state: rbf_features(state, centers, sigmas),
        lambda state: tile_features(state, centers, widths, offsets),
        lambda state: coarse_features(state, centers, widths, offsets),
        lambda state: aggregation_features(state, centers),
    ],
    alphas,
    max_iters):
    
    # phi = get_phi(state)
    phi = get_phi(x[..., None])  # from (N,) to (N, S) with S = 1

    weights, mse = gradient_descent(phi, y, max_iter=max_iter, alpha=alpha, thresh=thresh)
    y_hat = phi @ weights

    print(f"Feature Approximation: {name}")
    print(f"Iterations: {max_iter}, Final MSE: {mse}, Number of Features: {phi.shape[1]}")

    # Plotting the results
    fig, axs = plt.subplots(1, 2)
    axs[0].plot(x, y)
    axs[1].plot(x, y_hat)
    axs[0].set_title("True Function")
    axs[1].set_title(f"Approximation with {name} (MSE: {mse:.3f})")
    plt.show()







