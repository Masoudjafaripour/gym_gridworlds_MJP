import numpy as np
import gymnasium
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import PolynomialFeatures

np.set_printoptions(precision=3, suppress=True)

## Feature Functions
def poly_features(state: np.array, degree: int) -> np.array:
    poly = PolynomialFeatures(degree)
    return poly.fit_transform(state)

def rbf_features(state: np.array, centers: np.array, sigmas: float) -> np.array:
    diff = state[:, np.newaxis, :] - centers[np.newaxis, :, :]
    norm_squared = np.sum(diff ** 2, axis=-1)
    return np.exp(-norm_squared / (2 * sigmas ** 2))

def tile_features(state: np.array, centers: np.array, widths: np.array, offsets: list = [0]) -> np.array:
    activations = np.zeros((state.shape[0], centers.shape[0]))
    for offset in offsets:
        shifted_centers = centers + offset
        in_tile = np.all(np.abs(state[:, np.newaxis] - shifted_centers) <= widths / 1, axis=2)
        activations += in_tile
    return activations / len(offsets)

def coarse_features(state: np.array, centers: np.array, widths: float, offsets: list = [0]) -> np.array:
    n_samples, n_dimensions = state.shape
    n_centers = centers.shape[0]
    features = np.zeros((n_samples, n_centers))

    for offset in offsets:
        shifted_centers = centers + np.array(offset)
        distances = np.linalg.norm(state[:, None, :] - shifted_centers[None, :, :], ord=2, axis=-1)  # L2 norm
        within_circle = distances <= widths
        features += within_circle

    return features.astype(float)

def aggregation_features(state: np.array, centers: np.array) -> np.array:
    distances = np.linalg.norm(state[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=-1)
    closest_center = np.argmin(distances, axis=1)
    
    features = np.zeros((state.shape[0], centers.shape[0]))
    features[np.arange(state.shape[0]), closest_center] = 1
    return features

# Generate random states and centers for testing the feature functions
state_size = 2
n_samples = 10
n_centers = 100
state = np.random.rand(n_samples, state_size)  # Random state in [0, 1]

# Generate grid of centers for RBF, tile, and coarse coding
state_1_centers = np.linspace(-0.2, 1.2, n_centers)
state_2_centers = np.linspace(-0.2, 1.2, n_centers)
centers = np.array(np.meshgrid(state_1_centers, state_2_centers)).reshape(state_size, -1).T  # Grid of uniformly spaced centers

# Set hyperparameters
sigmas = 0.2
widths = 0.2
offsets = [(-0.1, 0.0), (0.0, 0.1), (0.1, 0.0), (0.0, -0.1)]

# Compute features
poly = poly_features(state, 2)
aggr = aggregation_features(state, centers)
rbf = rbf_features(state, centers, sigmas)
tile_one = tile_features(state, centers, widths)
tile_multi = tile_features(state, centers, widths, offsets)
coarse_one = coarse_features(state, centers, widths)
coarse_multi = coarse_features(state, centers, widths, offsets)

# Plot the heatmaps for the different feature representations
fig, axs = plt.subplots(1, 6, figsize=(20, 4))
extent = [state_1_centers[0], state_1_centers[-1], state_2_centers[0], state_2_centers[-1]]

axs[0].imshow(rbf[0].reshape(n_centers, n_centers), extent=extent, origin='lower')
axs[1].imshow(tile_one[0].reshape(n_centers, n_centers), extent=extent, origin='lower')
axs[2].imshow(tile_multi[0].reshape(n_centers, n_centers), extent=extent, origin='lower')
axs[3].imshow(coarse_one[0].reshape(n_centers, n_centers), extent=extent, origin='lower')
axs[4].imshow(coarse_multi[0].reshape(n_centers, n_centers), extent=extent, origin='lower')
axs[5].imshow(aggr[0].reshape(n_centers, n_centers), extent=extent, origin='lower')

# Titles for the subplots
titles = ["RBFs", "Tile (1 Tiling)", "Tile (4 Tilings)", "Coarse (1 Field)", "Coarse (4 Fields)", "Aggreg."]
for ax, title in zip(axs, titles):
    ax.plot(state[0][0], state[0][1], marker="+", markersize=12, color="red")
    ax.set_title(title)

plt.suptitle(f"State {state[0]}")
plt.show()
