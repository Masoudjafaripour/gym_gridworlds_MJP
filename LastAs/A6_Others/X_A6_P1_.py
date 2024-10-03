import numpy as np
import gymnasium
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import PolynomialFeatures


np.set_printoptions(precision=3, suppress=True)

# Notation for array sizes:
# - S: state dimensionality
# - D: features dimensionality
# - N: number of samples
#
# N is always the first dimension, meaning that states come in arrays of shape (N, S)
# and features in arrays of shape (N, D).
# We recommend to implement the functions below assuming that the input has
# always shape (N, S) and the output (N, D), even when N = 1.

def poly_features(state: np.array, degree: int) -> np.array:
    """
    Compute polynomial features. For example, if state = (s1, s2) and degree = 2,
    the output must be 1 + s1 + s2 + s1*s2 + s1**2 + s2**2
    """
    return PolynomialFeatures(degree, include_bias=True).fit_transform(state)

def rbf_features(state: np.array, centers: np.array, sigmas: np.array) -> np.array:
    """
    Compute radial basis functions features: exp(- ||s - c||**2 / w**2)
    where s is the state, c are the centers, and w are the widths (sigmas) of the Gaussians.
    Assume sigma is the same for all dimensions.
    """
    # Compute the squared distances from each state to each center
    distances = np.linalg.norm(state[:, np.newaxis] - centers, axis=2)**2
    # Compute the RBF features
    return np.exp(-distances / (sigmas**2))


def tile_features(state: np.array, centers: np.array, widths: np.array) -> np.array:
    n_samples = state.shape[0]
    n_tiles = centers.shape[0]
    features = np.zeros((n_samples, n_tiles))

    for i in range(n_samples):
        for j in range(n_tiles):
            # Calculate the boundaries of the tile
            lower_bound = centers[j] - (widths[j] / 2)
            upper_bound = centers[j] + (widths[j] / 2)

            # Check if the state falls within the bounds of the tile
            if np.all(state[i] >= lower_bound) and np.all(state[i] <= upper_bound):
                features[i, j] = 1  # Assign tile feature
    
    return features



def aggregation_features(state: np.array, centers: np.array) -> np.array:
    """
    Compute tile coding features: given centers and widths, the output will be an
    array of 0s and one 1 corresponding to the closest tile the state belongs to.
    This is basically tile coding with non-overlapping tiles (a state belongs to one
    tile only).
    Note that we can turn this into a discrete (finite) representation of the state,
    because we will have as many feature representations as centers.
    """
    n_samples = state.shape[0]
    n_centers = centers.shape[0]
    features = np.zeros((n_samples, n_centers))
    
    for i in range(n_samples):
        distances = np.linalg.norm(state[i] - centers, axis=1)
        closest_tile_index = np.argmin(distances)  # Get index of closest center
        features[i, closest_tile_index] = 1  # Assign feature to closest tile
    
    return features


state_size = 2
n_samples = 10
n_centers = 10 # should we change it? hyper?
state = np.random.rand(n_samples, state_size)  # in [0, 1]

centers = np.array(
    np.meshgrid(np.linspace(-0.2, 1.0, n_centers), np.linspace(-0.2, 1.0, n_centers))
).reshape(2, -1).T
sigmas = np.ones(n_centers * n_centers) * 0.2
widths = np.ones(n_centers * n_centers) * 0.2 # initial -->> 0.05 fixed in class

poly = poly_features(state, 2)
aggr = aggregation_features(state, centers)
rbf = rbf_features(state, centers, sigmas)
tile = tile_features(state, centers, widths)

fig, axs = plt.subplots(1, 3)

axs[0].tricontourf(centers[:, 0], centers[:, 1], rbf[0])
axs[1].tricontourf(centers[:, 0], centers[:, 1], tile[0])
axs[2].tricontourf(centers[:, 0], centers[:, 1], aggr[0])
for ax, title in zip(axs, ["RBFs", "Tile", "Aggreg."]):  # we can't plot poly like this
    ax.plot(state[0][0], state[0][1], marker="+", markersize=12, color="red")
    ax.set_title(title + f" for {state[0]}")
plt.show()

#################### PART 1
# Submit your heatmaps.
# What are the hyperparameters of each FA and how do they affect the shape of
# the function they can approximate?
# - In RBFs the hyperparameter(s) is/are ... More/less ... will affect ...,
#   while narrower/wider ... will affect ...
# - In tile coding the hyperparameter(s) is/are ...
# - In polynomials the hyperparameter(s) is/are ...
# - In state aggregation the hyperparameter(s) is/are ...