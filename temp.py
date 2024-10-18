import numpy as np

# Example usage
phi = np.array([1, 2, 3])
weights = np.array([0.1, 0.2, 0.3])
sigma = np.array([0.1, 0.2, 0.3])  # different std dev for each dimension

# def gaussian_action(phi, weights, sigma: np.array):
#     mu = np.dot(phi, weights)
#     return np.random.normal(mu, sigma**2)

def gaussian_action(phi: np.array, weights: np.array, sigma: np.array):
    mu = np.dot(phi, weights)
    return np.random.normal(mu, sigma**2)



action = gaussian_action(phi, weights, sigma)

print(action)