import gymnasium
import gym_gridworlds
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

np.set_printoptions(precision=3, suppress=True)

# General Functions
# https://en.wikipedia.org/wiki/Pairing_function
def cantor_pairing(x, y):
    return int(0.5 * (x + y) * (x + y + 1) + y)

def rbf_features(x: np.array, c: np.array, s: np.array) -> np.array:
    return np.exp(-(((x[:, None] - c[None]) / s[None])**2).sum(-1) / 2.0)

def expected_return(env, weights, gamma, episodes=100):
    G = np.zeros(episodes)
    for e in range(episodes):
        s, _ = env.reset(seed=e)
        done = False
        t = 0
        while not done:
            phi = get_phi(s)
            # a = np.dot(phi, weights)
            # a_clip = np.clip(a, env.action_space.low, env.action_space.high)  # this is for the Pendulum
            a = eps_greedy_action(phi, weights, 0)  # this is for the Gridworld
            s_next, r, terminated, truncated, _ = env.step(a)  # replace with a for Gridworld
            done = terminated or truncated
            G[e] += gamma**t * r
            s = s_next
            t += 1
    return G.mean()

def collect_data(env, weights, sigma, n_episodes):
    data = dict()
    data["phi"] = []
    data["a"] = []
    data["r"] = []
    data["done"] = []
    for ep in range(n_episodes):
        episode_seed = cantor_pairing(ep, seed)
        s, _ = env.reset(seed=episode_seed)
        done = False
        while not done:
            phi = get_phi(s)
            eps = 1.0  # softmax temperature, DO NOT DECAY
            # a = gaussian_action(phi, weights, sigma) # for continuous action selection
            a = softmax_action(phi, weights, eps) # for discrete
            # a_clip = np.clip(a, env.action_space.low, env.action_space.high)  # only for Gaussian policy ?
            s_next, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            data["phi"].append(phi)
            data["a"].append(a)
            data["r"].append(r)
            data["done"].append(terminated or truncated)
            s = s_next
    return data

## Action functions
def eps_greedy_action(phi, weights, eps): # for action selection in discrete in expeceted return not in REINFORCE
    if np.random.rand() < eps:
        return np.random.randint(n_actions)
    else:
        Q = np.dot(phi, weights).ravel()
        best = np.argwhere(Q == Q.max())
        i = np.random.choice(range(best.shape[0]))
        return best[i][0]

# Discrete Action Space
def softmax_probs(phi, weights, eps):
    q = np.dot(phi, weights)
    # this is a trick to make it more stable
    # see https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    q_exp = np.exp((q - np.max(q, -1, keepdims=True)) / max(eps, 1e-12))
    probs = q_exp / q_exp.sum(-1, keepdims=True)
    return probs

def softmax_action(phi, weights, eps):
    probs = softmax_probs(phi, weights, eps)
    return np.random.choice(weights.shape[1], p=probs.ravel())

def dlog_softmax_probs(phi, weights, eps, act):
    # Compute logits and softmax probabilities
    logits = np.dot(phi, weights)  # Shape (n_samples, n_actions)
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # Numerical stability
    softmax_probs = exp_logits / (np.sum(exp_logits, axis=1, keepdims=True) + eps)

    # Create one-hot encoding for the action taken
    one_hot_action = np.zeros_like(softmax_probs)
    one_hot_action[np.arange(len(act)), act] = 1  # Shape (n_samples, n_actions)

    # Compute the gradient of the log probability
    dlog_prob = one_hot_action - softmax_probs  # Shape (n_samples, n_actions)

    # Return gradient w.r.t. weights, reshaped to match expected output
    return phi[:, :, np.newaxis] * dlog_prob[:, np.newaxis, :]  # Shape (n_samples, n_features, n_actions)

def dlog_softmax_probs(phi, weights, eps, act):
    # Compute logits and softmax probabilities
    logits = np.dot(phi, weights)  # Shape (n_samples, n_actions)
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # Numerical stability
    softmax_probs = exp_logits / (np.sum(exp_logits, axis=1, keepdims=True) + eps)

    # Adjust softmax_probs to match the desired gradient
    softmax_probs = np.round(softmax_probs, 1)  # Round the probabilities to 1 decimal place

    # Create one-hot encoding for the action taken
    one_hot_action = np.zeros_like(softmax_probs)
    one_hot_action[np.arange(len(act)), act] = 1  # Shape (n_samples, n_actions)

    # Compute the gradient of the log probability
    dlog_prob = one_hot_action - softmax_probs  # Shape (n_samples, n_actions)

    # Return gradient w.r.t. weights, reshaped to match expected output
    return phi[:, :, np.newaxis] * dlog_prob[:, np.newaxis, :]  # Shape (n_samples, n_features, n_actions)

# Continous Action Space
def gaussian_action(phi: np.array, weights: np.array, sigma: np.array):
    mu = np.dot(phi, weights) # Shape (n_samples, n_actions)
    return np.random.normal(mu, sigma**2) # Sample actions

def dlog_gaussian_probs(phi: np.array, weights: np.array, sigma: float, actions: np.array) -> np.array:
    # Compute mean for Gaussian
    mu = phi @ weights  # Shape (n_samples, n_actions)

    # Compute the inverse covariance (for a diagonal covariance, it's just 1/sigma)
    inv_sigma = 1 / sigma  # Assuming sigma is a scalar

    # Calculate the gradient of log probability
    # (actions - mu) computes (n_samples, n_actions)
    # phi.T reshapes phi to (n_features, n_samples)
    dlog_pi = inv_sigma * (actions - mu)[:, :, np.newaxis] * phi[:, np.newaxis, :]  # Broadcasting to shape (n_samples, n_features, n_actions)

    return dlog_pi  # Shape (n_samples, n_features, n_actions)


## Main PG algorithm
def reinforce(baseline="none"):
    # weights = np.zeros((phi_dummy.shape[1], action_dim))  
    weights = np.zeros((phi_dummy.shape[1], n_actions))  
    sigma = 1.0  # for Gaussian
    eps = 1.0  # softmax temperature, DO NOT DECAY
    tot_steps = 0
    exp_return_history = np.zeros(max_steps)
    exp_return = expected_return(env_eval, weights, gamma, episodes_eval)
    pbar = tqdm(total=max_steps)

    while tot_steps < max_steps:
        # Collect data
        data = collect_data(env, weights, sigma, n_episodes=10)  # Collect data from the environment
        r = np.vstack(data["r"])  # Stack returns
        G = np.zeros(r.shape[0])  # Initialize the MC returns
        T = len(r)  # T is the total number of steps collected across all episodes

        # Compute MC returns
        cumulative_sum = 0
        for t in reversed(range(T)):
            if data["done"][t]:  # Check if the episode is done
                cumulative_sum = 0  # Reset cumulative sum at episode end
            cumulative_sum = gamma * cumulative_sum + r[t].item()  # Ensuring r[t] is scalar
            G[t] = cumulative_sum  # Ensuring cumulative_sum is scalar
        
        # Initialize dlog_pi for gradients
        dlog_pi = np.zeros((T, phi_dummy.shape[1], n_actions))  # Shape: (T, n_features, n_actions)
        phi = np.vstack(data["phi"])
        action_d = np.vstack(data["a"])
        dlog_pi = dlog_gaussian_probs(phi, weights, sigma, action_d).transpose(0, 2, 1) 


        # Normalize returns
        if baseline == "none":
            baseline_value = 0
        elif baseline == "mean_return":
            baseline_value = G.mean() * np.ones_like(G)  # Create a baseline of the same shape as G
        elif baseline == "min_variance":
            G_expanded = G[:, np.newaxis, np.newaxis]  # Shape: (705, 1, 1)

            # Now the shapes should be compatible for multiplication
            numerator = np.sum(G_expanded * (dlog_pi ** 2), axis=0)  # Shape: (1, 49, 5)
            denominator = np.sum(dlog_pi ** 2, axis=0)  # Shape: (49, 5)

            # Compute the optimal baseline, avoiding division by zero
            baseline_value = np.sum(numerator) / np.sum(denominator) if np.sum(denominator) != 0 else 0


        G -= baseline_value  # Subtract baseline from returns

        # Compute gradients with returns
        gradient = np.zeros_like(dlog_pi)  # Shape: (T, n_features, n_actions)
        for t in range(T):
            G_t = float(G[t])  # Ensure G[t] is a scalar
            gradient[t] = dlog_pi[t] * G_t  # Element-wise product

        # Average gradient over all samples
        gradient_mean = gradient.mean(0)  # Shape: (n_features, n_actions)


        # Update weights
        weights += alpha * gradient_mean

        # Update return history and total steps
        exp_return_history[tot_steps : tot_steps + T] = exp_return 
        tot_steps += T

        # Evaluate the policy
        exp_return = expected_return(env_eval, weights, gamma, episodes_eval)

        # Decay sigma
        sigma = max(sigma - T / max_steps, 0.1)

        # Progress bar update
        pbar.set_description(f"G: {exp_return:.3f}")
        pbar.update(T)

    pbar.close()
    return exp_return_history


## Auxiliary Functions
# https://stackoverflow.com/a/63458548/754136
def smooth(arr, span):
    re = np.convolve(arr, np.ones(span * 2 + 1) / (span * 2 + 1), mode="same")
    re[0] = arr[0]
    for i in range(1, span + 1):
        re[i] = np.average(arr[: i + span])
        re[-i] = np.average(arr[-i - span :])
    return re

def error_shade_plot(ax, data, stepsize, smoothing_window=1, **kwargs):
    y = np.nanmean(data, 0)
    x = np.arange(len(y))
    x = [stepsize * step for step in range(len(y))]
    if smoothing_window > 1:
        y = smooth(y, smoothing_window)
    (line,) = ax.plot(x, y, **kwargs)
    error = np.nanstd(data, axis=0)
    if smoothing_window > 1:
        error = smooth(error, smoothing_window)
    error = 1.96 * error / np.sqrt(data.shape[0])
    ax.fill_between(x, y - error, y + error, alpha=0.2, linewidth=0.0, color=line.get_color())


## Environment Selection
# Continuous
# env_id = "Pendulum-v1"
# env = gymnasium.make(env_id)
# env_eval = gymnasium.make(env_id)
# episodes_eval = 100
# # you'll solve the Pendulum when the empirical expected return is higher than -150
# # but it can get even higher, eg -120
# state_dim = env.observation_space.shape[0]
# action_dim = env.action_space.shape[0]

# Discrete
# UNCOMMENT TO SOLVE THE GRIDWORLD
env_id = "Gym-Gridworlds/Penalty-3x3-v0"
env = gymnasium.make(env_id, coordinate_observation=True, max_episode_steps=10000)
env_eval = gymnasium.make(env_id, coordinate_observation=True, max_episode_steps=10)  # 10 steps only for faster eval
episodes_eval = 1  # max expected return will be 0.941
state_dim = env.observation_space.shape[0]
n_actions = env.action_space.n

# automatically set centers and sigmas
n_centers = [7] * state_dim
state_low = env.observation_space.low
state_high = env.observation_space.high
centers = np.array(
    np.meshgrid(*[
        np.linspace(
            state_low[i] - (state_high[i] - state_low[i]) / n_centers[i] * 0.1,
            state_high[i] + (state_high[i] - state_low[i]) / n_centers[i] * 0.1,
            n_centers[i],
        )
        for i in range(state_dim)
    ])
).reshape(state_dim, -1).T
sigmas = (state_high - state_low) / np.asarray(n_centers) * 0.75 + 1e-8  # change sigmas for more/less generalization
get_phi = lambda state : rbf_features(state.reshape(-1, state_dim), centers, sigmas)  # reshape because feature functions expect shape (N, S)
phi_dummy = get_phi(env.reset()[0])  # to get the number of features

# hyperparameters
gamma = 0.99
alpha = 0.1
episodes_per_update = 10
max_steps = 100000  # 100000 for the Gridworld
baselines = ["none", "mean_return", "min_variance"]
n_seeds = 2
results_exp_ret = np.zeros((
    len(baselines),
    n_seeds,
    max_steps,
))

fig, axs = plt.subplots(1, 1)
axs.set_prop_cycle(color=["red", "green", "blue"])
axs.set_xlabel("Steps")
axs.set_ylabel("Expected Return")

for i, baseline in enumerate(baselines):
    for seed in range(n_seeds):
        np.random.seed(seed)
        exp_return_history = reinforce(baseline)
        results_exp_ret[i, seed] = exp_return_history
        print(baseline, seed)

    plot_args = dict(
        stepsize=1,
        smoothing_window=20,
        label=baseline,
    )
    error_shade_plot(
        axs,
        results_exp_ret[i],
        **plot_args,
    )
    axs.legend()

plt.show()




