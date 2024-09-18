import gymnasium
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=3)

env = gymnasium.make("Gym-Gridworlds/Penalty-3x3-v0")
n_states = env.observation_space.n
n_actions = env.action_space.n

R = np.zeros((n_states, n_actions))
P = np.zeros((n_states, n_actions, n_states))
T = np.zeros((n_states, n_actions))

env.reset()
for s in range(n_states):
    for a in range(n_actions):
        env.unwrapped.set_state(s)
        s_next, r, terminated, _, _ = env.step(a)
        R[s, a] = r
        P[s, a, s_next] = 1.0
        T[s, a] = terminated

P = P * (1.0 - T[..., None])  # next state probability for terminal transitions is 0

# Functions
def bellman_q(pi, gamma):
    I = np.eye(n_states * n_actions)

    P_under_pi = (P[..., None] * pi[None, None]).reshape(n_states * n_actions, n_states * n_actions) # ? in PE? 

    return (R.ravel() * np.linalg.inv(I - gamma * P_under_pi)).sum(-1).reshape(n_states, n_actions)

def episode(env, Q, eps, seed):
    data = dict()
    data["s"] = []
    data["a"] = []
    data["r"] = []
    s, _ = env.reset(seed=seed)
    done = False
    while not done:
        a = eps_greedy_action(Q[s], eps)
        s_next, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        data["s"].append(s)
        data["a"].append(a)
        data["r"].append(r)
        s = s_next
    return data


def eps_greedy_probs(Q, eps):

    return action # probabilities

def eps_greedy_action(Q, eps):
    ...
    return action # drawn according to eps-greedy policy

def monte_carlo(env, Q, gamma, eps_decay, max_steps, episodes_per_iteration, use_is):
    eps = max(eps - decay_factor / max_steps * episode_steps, 0.01)

    
    return Q, be


def error_shade_plot(ax, data, stepsize, **kwargs):
    y = np.nanmean(data, 0)
    x = np.arange(len(y))
    x = [stepsize * step for step in range(len(y))]
    (line,) = ax.plot(x, y, **kwargs)
    error = np.nanstd(data, axis=0)
    error = 1.96 * error / np.sqrt(data.shape[0])
    ax.fill_between(x, y - error, y + error, alpha=0.2, linewidth=0.0, color=line.get_color())

# Hyperparameters
init_value = 0.0
gamma = 0.9
max_steps = 2000
horizon = 10

episodes_per_iteration = [1, 10, 50]
decays = [1, 2, 5]
seeds = np.arange(50)

results = np.zeros((
    len(episodes_per_iteration),
    len(decays),
    len(seeds),
    max_steps,
))

# Plots
fig, axs = plt.subplots(1, 2)
plt.ion()
plt.show()

# Loops
use_is = False  # repeat with True
for ax, reward_noise_std in zip(axs, [0.0, 3.0]):
    ax.set_prop_cycle(
        color=["red", "green", "blue", "black", "orange", "cyan", "brown", "gray", "pink"]
    )
    ax.set_xlabel("Steps")
    ax.set_ylabel("Absolute Bellman Error")
    env = gymnasium.make(
        "Gym-Gridworlds/Penalty-3x3-v0",
        max_episode_steps=horizon,
        reward_noise_std=reward_noise_std,
    )
    for j, episodes in enumerate(episodes_per_iteration):
        for k, decay in enumerate(decays):
            for seed in seeds:
                np.random.seed(seed)
                Q = np.zeros((n_states, n_actions)) + init_value
                Q, be = monte_carlo(env, Q, gamma, decay / max_steps, max_steps, episodes, use_is)
                results[j, k, seed] = be
            error_shade_plot(
                ax,
                results[j, k],
                stepsize=1,
                label=f"Episodes: {episodes}, Decay: {decay}",
            )
            ax.legend()
            plt.draw()
            plt.pause(0.001)

plt.ioff()
plt.show()

