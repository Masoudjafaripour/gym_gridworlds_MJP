import gymnasium
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=3)

# Define constants
init_value = 0.0
gamma = 0.9
max_steps = 2000
horizon = 10
episodes_per_iteration = [1, 10, 50]
decays = [1, 2, 5]
seeds = np.arange(10)

# Initialize environment and matrices
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

def bellman_q(pi, gamma):
    I = np.eye(n_states * n_actions)
    P_under_pi = (P[..., None] * pi[None, None]).reshape(n_states * n_actions, n_states * n_actions)
    return (R.ravel() * np.linalg.inv(I - gamma * P_under_pi)).sum(-1).reshape(n_states, n_actions)

def episode(env, Q, eps, seed):
    data = {"s": [], "a": [], "r": []}
    s, _ = env.reset(seed=seed)
    done = False
    while not done:
        a = eps_greedy_action(Q, s, eps)
        s_next, r, terminated, truncated, _ = env.step(a)
        # Add stochastic noise to reward
        r += np.random.normal(0, 3.0)
        done = terminated or truncated
        data["s"].append(s)
        data["a"].append(a)
        data["r"].append(r)
        s = s_next
    return data

def eps_greedy_probs(Q, eps):
    n_states = Q.shape[0]
    n_actions = Q.shape[1]
    
    # Initialize the probabilities array for each state
    probs = np.ones((n_states, n_actions)) * eps / n_actions
    
    for s in range(n_states):
        # Create action probabilities for each state
        action_probs = np.ones(n_actions) * eps / n_actions
        best_action = np.argmax(Q[s])
        action_probs[best_action] += (1.0 - eps)
        
        # Assign the probabilities to the corresponding state
        probs[s] = action_probs
        
    return probs

def eps_greedy_action(Q, s, eps):
    action_probs = eps_greedy_probs(Q, eps)
    return np.random.choice(n_actions, p=action_probs[s])

def compute_bellman_error(Q, pi, gamma):
    Q_true = bellman_q(pi, gamma)
    bellman_error = np.abs(Q - Q_true).mean()
    return bellman_error

def monte_carlo(env, Q, gamma, eps_decay, max_steps, episodes_per_iteration, use_is):
    total_steps = 0
    eps = 1.0
    bellman_errors = []
    
    while total_steps < max_steps:
        episodes_data = []
        for _ in range(episodes_per_iteration):
            episode_data = episode(env, Q, eps, seed=np.random.randint(1000))
            episodes_data.append(episode_data)
            total_steps += len(episode_data["s"])
            
            eps = max(eps - eps_decay * len(episode_data["s"]) / max_steps, 0.01)
        
        if use_is:
            for data in episodes_data:
                returns = np.zeros((n_states, n_actions))
                weight = np.ones((n_states, n_actions))
                
                for t in reversed(range(len(data["s"]))):
                    s, a, r = data["s"][t], data["a"][t], data["r"][t]
                    returns[s, a] += (gamma ** t) * r
                    weight[s, a] *= 1.0 / eps_greedy_probs(Q, eps)[s, a]
                    
                Q += (returns * weight - Q) / len(data["s"])
        
        pi = eps_greedy_probs(Q, eps)
        bellman_error = compute_bellman_error(Q, pi, gamma)
        bellman_errors.append(bellman_error)
    
    if len(bellman_errors) < max_steps:
        bellman_errors.extend([bellman_errors[-1]] * (max_steps - len(bellman_errors)))
    
    return Q, np.array(bellman_errors[:max_steps])

def error_shade_plot(ax, data, stepsize, **kwargs):
    y = np.nanmean(data, 0)
    x = np.arange(len(y))
    x = [stepsize * step for step in range(len(y))]
    (line,) = ax.plot(x, y, **kwargs)
    error = np.nanstd(data, axis=0)
    error = 1.96 * error / np.sqrt(data.shape[0])
    ax.fill_between(x, y - error, y + error, alpha=0.2, linewidth=0.0, color=line.get_color())

results = np.zeros((len(episodes_per_iteration), len(decays), len(seeds), max_steps))

fig, axs = plt.subplots(1, 2)
plt.ion()
plt.show()

use_is = True  # Set to True for importance sampling
for ax, reward_noise_std in zip(axs, [0.0, 3.0]):
    ax.set_prop_cycle(color=["red", "green", "blue", "black", "orange", "cyan", "brown", "gray", "pink"])
    ax.set_xlabel("Steps")
    ax.set_ylabel("Absolute Bellman Error")
    
    env = gymnasium.make("Gym-Gridworlds/Penalty-3x3-v0", max_episode_steps=horizon, reward_noise_std=reward_noise_std)
    
    for j, episodes in enumerate(episodes_per_iteration):
        for k, decay in enumerate(decays):
            for seed in seeds:
                np.random.seed(seed)
                Q = np.zeros((n_states, n_actions)) + init_value
                Q, bellman_error_trend = monte_carlo(env, Q, gamma, decay / max_steps, max_steps, episodes, use_is)
                results[j, k, seed] = bellman_error_trend
            
            error_shade_plot(ax, results[j, k], stepsize=1, label=f"Episodes: {episodes}, Decay: {decay}")
            ax.legend()
            plt.draw()
            plt.pause(0.001)

plt.ioff()
plt.show()
