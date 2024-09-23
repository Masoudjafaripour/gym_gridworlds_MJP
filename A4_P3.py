import gymnasium
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


np.set_printoptions(precision=3)

# Set up the environment
env = gymnasium.make("Gym-Gridworlds/Penalty-3x3-v0")
n_states = env.observation_space.n
n_actions = env.action_space.n

# Initialize reward matrix, transition probabilities, and terminal states
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
#------------------------------------------------------------------------------------------------------
# Bellman Equation Function
def bellman_q(pi, gamma):
    I = np.eye(n_states * n_actions)
    P_under_pi = (P[..., None] * pi[None, None]).reshape(n_states * n_actions, n_states * n_actions)
    return (R.ravel() * np.linalg.inv(I - gamma * P_under_pi)).sum(-1).reshape(n_states, n_actions)

# Epsilon-greedy action selection
def eps_greedy_action(Q, s, eps):
    if np.random.rand() < eps:
        return np.random.choice(np.arange(len(Q[s])))  # random action
    else:
        return np.argmax(Q[s])  # greedy action
#------------------------------------------------------------------------------------------------------

# Epsilon-greedy policy
def eps_greedy_probs(Q, eps):
    probs = np.ones((n_states, n_actions)) * (eps / n_actions)
    for s in range(n_states):
        if s in Q and len(Q[s]) == n_actions:  # Ensure Q[s] is initialized
            best_action = np.argmax(Q[s])
            probs[s, best_action] += (1 - eps)
    return probs
#------------------------------------------------------------------------------------------------------

# Collect an episode following the current policy
def episode(env, Q, eps, seed):
    np.random.seed(seed)  # for reproducibility
    data = {"s": [], "a": [], "r": []}
    s, _ = env.reset(seed=seed)
    done = False
    while not done:
        a = eps_greedy_action(Q, s, eps)
        s_next, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        data["s"].append(s)
        data["a"].append(a)
        data["r"].append(r)
        s = s_next
    return data
#------------------------------------------------------------------------------------------------------

# Compute the Bellman error for a given policy
def compute_bellman_error(Q, pi, gamma):
    Q_true = bellman_q(pi, gamma)
    bellman_error = np.abs(Q - Q_true).sum()  # sum -->> Max absolute Bellman error
    return bellman_error
#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------

# Monte Carlo Control with On-policy E-soft Policy
def monte_carlo(env, Q, gamma, eps_decay, max_steps, episodes_per_iteration):
    eps = 1.0
    total_steps = 0
    bellman_errors = []
    pi = eps_greedy_probs(Q, eps) 
    Q_true = bellman_q(pi, gamma)
    bellman_error = np.abs(Q - Q_true).sum()
    bellman_errors.append(bellman_error)

    while total_steps < max_steps:
        visit_count = np.zeros((n_states, n_actions))  # This stores how many times (state, action) has been visited
        returns = np.zeros((n_states, n_actions)) 

        for _ in range(episodes_per_iteration):
            # Generate episodes
            episode_data = episode(env, Q, eps, int(seed))
            G = 0
            W = 1

            # Loop over episode steps in reverse (every-visit MC)
            for t in range(len(episode_data["s"]) - 1, -1, -1):
                state, action, reward = episode_data["s"][t], episode_data["a"][t], episode_data["r"][t]
                G = gamma * G + reward
                    
                # Update the visit count
                returns[state, action] += G
                visit_count[(state, action)] += 1
                
                # Calculate the running mean for the Q-value
                Q[state][action] = returns[(state, action)]/visit_count[(state, action)] # Incremental mean update

                bellman_errors.append(bellman_error)
            
            # Decay epsilon after each episode
            eps = max(eps - eps_decay * len(episode_data["s"]) / max_steps, 0.01)
            total_steps += len(episode_data["s"])           

        # Update the epsilon-greedy policy progressively
        pi = eps_greedy_probs(Q, eps)
        Q_true = bellman_q(pi, gamma)
        bellman_error = np.abs(Q - Q_true).sum()


    # Ensure bellman_errors has the same length as max_steps
    if len(bellman_errors) > max_steps:
        bellman_errors = bellman_errors[:max_steps]  # Trim to match max_steps
    else:
        # Pad with the last bellman error if the length is shorter
        bellman_errors += [bellman_errors[-1]] * (max_steps - len(bellman_errors))

    return Q, bellman_errors 


def off_policy_mc_control(env, Q, gamma, eps_decay, max_steps, episodes_per_iteration):
    eps = 1.0
    total_steps = 0
    bellman_errors = []
    C = np.zeros_like(Q)  # To store cumulative weights for each (state, action)
    pi = eps_greedy_probs(Q, eps)  # Greedy policy
    Q_true = bellman_q(pi, gamma)
    bellman_error = np.abs(Q - Q_true).sum()
    bellman_errors.append(bellman_error)
    
    while total_steps < max_steps:
        for _ in range(episodes_per_iteration):
            # Generate an episode using a soft behavior policy `b`
            episode_data = episode(env, Q, eps, int(seed))  # Assume `episode()` returns states, actions, rewards
            G = 0
            W = 1

            # Loop over episode steps in reverse
            for t in range(len(episode_data["s"]) - 1, -1, -1):
                state, action, reward = episode_data["s"][t], episode_data["a"][t], episode_data["r"][t]
                G = gamma * G + reward

                # Update cumulative weight
                C[state][action] += W
                
                # Update Q-value with weighted importance sampling
                Q[state][action] += W / C[state][action] * (G - Q[state][action])
                
                # Update the greedy policy `pi`
                pi[state] = np.argmax(Q[state])  # Greedy action for current state

                # If action taken is not the greedy action, stop updating
                if action != pi[state]:
                    break
                
                # Update importance sampling weight
                W *= 1.0 / eps_greedy_probs(Q, eps)[action][state]

            # Decay epsilon after each episode (soft behavior policy)
            eps = max(eps - eps_decay * len(episode_data["s"]) / max_steps, 0.01)
            total_steps += len(episode_data["s"])

        # Update the epsilon-greedy policy progressively
        pi = eps_greedy_probs(Q, eps)
        Q_true = bellman_q(pi, gamma)
        bellman_error = np.abs(Q - Q_true).sum()
        bellman_errors.append(bellman_error)

    # Ensure bellman_errors has the same length as max_steps
    if len(bellman_errors) > max_steps:
        bellman_errors = bellman_errors[:max_steps]  # Trim to match max_steps
    else:
        # Pad with the last bellman error if the length is shorter
        bellman_errors += [bellman_errors[-1]] * (max_steps - len(bellman_errors))

    return Q, bellman_errors

#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------

# Error plot with confidence interval
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
max_steps = 2000 #2000
horizon = 10

episodes_per_iteration = [1, 10, 50] # for updating Q after episodes_per_iteration
decays = [0.5, 1, 2] #[1, 2, 5]
seeds = np.arange(10) #50

# Initialize the results array
results = np.zeros((
    len(episodes_per_iteration),
    len(decays),
    len(seeds),
    max_steps,
))

# Initialize the plot
fig, axs = plt.subplots(1, 2)
plt.ion()
plt.show()

use_is = False  # Not using Importance Sampling for Part 1
for ax, reward_noise_std in zip(axs, [0.0, 3.0]):  # Two subplots: one with reward noise, one without
    ax.set_prop_cycle(
        color=["red", "green", "blue", "black", "orange", "cyan", "brown", "gray", "pink"]
    )
    ax.set_xlabel("Steps")
    ax.set_ylabel("Absolute Bellman Error")
    
    # Create environment with reward noise if applicable
    env = gymnasium.make(
        "Gym-Gridworlds/Penalty-3x3-v0",
        max_episode_steps=horizon,
        reward_noise_std=reward_noise_std,
    )
    
    # Run the experiment for all combinations of episodes per iteration and decay
    for j, episodes_p in enumerate(episodes_per_iteration):
        for k, decay in enumerate(decays):
            for seed in seeds:
                np.random.seed(seed)
                Q = np.zeros((n_states, n_actions)) + init_value
                Q, be = monte_carlo(env, Q, gamma, decay / 1, max_steps, episodes_p) #/max_steps
                results[j, k, seed] = be
                print(k, j, seed)

            error_shade_plot(
                ax,
                results[j, k],
                stepsize=1,
                label=f"Episodes: {episodes_p}, Decay: {decay}",
            )
            ax.legend()
            plt.draw()
            plt.pause(0.001)


print("done")

plt.ioff()
plt.show()


