import gymnasium
import numpy as np
import matplotlib.pyplot as plt

env = gymnasium.make("Gym-Gridworlds/Penalty-3x3-v0")

# Environment parameters
n_states = env.observation_space.n
n_actions = env.action_space.n

# Transition matrix P, rewards R, and terminal flags T
R = np.zeros((n_states, n_actions))
P = np.zeros((n_states, n_actions, n_states))
T = np.zeros((n_states, n_actions))

# Populate P, R, and T using environment dynamics
env.reset()
for s in range(n_states):
    for a in range(n_actions):
        env.set_state(s)
        s_next, r, terminated, _, _ = env.step(a)
        R[s, a] = r
        P[s, a, s_next] = 1.0
        T[s, a] = terminated

# Bellman evaluation for state value function V
def bellman_v(P, R, T, gamma, V, log=False):
    delta = 0
    V_new = np.copy(V)
    for s in range(n_states):
        V_new[s] = sum([P[s, a, s_next] * (R[s, a] + gamma * V[s_next] * (not T[s, a]))
                        for a in range(n_actions) for s_next in range(n_states)])
        delta += abs(V_new[s] - V[s])
    if log:
        print(f"Total Bellman error: {delta}")
    return V_new, delta

# Bellman evaluation for action-value function Q
def bellman_q(P, R, T, gamma, Q, log=False):
    delta = 0
    Q_new = np.copy(Q)
    for s in range(n_states):
        for a in range(n_actions):
            Q_new[s, a] = sum([P[s, a, s_next] * (R[s, a] + gamma * np.max(Q[s_next, :]) * (not T[s, a]))
                               for s_next in range(n_states)])
            delta += abs(Q_new[s, a] - Q[s, a])
    if log:
        print(f"Total Bellman error: {delta}")
    return Q_new, delta

# Running experiments with different initializations
gammas = [0.01, 0.5, 0.99]
initializations = [-10, 0, 10]

for init_value in initializations:
    V = np.full(n_states, init_value)
    Q = np.full((n_states, n_actions), init_value)

    fig, axs = plt.subplots(2, len(gammas))
    fig.suptitle(f"V_0: {init_value}")
    
    for i, gamma in enumerate(gammas):
        bellman_errors_v = []
        V_current = np.copy(V)
        while True:
            V_next, error = bellman_v(P, R, T, gamma, V_current)
            bellman_errors_v.append(error)
            if error < 1e-5:
                break
            V_current = V_next

        axs[0][i].imshow(V_current.reshape(3, 3), cmap="coolwarm")
        axs[1][i].plot(bellman_errors_v)
        axs[0][i].set_title(f'γ = {gamma}')

    fig, axs = plt.subplots(n_actions + 1, len(gammas))
    fig.suptitle(f"Q_0: {init_value}")
    
    for i, gamma in enumerate(gammas):
        bellman_errors_q = []
        Q_current = np.copy(Q)
        while True:
            Q_next, error = bellman_q(P, R, T, gamma, Q_current)
            bellman_errors_q.append(error)
            if error < 1e-5:
                break
            Q_current = Q_next

        for a in range(n_actions):
            axs[a][i].imshow(Q_current[:, a].reshape(3, 3), cmap="coolwarm")
        axs[-1][i].plot(bellman_errors_q)
        axs[0][i].set_title(f'γ = {gamma}')
    
    plt.show()
