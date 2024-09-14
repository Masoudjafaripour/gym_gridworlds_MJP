import gymnasium
import numpy as np
import matplotlib.pyplot as plt

# Create the environment
env = gymnasium.make("Gym-Gridworlds/Penalty-3x3-v0")

# Initialize parameters
n_states = env.observation_space.n
n_actions = env.action_space.n

R = np.zeros((n_states, n_actions))
P = np.zeros((n_states, n_actions, n_states))
T = np.zeros((n_states, n_actions))

# Populate R, P, and T matrices
env.reset()
for s in range(n_states):
    for a in range(n_actions):
        env.set_state(s)
        s_next, r, terminated, _, _ = env.step(a)
        R[s, a] = r
        if terminated:
            P[s, a, s_next] = 1.0
        else:
            P[s, a, s_next] = 1.0
        T[s, a] = terminated

# Define the optimal policy
optimal_policy = np.zeros(n_states, dtype=int)
optimal_policy[0] = 1  # Move DOWN from state 0 to 3
optimal_policy[3] = 1  # Move DOWN from state 3 to 6
optimal_policy[6] = 2  # Move RIGHT from state 6 to 7
optimal_policy[7] = 2  # Move RIGHT from state 7 to 8
optimal_policy[8] = 3  # Move UP from state 8 to 5
optimal_policy[5] = 3  # Move UP from state 5 to 2
optimal_policy[2] = 4  # STAY at state 2

def bellman_v(policy, gamma, init_value, max_iterations=10000, tol=1e-7):
    V = np.full(n_states, init_value)  # Initialize V with init_value
    bellman_errors = []
    for _ in range(max_iterations):
        V_prev = V.copy()
        for s in range(n_states):
            a = policy[s]
            V[s] = np.sum(P[s, a, :] * (R[s, a] + gamma * V_prev))
        error = np.max(np.abs(V - V_prev))
        bellman_errors.append(error)
        if error < tol:
            break
    return V, bellman_errors

def bellman_q(gamma, init_value, max_iterations=1000, tol=1e-7):
    Q = np.full((n_states, n_actions), init_value)  # Initialize Q with init_value
    V = np.full(n_states, init_value)  # Initialize V with init_value
    for _ in range(max_iterations):
        V_prev = V.copy()
        for s in range(n_states):
            for a in range(n_actions):
                Q[s, a] = np.sum(P[s, a, :] * (R[s, a] + gamma * V_prev))
            V[s] = np.max(Q[s, :])
        if np.max(np.abs(V - V_prev)) < tol:
            break
    return Q

# Evaluation and plotting
gammas = [0.01, 0.5, 0.99]
for init_value in [-10, 0, 10]:
    fig, axs = plt.subplots(2, len(gammas), figsize=(12, 8))
    fig.suptitle(f'Value Function ($V$) - Initial Value: {init_value}', fontsize=16)
    for i, gamma in enumerate(gammas):
        V, bellman_errors_v = bellman_v(optimal_policy, gamma, init_value)
        
        # Reshape V for visualization
        V_reshaped = V.reshape((3, 3))
        cax = axs[0][i].imshow(V_reshaped, cmap='viridis')
        axs[0][i].set_title(f'$\gamma$ = {gamma}', fontsize=14)
        axs[0][i].set_xticks([])
        axs[0][i].set_yticks([])
        fig.colorbar(cax, ax=axs[0][i])
        
        axs[1][i].plot(bellman_errors_v, marker='o')
        axs[1][i].set_title('Bellman Error', fontsize=16)
        axs[1][i].set_xlabel('Iteration', fontsize=14)
        axs[1][i].set_ylabel('Total Absolute Bellman Error', fontsize=14)
        axs[1][i].tick_params(axis='both', labelsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    fig, axs = plt.subplots(n_actions, len(gammas), figsize=(12, 15))
    fig.suptitle(f'Action-Value Function ($Q$) - Initial Value: {init_value}', fontsize=16)
    for i, gamma in enumerate(gammas):
        Q = bellman_q(gamma, init_value)
        
        # Plot Q-values for each action
        for a in range(n_actions):
            Q_reshaped = Q[:, a].reshape((3, 3))
            cax = axs[a][i].imshow(Q_reshaped, cmap='viridis')
            axs[a][i].set_title(f'Action {a}', fontsize=14)
            axs[a][i].set_xticks([])
            axs[a][i].set_yticks([])
            fig.colorbar(cax, ax=axs[a][i])
        
        axs[0][i].set_title(f'$\gamma$ = {gamma}', fontsize=14, pad=20)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
