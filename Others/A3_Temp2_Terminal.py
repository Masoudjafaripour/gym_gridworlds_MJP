import gymnasium
import numpy as np
import matplotlib.pyplot as plt

# Create the environment
env = gymnasium.make("Gym-Gridworlds/Penalty-3x3-v0")

# Initialize parameters
n_states = env.observation_space.n
n_actions = env.action_space.n

# Add an extra state to represent the terminal state
R = np.zeros((n_states + 1, n_actions))
P = np.zeros((n_states + 1, n_actions, n_states + 1))
T = np.zeros((n_states, n_actions))  # No need to include terminal state in T

# Populate R, P, and T matrices
env.reset()
for s in range(n_states):
    for a in range(n_actions):
        env.set_state(s)
        s_next, r, terminated, _, _ = env.step(a)
        R[s, a] = r
        if terminated:
            P[s, a, n_states] = 1.0  # Transition to the terminal state
        else:
            P[s, a, s_next] = 1.0
        T[s, a] = terminated  # Mark if the transition is terminal

# Define the optimal policy
optimal_policy = np.zeros(n_states, dtype=int)
optimal_policy[0] = 1  # Move DOWN from state 0 to 3
optimal_policy[3] = 1  # Move DOWN from state 3 to 6
optimal_policy[6] = 2  # Move RIGHT from state 6 to 7
optimal_policy[7] = 2  # Move RIGHT from state 7 to 8
optimal_policy[8] = 3  # Move UP from state 8 to 5
optimal_policy[5] = 3  # Move UP from state 5 to 2
optimal_policy[2] = 4  # STAY at state 2

def policy_evaluation(policy, gamma, init_value, max_iterations=10000, tol=1e-7):
    V = np.full(n_states + 1, init_value)
    bellman_errors = []
    
    for _ in range(max_iterations):
        V_prev = V.copy()
        
        for s in range(n_states):
            a = policy[s]
            if T[s, a] == 1:  # Skip terminal states
                continue
            V[s] = np.sum(P[s, a, :] * (R[s, a] + gamma * V))
        
        error = np.max(np.abs(V[:n_states] - V_prev[:n_states]))
        bellman_errors.append(error)
        
        if error < tol:
            break
    
    return V[:n_states], bellman_errors

def policy_improvement(V, gamma):
    policy = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        action_values = np.zeros(n_actions)
        for a in range(n_actions):
            if T[s, a] == 1:
                action_values[a] = R[s, a]  # Direct reward if it's terminal
            else:
                action_values[a] = np.sum(P[s, a, :] * (R[s, a] + gamma * np.concatenate([V, [0]])))
        policy[s] = np.argmax(action_values)
    return policy

def policy_iteration(gamma, init_value, max_policy_evaluations=1000):
    policy = np.random.randint(0, n_actions, size=n_states)
    total_policy_evaluations = 0
    all_bellman_errors = []

    while True:
        V, bellman_errors = policy_evaluation(policy, gamma, init_value)
        all_bellman_errors.extend(bellman_errors)
        total_policy_evaluations += len(bellman_errors)
        
        new_policy = policy_improvement(V, gamma)
        if np.all(policy == new_policy):
            break
        policy = new_policy

    return policy, total_policy_evaluations, all_bellman_errors

def generalized_policy_iteration(gamma, init_value, max_iterations=1000, eval_steps=5):
    policy = np.random.randint(0, n_actions, size=n_states)
    V = np.full(n_states + 1, init_value)
    total_policy_evaluations = 0
    all_bellman_errors = []

    for _ in range(max_iterations):
        for _ in range(eval_steps):
            V_prev, bellman_errors = policy_evaluation(policy, gamma, init_value)
            all_bellman_errors.extend(bellman_errors)
            total_policy_evaluations += len(bellman_errors)
            if np.max(np.abs(V - V_prev)) < 1e-7:
                break
            V = V_prev

        new_policy = policy_improvement(V, gamma)
        if np.all(policy == new_policy):
            break
        policy = new_policy

    return policy, total_policy_evaluations, all_bellman_errors

def value_iteration(gamma, init_value, max_iterations=1000):
    V = np.full(n_states + 1, init_value)
    all_bellman_errors = []

    for _ in range(max_iterations):
        V_prev = V.copy()
        for s in range(n_states):
            action_values = np.zeros(n_actions)
            for a in range(n_actions):
                if T[s, a] == 1:
                    action_values[a] = R[s, a]  # Direct reward if it's terminal
                else:
                    action_values[a] = np.sum(P[s, a, :] * (R[s, a] + gamma * np.concatenate([V, [0]])))
            V[s] = np.max(action_values)
        
        error = np.max(np.abs(V[:n_states] - V_prev[:n_states]))
        all_bellman_errors.append(error)
        if error < 1e-7:
            break

    policy = policy_improvement(V, gamma)
    return policy, len(all_bellman_errors), all_bellman_errors

# Initialize plot
fig, axs = plt.subplots(3, 7, figsize=(20, 12))
tot_iter_table = np.zeros((3, 7))
bellman_error_trends = {'VI': [], 'PI': [], 'GPI': []}
algorithms = {'VI': value_iteration, 'PI': policy_iteration, 'GPI': generalized_policy_iteration}

# Run algorithms
for i, init_value in enumerate([-100, -10, -5, 0, 5, 10, 100]):
    for algo_name, algo_func in algorithms.items():
        policy, tot_iter, bellman_errors = algo_func(gamma=0.99, init_value=init_value)
        tot_iter_table[{'VI': 0, 'PI': 1, 'GPI': 2}[algo_name], i] = tot_iter
        # assert np.allclose(policy, optimal_policy)

        # Plot Bellman Error Trend
        axs[{'VI': 0, 'PI': 1, 'GPI': 2}[algo_name]][i].plot(bellman_errors, marker='o')
        axs[{'VI': 0, 'PI': 1, 'GPI': 2}[algo_name]][i].set_title(f'Init: {init_value}')
        axs[{'VI': 0, 'PI': 1, 'GPI': 2}[algo_name]][i].set_xlabel('Policy Evaluation Iteration')
        axs[{'VI': 0, 'PI': 1, 'GPI': 2}[algo_name]][i].set_ylabel('Bellman Error')

plt.tight_layout()
plt.show()

# Report results
mean_iter = np.mean(tot_iter_table, axis=1)
std_iter = np.std(tot_iter_table, axis=1)
for algo, mean, std in zip(['VI', 'PI', 'GPI'], mean_iter, std_iter):
    print(f'{algo}: Mean iterations = {mean}, Std = {std}')
