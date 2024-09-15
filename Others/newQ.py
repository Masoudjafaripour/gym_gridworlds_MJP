import numpy as np
import matplotlib.pyplot as plt
import gymnasium

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

# RL Methods
# Policy Evaluation using Q-values
def policy_evaluation(policy, gamma, init_value, max_iterations=10000, tol=1e-7):
    Q = np.full((n_states, n_actions), init_value)
    bellman_errors = []
    for _ in range(max_iterations):
        Q_prev = Q.copy()
        for s in range(n_states):
            a = policy[s]
            Q[s, a] = np.sum(P[s, a, :] * (R[s, a] + gamma * np.max(Q, axis=1)))
        error = np.max(np.abs(Q - Q_prev))
        bellman_errors.append(error)
        if error < tol:
            break
    return Q, bellman_errors

# Policy Improvement using Q-values
def policy_improvement(Q, gamma):
    policy = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        action_values = np.zeros(n_actions)
        for a in range(n_actions):
            action_values[a] = np.sum(P[s, a, :] * (R[s, a] + gamma * np.max(Q, axis=1)))
        policy[s] = np.argmax(action_values)
    return policy

# Policy Iteration
def policy_iteration(gamma, init_value, max_policy_evaluations=1000):
    policy = np.random.choice(n_actions, size=n_states, p=[1/n_actions]*n_actions)
    total_policy_evaluations = 0
    all_bellman_errors = []

    while True:
        Q, bellman_errors = policy_evaluation(policy, gamma, init_value)
        all_bellman_errors.extend(bellman_errors)
        total_policy_evaluations += len(bellman_errors)
        
        new_policy = policy_improvement(Q, gamma)
        if np.all(policy == new_policy):
            break
        policy = new_policy

    return policy, total_policy_evaluations, all_bellman_errors

# Generalized Policy Iteration
def generalized_policy_iteration(gamma, init_value, max_iterations=10000, eval_steps=5):
    policy = np.random.choice(n_actions, size=n_states, p=[1/n_actions]*n_actions)

    Q = np.full((n_states, n_actions), init_value)
    total_policy_evaluations = 0
    all_bellman_errors = []

    for _ in range(max_iterations):
        for _ in range(eval_steps):
            Q_prev, bellman_errors = policy_evaluation(policy, gamma, init_value)
            all_bellman_errors.extend(bellman_errors)
            total_policy_evaluations += len(bellman_errors)
            if np.max(np.abs(Q - Q_prev)) < 1e-7:
                break
            Q = Q_prev

        new_policy = policy_improvement(Q, gamma)
        if np.all(policy == new_policy):
            break
        policy = new_policy

    return policy, total_policy_evaluations, all_bellman_errors

# Value Iteration
def value_iteration(gamma, init_value, max_iterations=10000):
    Q = np.full((n_states, n_actions), init_value)
    all_bellman_errors = []

    for _ in range(max_iterations):
        Q_prev = Q.copy()
        for s in range(n_states):
            for a in range(n_actions):
                Q[s, a] = np.sum(P[s, a, :] * (R[s, a] + gamma * np.max(Q, axis=1)))
        V = np.max(Q, axis=1)
        error = np.max(np.abs(Q - Q_prev))
        all_bellman_errors.append(error)
        if error < 1e-7:
            break

    policy = policy_improvement(Q, gamma)
    return policy, len(all_bellman_errors), all_bellman_errors

# Define the optimal policy (based on your description)
pi_opt = np.zeros(n_states, dtype=int)
pi_opt[0] = 1  # Move DOWN from state 0 to 3
pi_opt[3] = 1  # Move DOWN from state 3 to 6
pi_opt[6] = 2  # Move RIGHT from state 6 to 7
pi_opt[7] = 2  # Move RIGHT from state 7 to 8
pi_opt[8] = 3  # Move UP from state 8 to 5
pi_opt[5] = 3  # Move UP from state 5 to 2
pi_opt[2] = 4  # STAY at state 2
pi_opt[1] = 2  # Move RIGHT from state 1 to 2
pi_opt[4] = 2  # Move RIGHT from state 4 to 5

# Initialize plot
fig, axs = plt.subplots(3, 7, figsize=(20, 12))
fig.suptitle('Bellman Error Across Algorithms and Initial Q Values', fontsize=16, y=1.02)  # Overall title
tot_iter_table = np.zeros((3, 7))
bellman_error_trends = {'VI': [], 'PI': [], 'GPI': []}
algorithms = {'VI': value_iteration, 'PI': policy_iteration, 'GPI': generalized_policy_iteration}

for i, init_value in enumerate([-100, -10, -5, 0, 5, 10, 100]):
    for algo_name, algo_func in algorithms.items():
        policy, tot_iter, bellman_errors = algo_func(gamma=0.99, init_value=init_value)
        tot_iter_table[{'VI': 0, 'PI': 1, 'GPI': 2}[algo_name], i] = tot_iter
        # Check if the policy is close to the optimal policy
        if np.allclose(policy, pi_opt):
            print(f'Algorithm {algo_name} with initialization {init_value}: Policy is close to optimal.')
        else:
            print(f'Algorithm {algo_name} with initialization {init_value}: Policy is NOT close to optimal.')
            print("\n", "Learned Policy = ", policy, "\n", "Optimal Policy = ", pi_opt)

        # Plot Bellman Error Trend
        row = {'VI': 0, 'PI': 1, 'GPI': 2}[algo_name]
        title = f'Q_0: {init_value}' if row == 0 else ''  # Only add Q_0 title in the first row
        axs[row, i].plot(bellman_errors, marker='o')
        axs[row, i].set_title(title)
        axs[row, i].set_xlabel('Policy Evaluation Iteration')
        axs[row, i].set_ylabel('Bellman Error')

# Add labels for each row
for i, algo_name in enumerate(['VI', 'PI', 'GPI']):
    axs[i, 0].set_ylabel(algo_name)

plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.96])  # Adjust layout to fit suptitle
plt.show()

# Report results
mean_iter = np.mean(tot_iter_table, axis=1)
std_iter = np.std(tot_iter_table, axis=1)
for algo, mean, std in zip(['VI', 'PI', 'GPI'], mean_iter, std_iter):
    print(f'{algo}: Mean iterations = {mean}, Std = {std}')
