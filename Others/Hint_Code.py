import gymnasium
import numpy as np
import matplotlib.pyplot as plt

# Create the environment
env = gymnasium.make("Gym-Gridworlds/Penalty-3x3-v0")

# Initialize parameters
n_states = env.observation_space.n
n_actions = env.action_space.n

# Add an extra state to represent the terminal state
R = np.zeros((n_states, n_actions))
P = np.zeros((n_states, n_actions, n_states))
T = np.zeros((n_states, n_actions))  # No need to include terminal state in T

# Populate R, P, and T matrices
env.reset()
for s in range(n_states):
    for a in range(n_actions):
        env.set_state(s)
        s_next, r, terminated, _, _ = env.step(a)
        R[s, a] = r
        if terminated:
            P[s, a, s] = 1.0  # Transition to itself in case of terminal state
        else:
            P[s, a, s_next] = 1.0
        T[s, a] = terminated  # Mark if the transition is terminal

def q_policy_evaluation(Q, policy, gamma, max_iterations=1000, tol=1e-4):
    for _ in range(max_iterations):
        Q_prev = Q.copy()
        
        for s in range(n_states):
            a = policy[s]
            if T[s, a] == 1:  # Skip terminal states
                continue
            Q[s, a] = np.sum(P[s, a, :] * (R[s, a] + gamma * np.max(Q, axis=1)))
        
        error = np.max(np.abs(Q - Q_prev))
        if error < tol:
            break
    
    return Q

def q_policy_improvement(Q, gamma):
    policy = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        if T[s, :] == 1:  # Skip terminal states
            continue
        policy[s] = np.argmax(np.sum(P[s, :, :] * (R[s, :, np.newaxis] + gamma * np.max(Q, axis=1)), axis=1))
    return policy

def q_value_iteration(gamma, init_value, max_iterations=1000):
    Q = np.full((n_states, n_actions), init_value)
    all_bellman_errors = []

    for _ in range(max_iterations):
        Q_prev = Q.copy()
        for s in range(n_states):
            if T[s, :].all():  # Skip terminal states
                continue
            for a in range(n_actions):
                Q[s, a] = np.sum(P[s, a, :] * (R[s, a] + gamma * np.max(Q, axis=1)))
        
        error = np.max(np.abs(Q - Q_prev))
        all_bellman_errors.append(error)
        if error < 1e-7:
            break

    policy = q_policy_improvement(Q, gamma)
    return policy, len(all_bellman_errors), all_bellman_errors

def q_policy_iteration(gamma, init_value, max_policy_evaluations=1000):
    Q = np.full((n_states, n_actions), init_value)
    policy = np.random.randint(0, n_actions, size=n_states)
    total_policy_evaluations = 0
    all_bellman_errors = []

    while True:
        Q = q_policy_evaluation(Q, policy, gamma)
        total_policy_evaluations += 1
        
        new_policy = q_policy_improvement(Q, gamma)
        if np.all(policy == new_policy):
            break
        policy = new_policy
        
        # Check if the max_policy_evaluations limit is reached
        if total_policy_evaluations >= max_policy_evaluations:
            break

    return policy, total_policy_evaluations, all_bellman_errors

def q_generalized_policy_iteration(gamma, init_value, max_iterations=1000, eval_steps=5):
    Q = np.full((n_states, n_actions), init_value)
    policy = np.random.randint(0, n_actions, size=n_states)
    total_policy_evaluations = 0
    all_bellman_errors = []

    for _ in range(max_iterations):
        for _ in range(eval_steps):
            Q_prev = q_policy_evaluation(Q, policy, gamma)
            all_bellman_errors.extend(np.max(np.abs(Q - Q_prev), axis=1))
            total_policy_evaluations += 1
            if np.max(np.abs(Q - Q_prev)) < 1e-7:
                break
            Q = Q_prev

        new_policy = q_policy_improvement(Q, gamma)
        if np.all(policy == new_policy):
            break
        policy = new_policy

    return policy, total_policy_evaluations, all_bellman_errors

# Initialize plot
fig, axs = plt.subplots(3, 7, figsize=(20, 12))
tot_iter_table = np.zeros((3, 7))
bellman_error_trends = {'VI': [], 'PI': [], 'GPI': []}
algorithms = {'VI': q_value_iteration, 'PI': q_policy_iteration, 'GPI': q_generalized_policy_iteration}

# Run algorithms
for i, init_value in enumerate([-100, -10, -5, 0, 5, 10, 100]):
    for algo_name, algo_func in algorithms.items():
        if algo_name == 'PI':
            policy, tot_iter, bellman_errors = algo_func(gamma=0.99, init_value=init_value, max_policy_evaluations=1000)
        elif algo_name == 'GPI':
            policy, tot_iter, bellman_errors = algo_func(gamma=0.99, init_value=init_value, max_iterations=1000, eval_steps=5)
        else:
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
