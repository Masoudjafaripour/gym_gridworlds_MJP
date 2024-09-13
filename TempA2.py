import gymnasium
import numpy as np
import matplotlib.pyplot as plt

env = gymnasium.make("Gym-Gridworlds/Penalty-3x3-v0")

n_states = env.observation_space.n
n_actions = env.action_space.n

# Initialize reward, transition probability matrices
R = np.zeros((n_states, n_actions))
P = np.zeros((n_states, n_actions, n_states))
T = np.zeros((n_states, n_actions))

env.reset()
for s in range(n_states):
    for a in range(n_actions):
        env.set_state(s)
        s_next, r, terminated, _, _ = env.step(a)
        R[s, a] = r
        P[s, a, s_next] = 1.0
        T[s, a] = terminated
        print(terminated)

# Bellman update for Vπ(s)
def bellman_v(V, gamma):
    V_new = np.zeros(n_states)
    for s in range(n_states):
        V_new[s] = sum([P[s, a, s_next] * (R[s, a] + gamma * V[s_next]) 
                        for a in range(n_actions) 
                        for s_next in range(n_states)])
    return V_new

# Bellman update for Qπ(s, a)
def bellman_q(Q, gamma):
    Q_new = np.zeros((n_states, n_actions))
    for s in range(n_states):
        for a in range(n_actions):
            Q_new[s, a] = sum([P[s, a, s_next] * (R[s, a] + gamma * max(Q[s_next])) 
                               for s_next in range(n_states)])
    return Q_new

gammas = [0.01, 0.5, 0.99]
initial_values = [-10, 0, 10]

for init_value in initial_values:
    # Initialize V(s) and Q(s, a)
    V = np.full(n_states, init_value)
    Q = np.full((n_states, n_actions), init_value)
    
    fig, axs = plt.subplots(2, len(gammas))
    fig.suptitle(f"$V_0$: {init_value}")
    
    for i, gamma in enumerate(gammas):
        # Iterate Bellman updates
        for _ in range(100):  # You can tweak the number of iterations
            V = bellman_v(V, gamma)
        
        axs[0][i].imshow(V.reshape(3, 3), cmap='coolwarm')  # Heatmap of V
        axs[1][i].plot(V)  # Plot the value
        
        axs[0][i].set_title(f'$\gamma$ = {gamma}')
    
    # Q function visualization
    fig, axs = plt.subplots(n_actions + 1, len(gammas))
    fig.suptitle(f"$Q_0$: {init_value}")
    
    for i, gamma in enumerate(gammas):
        for _ in range(100):
            Q = bellman_q(Q, gamma)
        
        for a in range(n_actions):
            axs[a][i].imshow(Q[:, a].reshape(3, 3), cmap='coolwarm')  # Heatmap of Q
        axs[-1][i].plot(Q.flatten())  # Plot Q
    
        axs[0][i].set_title(f'$\gamma$ = {gamma}')
    
    plt.show()
