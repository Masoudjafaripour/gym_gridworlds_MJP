import numpy as np

def select_action(Q, state):
    """
    Selects the action A* that maximizes Q(state, action) with ties broken arbitrarily.

    Parameters:
    Q -- Q-value table (2D array or matrix) where Q[s, a] represents the Q-value for state s and action a.
    state -- the current state for which we want to select the action.

    Returns:
    A* -- the selected action.
    """
    
    # Get the Q-values for the current state
    q_values = Q[state]
    # print(q_values)
    # Find the maximum Q-value
    max_q_value = np.max(q_values)
    # print(max_q_value)
    
    # Find all actions that have the maximum Q-value (i.e., handle ties)
    best_actions = np.where(q_values == max_q_value)[0]
    # print(best_actions)
    
    # Break ties arbitrarily by randomly selecting one of the best actions
    A_star = np.random.choice(best_actions)
    
    return A_star

# Example Q-table and state
Q = np.array([[1, 2, 2, 3],
              [0, 5, 5, 4],
              [3, 1, 3, 2]])

state = 1  # For example, we are in state 1
action = select_action(Q, state)
print("Selected Action:", action)
