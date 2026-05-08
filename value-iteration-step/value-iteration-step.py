def value_iteration_step(values, transitions, rewards, gamma):
    """
    Perform one step of value iteration and return updated values.
    """
    # Write code here
    new_values = []

    # iterate through each state
    for s in range(len(values)):
        action_values = []

        # Iterate through each action available at state s
        for a in range(len(transitions[s])):
            expected_value = 0

            # Compute sum of transition prob * current_values
            for s_next in range(len(values)):
                expected_value += transitions[s][a][s_next] * values[s_next]

            # Bellman update
            q_value = rewards[s][a] + gamma * expected_value
            action_values.append(q_value)

        # Take the maximum Q-value
        new_values.append(float(max(action_values)))

    return new_values
    