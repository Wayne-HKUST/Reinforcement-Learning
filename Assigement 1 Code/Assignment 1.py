import numpy as np
import matplotlib.pyplot as plt

# =====================
# Parameter Settings
# =====================
T = 10 # Time horizon
a = 0.5 # Risk aversion coefficient
risky_returns = {'high': 0.15, 'low': -0.05, 'prob_high': 0.6}
riskless_rate = 0.03
gamma = 0.95 # Discount factor

# Discretization
wealth_min = -200
wealth_max = 400
num_wealth_bins = 500 # Discretize Wealth Values
wealth_grid = np.linspace(wealth_min, wealth_max, num_wealth_bins)
num_actions = 100 # Action discretization resolution

# Q-Learning Parameters
alpha_initial = 0.3
alpha_decay = 0.9999
epsilon_initial = 0.5
epsilon_decay = 0.9999
num_episodes = 100000 # Training iterations

# Initialize Q-table
Q = np.zeros((T, num_wealth_bins, num_actions))

# Track parameters
epsilon = epsilon_initial
alpha = alpha_initial

# =====================
# TD Learning Loop
# =====================
for episode in range(num_episodes):
    if (episode + 1) % 10000 == 0:
        print(f"Training progress: {episode + 1}/{num_episodes}")

    W_current = np.random.uniform(wealth_min, wealth_max)
    i_W = np.clip(np.searchsorted(wealth_grid, W_current), 0, num_wealth_bins - 1)

    for t in range(T):

        # if t >= T-1: break

        # Generate dynamic action space
        x_min = -W_current if W_current > 0 else wealth_min
        x_max = W_current if W_current > 0 else wealth_max
        x_vals = np.linspace(x_min, x_max, num_actions)

        # Action selection
        if np.random.rand() < epsilon:
            x = np.random.choice(x_vals)
        else:
            x_idx = np.argmax(Q[t, i_W, :])
            x = x_vals[x_idx]

        # State transition
        if np.random.rand() < risky_returns['prob_high']:
            W_next = x * (risky_returns['high'] - riskless_rate) + W_current * (1 + riskless_rate)
        else:
            W_next = x * (risky_returns['low'] - riskless_rate) + W_current * (1 + riskless_rate)
        W_next = np.clip(W_next, wealth_min, wealth_max)
        i_next = np.clip(np.searchsorted(wealth_grid, W_next), 0, num_wealth_bins - 1)

        # ========== fix ==========

        if t + 1 == T:  # t=T-1
            reward = -np.exp(-a * W_next) / a
        else:
            reward = 0

        if t < T - 1:
            target = reward + gamma * np.max(Q[t + 1, i_next, :])
        else:
            target = reward

        # TD Update
        x_idx = np.clip(np.searchsorted(x_vals, x), 0, num_actions - 1)
        Q[t, i_W, x_idx] += alpha * (target - Q[t, i_W, x_idx])

        # Transition
        W_current, i_W = W_next, i_next

    # Annealing
    epsilon = max(0.01, epsilon * epsilon_decay)
    alpha = max(0.01, alpha * alpha_decay)

# =====================
# Policy Extraction
# =====================
policy = np.zeros((T, num_wealth_bins))
for t in range(T):
    for i_W in range(num_wealth_bins):
        W = wealth_grid[i_W]
        x_min = -W if W > 0 else wealth_min
        x_max = W if W > 0 else wealth_max
        x_vals = np.linspace(x_min, x_max, num_actions)
        valid_actions = np.arange(num_actions)
        optimal_x_idx = valid_actions[np.argmax(Q[t, i_W, valid_actions])]
        policy[t, i_W] = x_vals[optimal_x_idx]

# =====================
# Enhanced Output
# =====================
np.save('Q_table.npy', Q)
np.save('full_policy.npy', policy)

print("\n=== Enhanced Training Summary ===")
print(f"Final exploration rate: {epsilon:.4f}")
print(f"Final learning rate: {alpha:.4f}")
print(f"Q-value range at t=9: [{np.min(Q[9]):.2f}, {np.max(Q[9]):.2f}]")
print(f"Policy range at t=9: [{np.min(policy[9]):.2f}, {np.max(policy[9]):.2f}]")

sample_points = [-200, -190, -180, -170, -160, -150, -100, 0, 100, 200, 300, 400]
print("\n=== Full Policy Output ===")
for t in range(T):
    print(f"\nTime Step t = {t}:")
    print(f"{'Wealth':<10} | {'Optimal Investment':<15} | Q-value")
    print("-" * 40)
    for w in sample_points:
        idx = np.abs(wealth_grid - w).argmin()
        q_val = Q[t, idx, np.argmax(Q[t, idx, :])]
        print(f"{wealth_grid[idx]:<10.1f} | {policy[t, idx]:<15.2f} | {q_val:.2f}")

# =====================
# Visualization
# =====================
for t in range(T):
    plt.figure(figsize=(14, 6))
    # Optimal Investment at time t = 0,...,T - 1
    plt.subplot(1, 2, 1)
    plt.plot(wealth_grid, policy[t], 'b-', linewidth=2)
    plt.xlabel(f'Wealth $W_{t}$', fontsize=12)
    plt.ylabel(f'Optimal Investment $x_{t}^*$', fontsize=12)
    plt.title(f'TD Policy at t={t}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xlim(wealth_min, wealth_max)

    # Optimal Q-values at time t = 0,...,T - 1
    plt.subplot(1, 2, 2)
    Q_t = np.max(Q[t], axis=1)
    plt.plot(wealth_grid, Q_t, 'r-', lw=2)
    plt.xlabel(f'Terminal Wealth $W_{t}$', fontsize=12)
    plt.ylabel('Optimal Q-value', fontsize=12)
    plt.title(f'Q-values at Terminal Time t={t}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xlim(wealth_min, wealth_max)

    plt.tight_layout()
    plt.show()

