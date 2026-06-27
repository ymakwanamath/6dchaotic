"""
6D Orthogonal-Cascade Map (6D OCM) - Cryptographic Phase Space Suite.
Generates a high-resolution 2x3 grid displaying all 6 unique phase portraits
for the keystream variables x_1, x_2, x_3, and x_4.
"""

import numpy as np
import matplotlib.pyplot as plt
import itertools

# =====================================================================
# 1. PUBLICATION AESTHETICS CONFIGURATION
# =====================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.2,
    'grid.linestyle': '--'
})

# =====================================================================
# 2. DYNAMICAL SYSTEM CORE EQUATIONS
# =====================================================================
def map_6d_ocm(X, alpha, beta, mu, delta, gamma):
    """Computes the next state of the 6D OCM."""
    X_next = np.zeros_like(X)
    X_next[0] = (alpha[0]*X[0] + beta[0]*X[1] + gamma*X[4]) % 1.0
    X_next[1] = (alpha[1]*X[1] + beta[1]*X[2] + gamma*X[5]) % 1.0
    X_next[2] = (alpha[2]*X[2] + beta[2]*X[3] + gamma*X[4]) % 1.0
    X_next[3] = (alpha[3]*X[3] + beta[3]*X[0] + gamma*X[5]) % 1.0
    X_next[4] = (mu[0]*X[4] + delta[0]*np.sin(2*np.pi*X[0])*np.sin(2*np.pi*X[2])) % 1.0
    X_next[5] = (mu[1]*X[5] + delta[1]*np.sin(2*np.pi*X[1])*np.sin(2*np.pi*X[3])) % 1.0
    return X_next

# =====================================================================
# 3. TRAJECTORY GENERATION ENGINE
# =====================================================================
print("[*] Simulating hyperchaotic trajectory...")

# Cryptographically secure baseline parameters
alpha_arr = np.array([4.5, 4.5, 4.5, 4.5])
beta_arr  = np.array([1.0, 1.0, 1.0, 1.0])
mu_arr    = np.array([0.01, 0.01])
delta_arr = np.array([0.2, 0.2])
gamma_val = 0.03

N_transient = 1000   # Discard initialization states
N_points = 20000     # Length of the keystream sample array

X = np.random.rand(6, 1)

# Purge transient behaviors
for _ in range(N_transient):
    X = map_6d_ocm(X, alpha_arr, beta_arr, mu_arr, delta_arr, gamma_val)

# Accumulate the active phase orbit data
trajectory = np.zeros((4, N_points))  # Storing only x_1, x_2, x_3, x_4
for i in range(N_points):
    X = map_6d_ocm(X, alpha_arr, beta_arr, mu_arr, delta_arr, gamma_val)
    trajectory[:, i] = X[0:4, 0]

# =====================================================================
# 4. RENDER 6-COMBINATION SUBPLOT GRID
# =====================================================================
print("[*] Generating publication graphics...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# Generate the 6 unique mathematical combinations from the 4 arrays
pairs = list(itertools.combinations(range(4), 2))
var_labels = [r"$x_1$", r"$x_2$", r"$x_3$", r"$x_4$"]

for idx, (dim_x, dim_y) in enumerate(pairs):
    ax = axes[idx]
    
    # Render with ultra-fine scatter points and transparency to reveal tracking density
    ax.scatter(trajectory[dim_x], trajectory[dim_y], s=0.3, c='#313695', alpha=0.4, marker='.')
    
    # Labelling and formatting
    ax.set_title(f"({chr(97+idx)}) Projections: {var_labels[dim_x]} vs {var_labels[dim_y]}")
    ax.set_xlabel(var_labels[dim_x])
    ax.set_ylabel(var_labels[dim_y])
    
    # Establish strict normalized boundary limits
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

plt.tight_layout()
filename = "6D_OCM_Cryptographic_Phase_Portraits.png"
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"[+] Successfully exported: '{filename}'")
plt.show()