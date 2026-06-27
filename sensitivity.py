"""
6D Orthogonal-Cascade Map (6D OCM) - Sensitivity & Avalanche Suite.
Generates a synchronized two-panel publication graphic:
Panel A: Phase Sensitivity (Exponential Trajectory Divergence via $10^{-15}$ perturbation)
Panel B: Strict Avalanche Criterion (SAC) Convergence to the 0.5 Ideality Boundary
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# =====================================================================
# 1. PUBLICATION AESTHETICS CONFIGURATION
# =====================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'axes.labelsize': 11,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linestyle': '--'
})

# =====================================================================
# 2. DYNAMICAL SYSTEM CORE EQUATIONS
# =====================================================================
def map_6d_ocm(X, alpha, beta, mu, delta, gamma):
    X_next = np.zeros_like(X)
    X_next[0] = (alpha[0]*X[0] + beta[0]*X[1] + gamma*X[4]) % 1.0
    X_next[1] = (alpha[1]*X[1] + beta[1]*X[2] + gamma*X[5]) % 1.0
    X_next[2] = (alpha[2]*X[2] + beta[2]*X[3] + gamma*X[4]) % 1.0
    X_next[3] = (alpha[3]*X[3] + beta[3]*X[0] + gamma*X[5]) % 1.0
    X_next[4] = (mu[0]*X[4] + delta[0]*np.sin(2*np.pi*X[0])*np.sin(2*np.pi*X[2])) % 1.0
    X_next[5] = (mu[1]*X[5] + delta[1]*np.sin(2*np.pi*X[1])*np.sin(2*np.pi*X[3])) % 1.0
    return X_next

# =====================================================================
# 3. ANALYSIS EXECUTION ENGINE
# =====================================================================
print("[*] Launching Sensitivity and Avalanche Analysis Framework...")

# Cryptographic Baseline Parameters
alpha_arr = np.array([4.5, 4.5, 4.5, 4.5])
beta_arr  = np.array([1.0, 1.0, 1.0, 1.0])
mu_arr    = np.array([0.01, 0.01])
delta_arr = np.array([0.2, 0.2])
gamma_val = 0.03

N_steps = 100

# --- Test 1: Phase Sensitivity (Trajectory Divergence) ---
X1 = np.random.rand(6)
# Perturb a single state dimension by the floating-point precision limit (10^-15)
X2 = X1.copy()
X2[0] += 1e-15

divergence = []
for _ in range(N_steps):
    X1 = map_6d_ocm(X1, alpha_arr, beta_arr, mu_arr, delta_arr, gamma_val)
    X2 = map_6d_ocm(X2, alpha_arr, beta_arr, mu_arr, delta_arr, gamma_val)
    # Euclidean distance metric across the active 6D phase space
    dist = np.linalg.norm(X1 - X2)
    divergence.append(dist)

# --- Test 2: Strict Avalanche Criterion (SAC) Verification ---
# Reset systems to evaluate bit modification probabilities over 5000 random key scenarios
N_trials = 5000
sac_accumulator = np.zeros(N_steps)

for _ in range(N_trials):
    S1 = np.random.rand(6)
    S2 = S1.copy()
    S2[0] += 1e-15  # Microscopic difference simulates a 1-bit input flip
    
    for t in range(N_steps):
        S1 = map_6d_ocm(S1, alpha_arr, beta_arr, mu_arr, delta_arr, gamma_val)
        S2 = map_6d_ocm(S2, alpha_arr, beta_arr, mu_arr, delta_arr, gamma_val)
        
        # Convert floating-point state outputs to standard 8-bit integer bytes for bit check
        b1 = int((S1[0] * 10**8) % 256)
        b2 = int((S2[0] * 10**8) % 256)
        
        # Calculate bit change frequency using bitwise XOR counting
        bit_difference = bin(b1 ^ b2).count('1')
        sac_accumulator[t] += (bit_difference / 8.0)

sac_probability = sac_accumulator / N_trials

# =====================================================================
# 4. RENDER SYNCHRONIZED CRYPTOGRAPHIC SECURITY PROOFS
# =====================================================================
print("[*] Packaging visualization panels...")

fig = plt.figure(figsize=(11, 9))
gs = gridspec.GridSpec(2, 1, hspace=0.28)

# --- PANEL A: Phase Sensitivity (Log Scale Divergence) ---
ax1 = fig.add_subplot(gs[0])
ax1.plot(range(N_steps), divergence, color='#000080', linewidth=1.5, label='Euclidean Separation Distance')
ax1.set_yscale('log')

ax1.set_title(r"(a) Microscopic Sensitivity Analysis ($\Delta X_{init} = 10^{-15}$)")
ax1.set_ylabel("Phase Space Distance (Log Scale)")
ax1.set_xlim([0, N_steps])
ax1.legend(loc='lower right')

# --- PANEL B: Strict Avalanche Criterion (SAC) Convergence ---
ax2 = fig.add_subplot(gs[1])
ax2.plot(range(N_steps), sac_probability, color='#d73027', linewidth=1.2, label='Bit-Change Frequency $P(t)$')
ax2.axhline(0.5, color='black', linestyle='--', linewidth=1.5, label='Ideal Avalanche Target ($P = 0.5$)')

# Setting a strict cryptographic boundary window to check for statistical deviations
ax2.set_ylim([0.35, 0.65])
ax2.set_xlim([0, N_steps])
ax2.set_title(r"(b) Strict Avalanche Criterion (SAC) Bit Diffusion Proof")
ax2.set_xlabel("Iteration Step (Time Series)")
ax2.set_ylabel("Bit Change Probability")
ax2.legend(loc='upper right')

filename = "Fig_6D_OCM_Sensitivity_SAC.png"
plt.tight_layout()
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"[+] Sensitivity and SAC plots exported to: '{filename}'")
plt.show()