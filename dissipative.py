"""
6D Orthogonal-Cascade Map (6D OCM) - Dissipativity & Kaplan-Yorke Dimension Suite.
Generates a synchronized two-panel publication graphic:
Panel A: State-Space Volume Contraction (Dissipativity) over Parameter Space
Panel B: Kaplan-Yorke Fractal Dimension (D_KY) over Parameter Space
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
# 2. VECTORIZED SYSTEM CORE DEFINITIONS
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

def jacobian_6d_ocm(X, alpha, beta, mu, delta, gamma):
    N = X.shape[1]
    J = np.zeros((N, 6, 6))
    
    J[:, 0, 0] = alpha[0]; J[:, 0, 1] = beta[0]; J[:, 0, 4] = gamma
    J[:, 1, 1] = alpha[1]; J[:, 1, 2] = beta[1]; J[:, 1, 5] = gamma
    J[:, 2, 2] = alpha[2]; J[:, 2, 3] = beta[2]; J[:, 2, 4] = gamma
    J[:, 3, 3] = alpha[3]; J[:, 3, 0] = beta[3]; J[:, 3, 5] = gamma
    
    twopi = 2 * np.pi
    J[:, 4, 0] = delta[0] * twopi * np.cos(twopi * X[0]) * np.sin(twopi * X[2])
    J[:, 4, 2] = delta[0] * twopi * np.sin(twopi * X[0]) * np.cos(twopi * X[2])
    J[:, 4, 4] = mu[0]
    
    J[:, 5, 1] = delta[1] * twopi * np.cos(twopi * X[1]) * np.sin(twopi * X[3])
    J[:, 5, 3] = delta[1] * twopi * np.sin(twopi * X[1]) * np.cos(twopi * X[3])
    J[:, 5, 5] = mu[1]
    return J

# =====================================================================
# 3. NUMERICAL ANALYSIS ENGINE
# =====================================================================
print("[*] Initializing numerical simulation grid...")

N_param = 400
alpha_0_sweep = np.linspace(2.0, 5.0, N_param)

N_transient = 1000     
N_lyapunov = 1200      

alpha_mesh = np.full((4, N_param), 4.5); alpha_mesh[0, :] = alpha_0_sweep
beta_mesh  = np.full((4, N_param), 1.0)
mu_mesh    = np.full((2, N_param), 0.01)
delta_mesh = np.full((2, N_param), 0.2)
gamma_mesh = np.full(N_param, 0.03)

X = np.random.rand(6, N_param)
LE_spectrum = np.zeros((N_param, 6))
Q = np.repeat(np.eye(6)[np.newaxis, :, :], N_param, axis=0)

print("  [+] Purging transient phase dynamics...")
for _ in range(N_transient):
    X = map_6d_ocm(X, alpha_mesh, beta_mesh, mu_mesh, delta_mesh, gamma_mesh)

print("  [+] Running QR Decomposition for Lyapunov Spectrum...")
for _ in range(N_lyapunov):
    X = map_6d_ocm(X, alpha_mesh, beta_mesh, mu_mesh, delta_mesh, gamma_mesh)
    J = jacobian_6d_ocm(X, alpha_mesh, beta_mesh, mu_mesh, delta_mesh, gamma_mesh)
    B = np.matmul(J, Q)
    Q, R = np.linalg.qr(B)
    LE_spectrum += np.log(np.abs(np.diagonal(R, axis1=1, axis2=2)))

LE_spectrum /= N_lyapunov

# Calculate Dissipativity (Sum of all Lyapunov Exponents)
le_sum = np.sum(LE_spectrum, axis=1)

# Calculate Kaplan-Yorke Dimension (D_KY)
d_ky = np.zeros(N_param)
for p in range(N_param):
    spectrum = sorted(LE_spectrum[p, :], reverse=True)
    
    # Find the largest index k where the partial sum remains positive
    partial_sum = 0
    k = -1
    for i in range(6):
        if partial_sum + spectrum[i] >= 0:
            partial_sum += spectrum[i]
            k = i
        else:
            break
            
    if k == -1:
        d_ky[p] = 0.0
    elif k == 5: # All sums remain positive (non-dissipative fallback)
        d_ky[p] = 6.0
    else:
        d_ky[p] = (k + 1) + (partial_sum / np.abs(spectrum[k + 1]))

# =====================================================================
# 4. RENDER SYNCHRONIZED ARCHITECTURAL PROOFS
# =====================================================================
print("[*] Packaging visualization panels...")

fig = plt.figure(figsize=(11, 9))
gs = gridspec.GridSpec(2, 1, hspace=0.25)

# --- PANEL A: Dissipativity Plot ---
ax1 = fig.add_subplot(gs[0])
ax1.plot(alpha_0_sweep, le_sum, color='#000080', linewidth=1.5, label=r'$\sum_{i=1}^{6} LE_i$')
ax1.axhline(0, color='black', linestyle='-', linewidth=1.0)
ax1.axvline(4.1, color='#238b45', linestyle='--', linewidth=1.5)

ax1.set_title(r"(a) State-Space Dissipativity Verification (Volume Contraction Rate)")
ax1.set_ylabel(r"Sum of Lyapunov Spectrum ($\sum LE$)")
ax1.set_xlim([2.0, 5.0])
ax1.legend(loc='lower left')
ax1.set_xticklabels([]) 

# --- PANEL B: Kaplan-Yorke Fractal Dimension ---
ax2 = fig.add_subplot(gs[1])
ax2.plot(alpha_0_sweep, d_ky, color='#d73027', linewidth=1.5, label=r'Fractal Dimension $D_{KY}$')
ax2.axhline(6.0, color='black', linestyle=':', linewidth=1.0, label='Embedding Limit ($N=6$)')
ax2.axvline(4.1, color='#238b45', linestyle='--', linewidth=1.5, label=r'Cryptographic Range Threshold ($\alpha_0 \geq 4.1$)')

ax2.set_title(r"(b) Kaplan-Yorke (Lyapunov) Fractal Dimension Landscape")
ax2.set_xlabel(r"Control Parameter $\alpha_0$")
ax2.set_ylabel(r"Information Dimension $D_{KY}$")
ax2.set_xlim([2.0, 5.0])
ax2.set_ylim([0, 6.5])
ax2.legend(loc='lower left')

filename = "Fig_6D_OCM_Dissipativity_DKY.png"
plt.tight_layout()
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"[+] Dissipativity and Kaplan-Yorke figures exported to: '{filename}'")
plt.show()