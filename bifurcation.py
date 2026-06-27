"""
6D Orthogonal-Cascade Map (6D OCM) - Dark Blue Ultra-Visibility Analysis Suite.
Generates a synchronized two-panel publication graphic with maximum point density and contrast.
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

N_param = 600
alpha_0_sweep = np.linspace(2.0, 5.0, N_param)

N_transient = 1500     
N_bifurcation = 250   
N_lyapunov = 1500      

alpha_mesh = np.full((4, N_param), 4.5); alpha_mesh[0, :] = alpha_0_sweep
beta_mesh  = np.full((4, N_param), 1.0)
mu_mesh    = np.full((2, N_param), 0.01)
delta_mesh = np.full((2, N_param), 0.2)
gamma_mesh = np.full(N_param, 0.03)

X = np.random.rand(6, N_param)
bif_x, bif_y = [], []
LE_spectrum = np.zeros((N_param, 6))
Q = np.repeat(np.eye(6)[np.newaxis, :, :], N_param, axis=0)

print("  [+] Purging transient phase dynamics...")
for _ in range(N_transient):
    X = map_6d_ocm(X, alpha_mesh, beta_mesh, mu_mesh, delta_mesh, gamma_mesh)

print("  [+] Accumulating structural bifurcation arrays...")
for _ in range(N_bifurcation):
    X = map_6d_ocm(X, alpha_mesh, beta_mesh, mu_mesh, delta_mesh, gamma_mesh)
    bif_x.extend(alpha_0_sweep)
    bif_y.extend(X[0]) 

print("  [+] Executing QR Decomposition for Hyperchaos Spectrum...")
for _ in range(N_lyapunov):
    X = map_6d_ocm(X, alpha_mesh, beta_mesh, mu_mesh, delta_mesh, gamma_mesh)
    J = jacobian_6d_ocm(X, alpha_mesh, beta_mesh, mu_mesh, delta_mesh, gamma_mesh)
    B = np.matmul(J, Q)
    Q, R = np.linalg.qr(B)
    LE_spectrum += np.log(np.abs(np.diagonal(R, axis1=1, axis2=2)))

LE_spectrum /= N_lyapunov

# =====================================================================
# 4. RENDER SYNCHRONIZED DARK-BLUE HIGH-CONTRAST FIGURES
# =====================================================================
print("[*] Packaging visualization panels...")

fig = plt.figure(figsize=(11, 9))
gs = gridspec.GridSpec(2, 1, hspace=0.25)

# --- PANEL A: Dark Blue High-Contrast Bifurcation Diagram ---
ax1 = fig.add_subplot(gs[0])
# Configured with s=0.8, solid midnight blue (#000080), and alpha=1.0
ax1.scatter(bif_x, bif_y, s=0.8, c='#000080', alpha=1.0, edgecolors='none', marker='o')
ax1.axvline(4.1, color='#238b45', linestyle='--', linewidth=1.5)

ax1.set_title(r"(a) Bifurcation Tracking Analysis ($x_1$ vs $\alpha_0$)")
ax1.set_ylabel(r"State Variable $x_1$")
ax1.set_xlim([2.0, 5.0])
ax1.set_ylim([0, 1])
ax1.set_xticklabels([]) 

# --- PANEL B: Hyperchaos Lyapunov Spectrum ---
ax2 = fig.add_subplot(gs[1])
colors = ['#d73027', '#fc8d59', '#fee090', '#91bfdb', '#4575b4', '#313695']
labels = [r'$LE_1$ (Max)', r'$LE_2$', r'$LE_3$', r'$LE_4$', r'$LE_5$', r'$LE_6$']

for dim in range(6):
    ax2.plot(alpha_0_sweep, LE_spectrum[:, dim], color=colors[dim], linewidth=1.3, label=labels[dim])

ax2.axhline(0, color='black', linestyle='-', linewidth=1.2)
ax2.axvline(4.1, color='#238b45', linestyle='--', linewidth=1.5, label=r'Cryptographic Threshold ($\alpha_0 \geq 4.1$)')

ax2.set_title(r"(b) Multidimensional Lyapunov Exponent (Hyperchaos) Spectrum")
ax2.set_xlabel(r"Control Parameter $\alpha_0$")
ax2.set_ylabel("Lyapunov Exponents")
ax2.set_xlim([2.0, 5.0])
ax2.legend(loc='lower left', ncol=3)

filename = "Fig_Bifurcation_Lyapunov_DarkBlue.png"
plt.tight_layout()
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"[+] Successfully exported layout to: '{filename}'")
plt.show()