import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# 1. PUBLICATION AESTHETICS CONFIGURATION
# =====================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'figure.dpi': 300,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8
})

# =====================================================================
# 2. DYNAMICAL SYSTEM CORE FUNCTIONS
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
    J = np.zeros((6, 6))
    J[0,0] = alpha[0]; J[0,1] = beta[0]; J[0,4] = gamma
    J[1,1] = alpha[1]; J[1,2] = beta[1]; J[1,5] = gamma
    J[2,2] = alpha[2]; J[2,3] = beta[2]; J[2,4] = gamma
    J[3,3] = alpha[3]; J[3,0] = beta[3]; J[3,5] = gamma
    
    twopi = 2 * np.pi
    J[4,0] = delta[0] * twopi * np.cos(twopi*X[0]) * np.sin(twopi*X[2])
    J[4,2] = delta[0] * twopi * np.sin(twopi*X[0]) * np.cos(twopi*X[2])
    J[4,4] = mu[0]
    
    J[5,1] = delta[1] * twopi * np.cos(twopi*X[1]) * np.sin(twopi*X[3])
    J[5,3] = delta[1] * twopi * np.sin(twopi*X[1]) * np.cos(twopi*X[3])
    J[5,5] = mu[1]
    return J

def run_qr_le(alpha, beta, mu, delta, gamma, steps=400):
    X = np.random.rand(6)
    Q = np.eye(6)
    LE_acc = np.zeros(6)
    for _ in range(200): X = map_6d_ocm(X, alpha, beta, mu, delta, gamma)
    for _ in range(steps):
        X = map_6d_ocm(X, alpha, beta, mu, delta, gamma)
        J = jacobian_6d_ocm(X, alpha, beta, mu, delta, gamma)
        B = np.dot(J, Q)
        Q, R = np.linalg.qr(B)
        LE_acc += np.log(np.abs(np.diagonal(R)))
    return LE_acc / steps

# Baseline secure constants
base_alpha = np.array([4.5, 4.5, 4.5, 4.5])
base_beta  = np.array([1.0, 1.0, 1.0, 1.0])
base_mu    = np.array([0.01, 0.01])
base_delta = np.array([0.2, 0.2])
base_gamma = 0.03

N_res = 20  # Fast resolution grid (increase to 50+ for higher publication sharpness)

# =====================================================================
# FIGURE 1: SWEEPING ALPHA_2 VS ALPHA_3
# =====================================================================
print("[*] Generating Figure 1: Alpha Parameter Space Sweep...")
a2_range = np.linspace(4.1, 4.9, N_res)
a3_range = np.linspace(4.1, 4.9, N_res)
A2, A3 = np.meshgrid(a2_range, a3_range)

LE1_mat1, LE2_mat1 = np.zeros((N_res, N_res)), np.zeros((N_res, N_res))
fixed_pairs_alpha = [(4.1, 4.1), (4.5, 4.5)]

fig1, axes1 = plt.subplots(2, 2, figsize=(10, 8))
for p_idx, (a0, a1) in enumerate(fixed_pairs_alpha):
    for i in range(N_res):
        for j in range(N_res):
            alpha_test = np.array([a0, a1, A2[i,j], A3[i,j]])
            spectrum = run_qr_le(alpha_test, base_beta, base_mu, base_delta, base_gamma)
            LE1_mat1[i,j], LE2_mat1[i,j] = spectrum[0], spectrum[1]
            
    # Plot LE1 using corrected pcolormesh method
    im1 = axes1[p_idx, 0].pcolormesh(A2, A3, LE1_mat1, cmap='magma', shading='auto', vmin=0.1)
    axes1[p_idx, 0].set_title(f"$LE_1$ (Max) at $\\alpha_0={a0}, \\alpha_1={a1}$")
    fig1.colorbar(im1, ax=axes1[p_idx, 0], label="LE Value")
    
    # Plot LE2 using corrected pcolormesh method
    im2 = axes1[p_idx, 1].pcolormesh(A2, A3, LE2_mat1, cmap='inferno', shading='auto', vmin=0.01)
    axes1[p_idx, 1].set_title(f"$LE_2$ at $\\alpha_0={a0}, \\alpha_1={a1}$")
    fig1.colorbar(im2, ax=axes1[p_idx, 1], label="LE Value")

for ax in axes1.flatten():
    ax.set_xlabel(r"Parameter $\alpha_2$"); ax.set_ylabel(r"Parameter $\alpha_3$")
plt.tight_layout()
plt.savefig("LE_Sweep_AlphaSpace.png", dpi=300)

# =====================================================================
# FIGURE 2: SWEEPING BETA_2 VS BETA_3
# =====================================================================
print("[*] Generating Figure 2: Beta Parameter Space Sweep...")
b2_range = np.linspace(0.5, 2.0, N_res)
b3_range = np.linspace(0.5, 2.0, N_res)
B2, B3 = np.meshgrid(b2_range, b3_range)

LE1_mat2, LE2_mat2 = np.zeros((N_res, N_res)), np.zeros((N_res, N_res))
fixed_pairs_beta = [(0.5, 0.5), (1.0, 1.0)]

fig2, axes2 = plt.subplots(2, 2, figsize=(10, 8))
for p_idx, (b0, b1) in enumerate(fixed_pairs_beta):
    for i in range(N_res):
        for j in range(N_res):
            beta_test = np.array([b0, b1, B2[i,j], B3[i,j]])
            spectrum = run_qr_le(base_alpha, beta_test, base_mu, base_delta, base_gamma)
            LE1_mat2[i,j], LE2_mat2[i,j] = spectrum[0], spectrum[1]
            
    im1 = axes2[p_idx, 0].pcolormesh(B2, B3, LE1_mat2, cmap='magma', shading='auto', vmin=0.1)
    axes2[p_idx, 0].set_title(f"$LE_1$ (Max) at $\\beta_0={b0}, \\beta_1={b1}$")
    fig2.colorbar(im1, ax=axes2[p_idx, 0], label="LE Value")
    
    im2 = axes2[p_idx, 1].pcolormesh(B2, B3, LE2_mat2, cmap='inferno', shading='auto', vmin=0.01)
    axes2[p_idx, 1].set_title(f"$LE_2$ at $\\beta_0={b0}, \\beta_1={b1}$")
    fig2.colorbar(im2, ax=axes2[p_idx, 1], label="LE Value")

for ax in axes2.flatten():
    ax.set_xlabel(r"Parameter $\beta_2$"); ax.set_ylabel(r"Parameter $\beta_3$")
plt.tight_layout()
plt.savefig("LE_Sweep_BetaSpace.png", dpi=300)

# =====================================================================
# FIGURE 3: SWEEPING DELTA VS GAMMA
# =====================================================================
print("[*] Generating Figure 3: Delta vs Gamma Space Sweep...")
d_range = np.linspace(0.1, 0.3, N_res)
g_range = np.linspace(0.01, 0.05, N_res)
D, G = np.meshgrid(d_range, g_range)

LE1_mat3, LE2_mat3 = np.zeros((N_res, N_res)), np.zeros((N_res, N_res))
fixed_pairs_mu = [(0.005, 0.005), (0.015, 0.015)]

fig3, axes3 = plt.subplots(2, 2, figsize=(10, 8))
for p_idx, (m0, m1) in enumerate(fixed_pairs_mu):
    mu_test = np.array([m0, m1])
    for i in range(N_res):
        for j in range(N_res):
            delta_test = np.array([D[i,j], D[i,j]])
            spectrum = run_qr_le(base_alpha, base_beta, mu_test, delta_test, G[i,j])
            LE1_mat3[i,j], LE2_mat3[i,j] = spectrum[0], spectrum[1]
            
    im1 = axes3[p_idx, 0].pcolormesh(D, G, LE1_mat3, cmap='magma', shading='auto', vmin=0.1)
    axes3[p_idx, 0].set_title(f"$LE_1$ (Max) at $\\mu_0={m0}, \\mu_1={m1}$")
    fig3.colorbar(im1, ax=axes3[p_idx, 0], label="LE Value")
    
    im2 = axes3[p_idx, 1].pcolormesh(D, G, LE2_mat3, cmap='inferno', shading='auto', vmin=0.01)
    axes3[p_idx, 1].set_title(f"$LE_2$ at $\\mu_0={m0}, \\mu_1={m1}$")
    fig3.colorbar(im2, ax=axes3[p_idx, 1], label="LE Value")

for ax in axes3.flatten():
    ax.set_xlabel(r"Parameter $\delta$"); ax.set_ylabel(r"Parameter $\gamma$")
plt.tight_layout()
plt.savefig("LE_Sweep_NonLinearSpace.png", dpi=300)

print("[+] All 12 hyperchaotic verification contours successfully exported.")