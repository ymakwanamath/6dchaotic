import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigvals
from mpl_toolkits.mplot3d import Axes3D

# =====================================================================
# 1. PUBLICATION AESTHETICS CONFIGURATION
# =====================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'figure.dpi': 300
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

# =====================================================================
# 3. CONSTRUCT VARIABLE SPACE & FIXED INSTANCES
# =====================================================================
N_mesh = 40
alpha_2_vec = np.linspace(1.5, 6.0, N_mesh)
alpha_3_vec = np.linspace(1.5, 6.0, N_mesh)
A2, A3 = np.meshgrid(alpha_2_vec, alpha_3_vec)

fixed_instances = [
    (2.5, 2.5), (3.5, 3.5), (4.5, 2.5), 
    (2.5, 4.5), (4.5, 4.5), (5.5, 5.5)
]

beta_arr  = np.array([1.0, 1.0, 1.0, 1.0])
mu_arr    = np.array([0.01, 0.01])
delta_arr = np.array([0.2, 0.2])
gamma_val = 0.03
epsilon   = 0.025  # Exclude slow modulating variable eigenvalues

fig = plt.figure(figsize=(18, 12))

# =====================================================================
# 4. MAPPING MESH & HIGHLIGHTING SECURE INSTANCES
# =====================================================================
for idx, (a0, a1) in enumerate(fixed_instances):
    print(f"[*] Analyzing Space for Instance {idx+1}/6: (alpha_0={a0}, alpha_1={a1})...")
    Z_min_nonzero_eigen = np.zeros((N_mesh, N_mesh))
    
    for i in range(N_mesh):
        for j in range(N_mesh):
            a2 = A2[i, j]
            a3 = A3[i, j]
            alpha_arr = np.array([a0, a1, a2, a3])
            
            X = np.random.rand(6)
            for _ in range(400):
                X = map_6d_ocm(X, alpha_arr, beta_arr, mu_arr, delta_arr, gamma_val)
            
            local_mins = []
            for _ in range(20):
                X = map_6d_ocm(X, alpha_arr, beta_arr, mu_arr, delta_arr, gamma_val)
                J = jacobian_6d_ocm(X, alpha_arr, beta_arr, mu_arr, delta_arr, gamma_val)
                magnitudes = np.abs(eigvals(J))
                
                filtered_mags = magnitudes[magnitudes > epsilon]
                if len(filtered_mags) == 0:
                    local_mins.append(np.min(magnitudes))
                else:
                    local_mins.append(np.min(filtered_mags))
                
            Z_min_nonzero_eigen[i, j] = np.mean(local_mins)
            
    # -----------------------------------------------------------------
    # Render 3D Surface
    # -----------------------------------------------------------------
    ax = fig.add_subplot(2, 3, idx + 1, projection='3d')
    surf = ax.plot_surface(A2, A3, Z_min_nonzero_eigen, cmap='coolwarm', edgecolor='none', alpha=0.6)
    
    # Extract coordinates where the stability criteria passes (|lambda| > 1.0)
    secure_mask = Z_min_nonzero_eigen > 1.0
    secure_a2 = A2[secure_mask]
    secure_a3 = A3[secure_mask]
    secure_z  = Z_min_nonzero_eigen[secure_mask]
    
    # Highlight passing regions in the 3D plot with bright green markers
    if len(secure_z) > 0:
        ax.scatter(secure_a2, secure_a3, secure_z, color='#00ff00', s=1.5, depthshade=False, label=r'$|\lambda|_{min} > 1.0$')
        
        # Calculate axis limits for the highlighted lines
        min_a2_pass, max_a2_pass = np.min(secure_a2), np.max(secure_a2)
        min_a3_pass, max_a3_pass = np.min(secure_a3), np.max(secure_a3)
        
        # Highlight passing values directly on the x and y axes lines at the base floor
        z_floor = np.min(Z_min_nonzero_eigen)
        ax.plot([min_a2_pass, max_a2_pass], [1.5, 1.5], [z_floor, z_floor], color='#00cc00', linewidth=4.5, zorder=10)
        ax.plot([1.5, 1.5], [min_a3_pass, max_a3_pass], [z_floor, z_floor], color='#00cc00', linewidth=4.5, zorder=10)

    # Label adjustments
    ax.set_title(f"({chr(97+idx)}) $\\alpha_0 = {a0}$, $\\alpha_1 = {a1}$", fontsize=12, pad=12)
    ax.set_xlabel(r"Parameter $\alpha_2$")
    ax.set_ylabel(r"Parameter $\alpha_3$")
    ax.set_zlabel(r"Min Non-Zero $|\lambda|$", rotation=90)
    
    ax.set_xlim([1.5, 6.0])
    ax.set_ylim([1.5, 6.0])
    
    # Set the viewpoint to clearly see the axes and highlighted bands
    ax.view_init(elev=22, azim=-125)
    
    if idx == 0:
        ax.legend(loc='upper left', markerscale=4)

fig.subplots_adjust(right=0.92, hspace=0.28, wspace=0.18)
cbar_ax = fig.add_axes([0.94, 0.2, 0.015, 0.6])
fig.colorbar(surf, cax=cbar_ax, label=r"Minimum Non-Zero Eigenvalue Magnitude $|\lambda|$")

filename = "6D_OCM_3D_Highlighted_Stability.png"
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"\n[+] Highlighted 3D stability analysis plot saved as: '{filename}'")
plt.show()