import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ============================================================
# 2D Band Structure Solver — Square Lattice
# ============================================================
#
# STRUCTURE OF THIS FILE:
#   1. Physical parameters
#   2. Potential definition
#   3. Hamiltonian builder  H(kx, ky)
#   4. k-path along high-symmetry points  Γ → X → M → Γ
#   5. Band structure calculation
#   6. Plot
#   7. Surface plot
# ============================================================


# ------------------------------------------------------------
# 1. PHYSICAL PARAMETERS
# ------------------------------------------------------------

N   = 20       # grid points per spatial dimension (cell has N×N points)
a   = 1.0      # lattice constant (sets the unit of length)
dx  = a / N    # spatial step inside the unit cell

V0  = 5.0      # potential strength  (same order as t for visible gaps)

hbar = 1.0
m    = 1.0

# Kinetic hopping parameter — appears in every matrix element
# derived from the finite-difference approximation of ∇²
t = hbar**2 / (2 * m * dx**2)

print(f"Grid: {N}×{N} = {N**2} points per cell")
print(f"Hopping t = {t:.2f},  Potential V0 = {V0:.2f},  ratio V0/t = {V0/t:.4f}")


# ------------------------------------------------------------
# 2. PERIODIC POTENTIAL  V(x, y)
# ------------------------------------------------------------
#
# We use a separable cosine potential that respects the
# square-lattice periodicity:
#
#   V(x,y) = V0 * [cos(2π x/a) + cos(2π y/a)]
#
# Each cosine has period a in its respective direction.
# The two terms are independent — this is the simplest
# potential with the full square-lattice symmetry.

xs = np.linspace(0, a, N, endpoint=False)   # x coords in [0, a)
ys = np.linspace(0, a, N, endpoint=False)   # y coords in [0, a)

# Build 2D potential on the N×N grid
# X[i,j] = x-coordinate of grid point (i,j)
# Y[i,j] = y-coordinate of grid point (i,j)
X, Y = np.meshgrid(xs, ys, indexing='ij')   # shape (N, N)

V2D = V0 * (np.cos(2 * np.pi * X / a) +
            np.cos(2 * np.pi * Y / a))      # shape (N, N)

# Flatten to 1D for use in the diagonal of H
# This means point (i,j) → flat index i*N + j
V_flat = V2D.flatten()                      # shape (N²,)


# ------------------------------------------------------------
# 3. HAMILTONIAN BUILDER  H(kx, ky)
# ------------------------------------------------------------
#
# The 2D Schrödinger equation discretized on an N×N grid gives
# a matrix of size (N²) × (N²).
#
# Each grid point (i,j) has four neighbors:
#   right:  (i+1, j)
#   left:   (i-1, j)
#   up:     (i, j+1)
#   down:   (i, j-1)
#
# The finite-difference Laplacian in 2D is:
#
#   ∇²ψ[i,j] ≈ (ψ[i+1,j] + ψ[i-1,j] + ψ[i,j+1] + ψ[i,j-1]
#                - 4ψ[i,j]) / dx²
#
# so the diagonal gets 4t (one 2t from each direction)
# and there are off-diagonal -t entries for each neighbor.
#
# Bloch boundary conditions in 2D:
#   crossing the right edge (i=N-1 → i=0): gain phase e^{+ikx·a}
#   crossing the top  edge  (j=N-1 → j=0): gain phase e^{+iky·a}
#
# These appear in the matrix as corner elements with phase factors.

def build_H(kx, ky):
    """
    Build the N²×N² Hamiltonian matrix for quasi-momenta (kx, ky).

    The matrix is complex Hermitian — np.linalg.eigh can solve it
    efficiently and guarantees real eigenvalues.
    """
    dim = N * N
    H = np.zeros((dim, dim), dtype=complex)

    # Bloch phase factors for boundary crossings
    phase_x = np.exp(1j * kx * a)   # crossing x boundary
    phase_y = np.exp(1j * ky * a)   # crossing y boundary

    for i in range(N):
        for j in range(N):
            p = i * N + j   # flat index of point (i,j)

            # ── Diagonal: kinetic (4t) + potential ──────────────
            H[p, p] = 4 * t + V_flat[p]

            # ── x-direction neighbors ────────────────────────────
            # right neighbor: (i+1, j)
            if i < N - 1:
                q = (i + 1) * N + j
                H[p, q] = -t
                H[q, p] = -t
            else:
                # i = N-1: right neighbor wraps to i=0 → Bloch phase
                q = 0 * N + j
                H[p, q] = -t * phase_x            # ψ at i=0 represents i=N with phase
                H[q, p] = -t * np.conj(phase_x)  # Hermitian conjugate

            # ── y-direction neighbors ────────────────────────────
            # up neighbor: (i, j+1)
            if j < N - 1:
                q = i * N + (j + 1)
                H[p, q] = -t
                H[q, p] = -t
            else:
                # j = N-1: top neighbor wraps to j=0 → Bloch phase
                q = i * N + 0
                H[p, q] = -t * phase_y
                H[q, p] = -t * np.conj(phase_y)

    return H


# ------------------------------------------------------------
# 4. HIGH-SYMMETRY k-PATH:  Γ → X → M → Γ
# ------------------------------------------------------------
#
# For the square lattice, the first Brillouin zone is itself
# a square with corners at ±π/a in both directions.
#
# The high-symmetry points are:
#   Γ = (0,    0   )    — zone center
#   X = (π/a,  0   )    — zone edge midpoint
#   M = (π/a,  π/a )    — zone corner
#
# Plotting along Γ→X→M→Γ captures all the distinct features
# of the band structure.

nk = 60   # points per segment (total path has 3×nk points)

K_G = np.array([0,       0      ])
K_X = np.array([np.pi/a, 0      ])
K_M = np.array([np.pi/a, np.pi/a])

def make_segment(k_start, k_end, n):
    """Return n k-points linearly interpolated from k_start to k_end."""
    return np.array([k_start + s * (k_end - k_start)
                     for s in np.linspace(0, 1, n, endpoint=False)])

seg_GX = make_segment(K_G, K_X, nk)   # Γ → X
seg_XM = make_segment(K_X, K_M, nk)   # X → M
seg_MG = make_segment(K_M, K_G, nk)   # M → Γ

k_path = np.vstack([seg_GX, seg_XM, seg_MG])   # shape (3*nk, 2)
n_path = len(k_path)

# Build a scalar x-axis: cumulative distance along the path
# (so the tick spacing reflects the actual distance in k-space)
def path_distances(k_path):
    dist = [0.0]
    for i in range(1, len(k_path)):
        d = np.linalg.norm(k_path[i] - k_path[i-1])
        dist.append(dist[-1] + d)
    return np.array(dist)

k_dist = path_distances(k_path)

# Positions of the high-symmetry points on the x-axis
tick_positions = [k_dist[0], k_dist[nk], k_dist[2*nk], k_dist[-1]]
tick_labels    = ['Γ', 'X', 'M', 'Γ']


# ------------------------------------------------------------
# 5. BAND STRUCTURE CALCULATION
# ------------------------------------------------------------
#
# For each k-point on the path:
#   1. Build H(kx, ky)  — size N²×N²
#   2. Solve eigenvalue problem → N² energies
#   3. Keep only the lowest num_bands
#
# We use np.linalg.eigh (not eig) because H is Hermitian:
#   - guaranteed real eigenvalues
#   - faster and numerically stable
#   - eigenvalues returned in ascending order

num_bands = 8   # how many bands to plot

print(f"\nCalculating band structure along Γ→X→M→Γ ({n_path} k-points)...")

bands = np.zeros((n_path, num_bands))

for idx, (kx, ky) in enumerate(k_path):
    H = build_H(kx, ky)
    eigenvalues, _ = np.linalg.eigh(H)
    bands[idx] = eigenvalues[:num_bands]

    if (idx + 1) % 30 == 0:
        print(f"  {idx+1}/{n_path} done")

print("Band structure done.")


# ------------------------------------------------------------
# 6. PLOT
# ------------------------------------------------------------

fig = plt.figure(figsize=(12, 5))
gs  = GridSpec(1, 3, width_ratios=[2.5, 0.05, 1], wspace=0.05)

ax_bands = fig.add_subplot(gs[0])

# Color cycle for bands
colors = plt.cm.viridis(np.linspace(0.15, 0.85, num_bands))

# ── Band structure ───────────────────────────────────────────
for n in range(num_bands):
    ax_bands.plot(k_dist, bands[:, n], color=colors[n], lw=1.6)

# High-symmetry vertical lines
for pos in tick_positions:
    ax_bands.axvline(pos, color='gray', lw=0.6, ls='--', alpha=0.6)

ax_bands.set_xticks(tick_positions)
ax_bands.set_xticklabels(tick_labels, fontsize=13)
ax_bands.set_xlim(k_dist[0], k_dist[-1])
ax_bands.set_xlabel("k-path", fontsize=11)
ax_bands.set_ylabel("Energy  E (units of ℏ²/2ma²)", fontsize=11)
ax_bands.set_title(f"2D Square Lattice — Band Structure\n"
                   f"N={N}×{N}, V₀={V0}, t={t:.1f}", fontsize=11)
ax_bands.grid(alpha=0.2)


plt.savefig("band_structure_2d.png", dpi=150, bbox_inches='tight')
print("\nSaved: band_structure_2d.png")
plt.show()

 
# ------------------------------------------------------------
# 7.  SURFACE PLOT - FIRST BAND
# ------------------------------------------------------------
#
# The band-structure plot (section 8) only shows E along a 1D
# path through the BZ. Here we compute E_1(kx, ky) on a full
# 2D uniform mesh and store it as a surface for plotting.
#
 
nk_surf = 40   # k-points per direction for the surface
 
kx_surf = np.linspace(-np.pi/a, np.pi/a, nk_surf)
ky_surf = np.linspace(-np.pi/a, np.pi/a, nk_surf)
 
KX_surf, KY_surf = np.meshgrid(kx_surf, ky_surf)   # shape (nk_surf, nk_surf)
E_surf = np.zeros((nk_surf, nk_surf))               # will hold E_1(kx, ky)
 
print(f"\nComputing energy surface on {nk_surf}×{nk_surf} k-mesh...")
 
for i, kx in enumerate(kx_surf):
    for j, ky in enumerate(ky_surf):
        H  = build_H(kx, ky)
        ev = np.linalg.eigh(H)[0]   # sorted eigenvalues
        E_surf[i, j] = ev[0]        # only the lowest (first band)
 
print("Surface done.")
 
# High-symmetry points to mark on the surface
# These are the corners and edge-midpoints of the BZ square
surf_sym_points = {
    'Γ': (0,          0         ),
    'X': (np.pi/a,    0         ),
    'M': (np.pi/a,    np.pi/a   ),
}


fig2 = plt.figure(figsize=(9, 7))
ax3d = fig2.add_subplot(111, projection='3d')
 
surf = ax3d.plot_surface(
    KX_surf / (np.pi/a),    # normalize axes to units of π/a
    KY_surf / (np.pi/a),
    E_surf,
    cmap='viridis',
    alpha=0.92,
    linewidth=0,
    antialiased=True
)
 
# colorbar shows the energy scale
cbar = fig2.colorbar(surf, ax=ax3d, shrink=0.5, pad=0.1)
cbar.set_label("Energy  E  (units of ℏ²/2ma²)", fontsize=10)
 
# mark high-symmetry points on the surface
# we project them onto the surface by finding the nearest
# grid point and reading its energy
for label, (kx_sym, ky_sym) in surf_sym_points.items():
 
    # find the grid indices closest to this k-point
    ix = np.argmin(np.abs(kx_surf - kx_sym))
    iy = np.argmin(np.abs(ky_surf - ky_sym))
    E_sym = E_surf[ix, iy]
 
    # plot a dot on the surface and a label above it
    ax3d.scatter(
        kx_sym / (np.pi/a),
        ky_sym / (np.pi/a),
        E_sym,
        color='red', s=40, zorder=5
    )
    ax3d.text(
        kx_sym / (np.pi/a),
        ky_sym / (np.pi/a),
        E_sym + 0.5,            # offset upward so text doesn't overlap dot
        label,
        fontsize=12, color='red', ha='center'
    )
 
# axis labels — normalized to π/a
ax3d.set_xlabel("kₓ  (units of π/a)", fontsize=10, labelpad=8)
ax3d.set_ylabel("kᵧ  (units of π/a)", fontsize=10, labelpad=8)
ax3d.set_zlabel("Energy  E", fontsize=10, labelpad=8)
 
ax3d.set_xticks([-1, 0, 1])
ax3d.set_yticks([-1, 0, 1])
ax3d.set_xticklabels(['-π/a', '0', 'π/a'])
ax3d.set_yticklabels(['-π/a', '0', 'π/a'])
 
ax3d.set_title(
    f"First Band Energy Surface  E₁(kₓ, kᵧ)\n"
    f"N={N}×{N},  V₀={V0},  V₀/t = {V0/t:.4f}",
    fontsize=11
)
 
# viewing angle — adjust these to rotate the surface interactively
ax3d.view_init(elev=28, azim=-55)
 
fig2.savefig("band_surface_2d.png", dpi=150, bbox_inches='tight')
print("Saved: band_surface_2d.png")
 
plt.show()