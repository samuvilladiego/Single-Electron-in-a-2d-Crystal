import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1D Band Structure Solver — Uniform Lattice
# ============================================================
#
# STRUCTURE OF THIS FILE:
#   1. Physical parameters
#   2. Potential definition
#   3. Hamiltonian builder  H(k)
#   4. k-space sampling across the first Brillouin zone
#   5. Band structure calculation
#   6. Plot
#
# PHYSICAL SETUP:
#   A single electron moves along an infinite 1D chain of ions
#   separated by lattice constant a. Each ion creates an
#   attractive potential, giving a net periodic potential V(x)
#   with period a.
#
#   We solve the time-independent Schrödinger equation:
#
#       [ -ℏ²/2m · d²/dx² + V(x) ] ψ(x) = E ψ(x)
#
#   By the Bloch theorem, ψ satisfies:
#
#       ψ(x + a) = e^{ika} · ψ(x)
#
#   so it is enough to solve on one unit cell [0, a)
#   with this condition imposed at the boundary.
#
# UNITS:
#   ℏ = m = a = 1  (dimensionless). Energies are in units of
#   ℏ²/(2m·a²). To convert to eV, multiply by ℏ²/(2m·a²)
#   evaluated with physical constants.
# ============================================================


# ------------------------------------------------------------
# 1. PHYSICAL PARAMETERS
# ------------------------------------------------------------

N   = 100      # number of grid points inside one unit cell [0, a)
               # more points → smaller dx → more accurate derivatives
               # but also larger matrix → slower computation
               # N = 100 gives a good accuracy/speed tradeoff in 1D

a   = 1.0      # lattice constant — sets the unit of length
               # all lengths are measured in units of a

dx  = a / N    # spatial step: distance between adjacent grid points
               # dx = 0.01 with N=100, a=1

V0  = 5.0      # strength of the periodic potential (same units as energy)
               

hbar = 1.0     # reduced Planck constant (set to 1 by choice of units)
m    = 1.0     # electron mass (set to 1 by choice of units)

# Kinetic hopping parameter t
# ─────────────────────────────
# When we approximate d²ψ/dx² with finite differences:
#
#   d²ψ/dx²|ᵢ ≈ (ψᵢ₊₁ - 2ψᵢ + ψᵢ₋₁) / dx²
#
# the prefactor ℏ²/(2m·dx²) appears in every kinetic term.
# We call this t to keep the matrix entries clean.

t = hbar**2 / (2 * m * dx**2)

print(f"Grid:      {N} points in [0, a)")
print(f"Step:      dx = {dx:.4f}")
print(f"Hopping:   t  = {t:.2f}")
print(f"Potential: V0 = {V0:.2f}")
print(f"Ratio:     V0/t = {V0/t:.5f}  (controls gap size)")


# ------------------------------------------------------------
# 2. PERIODIC POTENTIAL  V(x)
# ------------------------------------------------------------
#
# We use a cosine potential — the simplest function that:
#   (a) has period a: V(x + a) = V(x)  ✓
#   (b) is smooth (no sharp edges)
#   (c) has a single minimum per cell (at x = 0, near each ion)
#
#   V(x) = V0 · cos(2π x / a)


x = np.linspace(0, a, N, endpoint=False)   # N points in [0, a)
                                            # endpoint=False: x=a is NOT
                                            # included because it equals
                                            # x=0 of the next cell under
                                            # Bloch boundary conditions

V = V0 * np.cos(2 * np.pi * x / a)        # shape (N,)

print(f"\nPotential: min = {V.min():.2f},  max = {V.max():.2f}")


# ------------------------------------------------------------
# 3. HAMILTONIAN BUILDER  H(k)
# ------------------------------------------------------------
#
# We build an N×N complex Hermitian matrix for each k value.
#
# Structure of H(k):
# ──────────────────
#
#   H = kinetic part + potential part + Bloch boundary terms
#
# DIAGONAL  H[i,i] = 2t + V[i]
#   • 2t comes from the finite-difference approximation:
#     the -2ψᵢ term in (ψᵢ₊₁ - 2ψᵢ + ψᵢ₋₁)/dx² gives +2t
#     on the diagonal after multiplying by -ℏ²/2m
#   • V[i] is the local potential at grid point i
#
# OFF-DIAGONAL  H[i, i±1] = -t
#   • couples each point to its left and right neighbors
#   • this is the kinetic energy of "hopping" one grid step
#   • the matrix is tridiagonal in the interior
#
# BLOCH CORNERS  H[0, N-1] and H[N-1, 0]
#   • connect the right edge (i=N-1) to the left edge (i=0)
#   • the Bloch condition requires: ψ(x+a) = e^{ika} ψ(x)
#     so the "ghost" neighbor of i=N-1 to the right is i=0
#     but with a phase factor e^{+ika}
#   • H[N-1, 0] = -t · e^{+ika}    (right edge looks left)
#   • H[0, N-1] = -t · e^{-ika}    (Hermitian conjugate)
#   • this is the ONLY place where k enters the matrix
#   • different k values → different phases → different eigenvalues
#
# Why Hermitian?
#   The Hamiltonian is a physical observable (energy), so its
#   eigenvalues must be real numbers. Hermitian matrices always
#   have real eigenvalues.

def build_H(k):
    """
    Build the N×N Hamiltonian matrix for quasi-momentum k.

    Parameters
    ----------
    k : float
        Quasi-momentum in the first Brillouin zone [-π/a, π/a].

    Returns
    -------
    H : ndarray, shape (N, N), dtype complex
        Hermitian Hamiltonian matrix with Bloch boundary conditions.
    """
    H = np.zeros((N, N), dtype=complex)

    # ── Bloch phase factor ───────────────────────────────────
    # e^{ika}: phase gained when crossing the right boundary
    phase = np.exp(1j * k * a)

    # ── Interior: tridiagonal kinetic + diagonal potential ───
    for i in range(N):

        # Diagonal element: kinetic self-energy + local potential
        # The 2t comes from the -2ψᵢ term in the finite difference
        H[i, i] = 2 * t + V[i]

        # Left neighbor: point i-1 couples to point i
        # Only exists for interior points (i > 0)
        if i > 0:
            H[i, i - 1] = -t    # row i, column i-1
            H[i - 1, i] = -t    # Hermitian conjugate (same value here,
                                 # since -t is real)

    # ── Bloch boundary: connect right edge to left edge ─────
    #
    # Point i = N-1 (rightmost) needs its right neighbor.
    # Its right neighbor is point i = 0 (leftmost of next cell),
    # but shifted by one lattice constant a, so it carries phase e^{+ika}.
    #
    # H[N-1, 0] = -t · e^{+ika}   (right edge → left edge, forward)
    # H[0, N-1] = -t · e^{-ika}   (left edge → right edge, backward)
    #                               = conj(H[N-1, 0])  ← Hermitian

    H[N - 1, 0    ] = -t * phase           # -t · e^{+ika}
    H[0,     N - 1] = -t * np.conj(phase)  # -t · e^{-ika}

    return H


# ------------------------------------------------------------
# 4. k-SPACE SAMPLING — FIRST BRILLOUIN ZONE
# ------------------------------------------------------------
#
# The first Brillouin zone (BZ) in 1D is the interval [-π/a, π/a].

num_k  = 300
k_vals = np.linspace(-np.pi / a, np.pi / a, num_k)

# x-axis label positions for high-symmetry points
# In 1D the only high-symmetry points are:
#   -π/a  (left zone boundary)
#    0    (zone center, called Γ)
#   +π/a  (right zone boundary)

print(f"\nk-space: {num_k} points from -π/a to +π/a")


# ------------------------------------------------------------
# 5. BAND STRUCTURE CALCULATION
# ------------------------------------------------------------
#
# Algorithm:
#   for each k in k_vals:
#       1. build H(k)              — N×N Hermitian matrix
#       2. solve H ψ = E ψ         — gives N eigenvalues
#       3. keep lowest num_bands   — lowest energy states
#
# Memory layout:
#   bands[i, n] = energy of band n at k = k_vals[i]
#   shape: (num_k, num_bands)

num_bands = 6   # number of bands to compute and plot

print(f"Computing {num_bands} bands at {num_k} k-points...")

bands = np.zeros((num_k, num_bands))

for idx, k in enumerate(k_vals):

    H = build_H(k)

    # eigh returns eigenvalues in ascending order — index 0 is lowest
    eigenvalues, _ = np.linalg.eigh(H)

    # store only the lowest num_bands energies for this k
    bands[idx] = eigenvalues[:num_bands]

print("Done.")

print(f"\nSanity check — zone boundary symmetry:")
print(f"  Band 0 at k=-π/a: {bands[0,  0]:.6f}")
print(f"  Band 0 at k=+π/a: {bands[-1, 0]:.6f}  (should match)")


# ------------------------------------------------------------
# 6. PLOT
# ------------------------------------------------------------
#
# We plot E_n(k) vs k for the first num_bands bands.
#
# Reading the plot:
#   • Each colored curve is one band E_n(k)
#   • Gaps between curves are forbidden energies (band gaps)
#   • Flat regions → slow electrons (low group velocity dE/dk)
#   • Steep regions → fast electrons (high group velocity)
#   • Band gaps are largest at the zone boundary k = ±π/a
#     where Bragg reflection is strongest
#   • At k = 0 (Γ point) bands can touch or nearly touch
#
# The x-axis is normalized to π/a so the ticks read -1, 0, 1
# instead of the raw values -3.14..., 0, 3.14...
# This is the standard convention in solid-state physics.

fig, ax = plt.subplots(figsize=(7, 5))

# color each band distinctly
colors = plt.cm.viridis(np.linspace(0.1, 0.9, num_bands))

for n in range(num_bands):
    ax.plot(
        k_vals / (np.pi / a),   # normalize x-axis to units of π/a
        bands[:, n],
        color=colors[n],
        lw=1.8,
        label=f"n = {n + 1}"
    )

# mark the high-symmetry points with vertical dashed lines
for xpos, label in [(-1, ''), (0, 'Γ'), (1, '')]:
    ax.axvline(xpos, color='gray', lw=0.6, ls='--', alpha=0.5)

# horizontal line at E = 0 for reference
# (energies below zero mean the electron is attracted to the ion)
ax.axhline(0, color='gray', lw=0.4, ls=':', alpha=0.4)

# axis labels and ticks
ax.set_xticks([-1, 0, 1])
ax.set_xticklabels([r'$-\pi/a$', r'$\Gamma$', r'$\pi/a$'], fontsize=12)
ax.set_xlabel("k  (units of π/a)", fontsize=11)
ax.set_ylabel("Energy  E  (units of ℏ²/2ma²)", fontsize=11)
ax.set_title(
    f"1D Band Structure — Uniform Lattice\n"
    f"N = {N},  V₀ = {V0},  t = {t:.1f},  V₀/t = {V0/t:.4f}",
    fontsize=11
)

ax.legend(loc='upper center', ncol=num_bands, fontsize=9,
          framealpha=0.7, handlelength=1.2)
ax.grid(alpha=0.2)
ax.set_xlim(-1, 1)

plt.tight_layout()
plt.savefig("band_structure_1d.png", dpi=150, bbox_inches='tight')
print("\nSaved: band_structure_1d.png")
plt.show()