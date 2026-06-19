# Single Electron in a 2D Crystalline Structure
# Band Structure Solver — 1D and 2D Periodic Crystals

Numerical simulation of the allowed electron energies in a periodic crystal, implemented via finite-difference discretization of the Schrödinger equation and Bloch boundary conditions. Produces band structure plots in 1D and 2D (square lattice), including a 3D energy surface of the first band.

## See results and figures directly in the report (report.pdf) 

---

## Physics background

An electron in a crystal moves under a periodic ionic potential `V(r + a) = V(r)`, where `a` is the lattice constant. The time-independent Schrödinger equation is:

```
[ -ℏ²/2m · ∇² + V(r) ] ψ(r) = E ψ(r)
```

By the **Bloch theorem**, every solution satisfies:

```
ψ(x + a) = e^{ika} · ψ(x)
```

where `k` is the **quasi-momentum**. This means the problem on the infinite crystal reduces to solving on a single unit cell `[0, a)` with the Bloch phase as a boundary condition. For each value of `k` in the **first Brillouin zone** `[-π/a, π/a]`, there are multiple allowed energies `E₁(k) ≤ E₂(k) ≤ ...` — these curves are the **band structure**.

The periodic potential used is a cosine:

```
V(x) = V₀ · cos(2π x / a)          (1D)
V(x,y) = V₀ · [cos(2πx/a) + cos(2πy/a)]   (2D)
```

The ratio `V₀/t` — where `t = ℏ²/(2m·Δx²)` is the kinetic hopping parameter — controls the competition between localization (potential) and delocalization (kinetic energy), and determines how large the band gaps are.

---

## Method

### Discretization

The unit cell is sampled on a uniform grid of `N` points (1D) or `N×N` points (2D) with spacing `Δx = a/N`. The second derivative is approximated by the **centered finite difference**:

```
d²ψ/dx²|ᵢ ≈ (ψᵢ₊₁ - 2ψᵢ + ψᵢ₋₁) / Δx²
```

In 2D, the full Laplacian uses the **five-point stencil** coupling each grid point `(i,j)` to its four neighbors.

### Hamiltonian matrix

The discretized Schrödinger equation becomes a matrix eigenvalue problem `H(k) ψ = E ψ`. The matrix has three types of entries:

| Entry | Value | Origin |
|---|---|---|
| Diagonal `H[i,i]` | `2t + Vᵢ` (1D) or `4t + Vᵢⱼ` (2D) | kinetic self-energy + local potential |
| Off-diagonal `H[i,i±1]` | `-t` | coupling to nearest neighbor |
| Bloch corners | `-t · e^{±ika}` | Bloch boundary condition at cell edges |

The corners are the **only entries that depend on k**. The rest of the matrix is identical for all k-values, so the interior is built once and the corners are updated at each k.

### Eigenvalue problem

For each k, `numpy.linalg.eigh` solves `H(k) ψ = E ψ`. It returns `N` (or `N²`) real eigenvalues in ascending order. The lowest `num_bands` are the physically relevant band energies at that k.

---

## Repository structure

```
.
├── band_structure.py    # 1D solver
├── 2D_lattice.py    # 2D solver (square lattice) + 3D surface
└── README.md
```

---

## Requirements

- Python 3.8+
- numpy
- matplotlib

Install dependencies:

```bash
pip install numpy matplotlib
```

---

## Usage

### 1D solver

```bash
python band_structure.py
```

Produces `band_structure_1d.png` — band structure along the full Brillouin zone `[-π/a, π/a]`.

### 2D solver

```bash
python 2D_lattice.py
```

Produces two figures:

- `band_structure_2d.png` — band structure along `Γ → X → M → Γ`
- `band_surface_2d.png` — 3D energy surface of the first band over the full BZ

---

## Key parameters

Both scripts use reduced units `ℏ = m = a = 1`. Energies are in units of `ℏ²/(2ma²)`.

### 1D (`band_structure_1d.py`)

| Parameter | Default | Description |
|---|---|---|
| `N` | 100 | Grid points per unit cell. Higher → more accurate but slower. |
| `V0` | 5.0 | Potential strength. Controls band gap size. |
| `num_k` | 300 | k-points sampled across the BZ. Higher → smoother curves. |
| `num_bands` | 6 | Number of bands to compute and plot. |

### 2D (`band_structure_2d.py`)

| Parameter | Default | Description |
|---|---|---|
| `N` | 20 | Grid points per spatial direction (matrix size: N²×N²). |
| `V0` | 5.0 | Potential strength. |
| `nk` | 60 | k-points per path segment (Γ→X, X→M, M→Γ). |
| `num_bands` | 8 | Number of bands to compute. |
| `nk_surf` | 40 | k-points per direction for the 3D energy surface. |

### Effect of V₀/t

The physically meaningful parameter is the dimensionless ratio `V₀/t`:

| Regime | Effect on band structure |
|---|---|
| `V₀/t → 0` | Free electron limit: parabolic bands, no gaps |
| `V₀/t ~ 0.01–0.1` | Nearly-free electron: small gaps at zone boundary |
| `V₀/t ~ 1` | Clear gaps and flat bands visible |
| `V₀/t >> 1` | Very flat bands, large gaps, electron nearly trapped |

Since `t = ℏ²/(2mΔx²) = N²/2` (with `a=1`), increasing `N` increases `t` and therefore decreases `V₀/t`. To study a particular physical regime, adjust `V0` rather than `N`.

---

## Reading the plots

### Band structure plot

- Each colored curve is one band `Eₙ(k)`
- Gaps between curves are **forbidden energies** — no electron can have those energies in this crystal
- Band gaps are widest at the zone boundary (`k = ±π/a` in 1D, `X` and `M` points in 2D) where Bragg reflection is strongest
- Band touching points indicate **degeneracy** enforced by crystal symmetry

### 3D energy surface (2D only)

Shows `E₁(kₓ, kᵧ)` — the energy landscape of the lowest band over the full Brillouin zone:

- **Minimum at Γ = (0,0)** — lowest energy state of the band
- **Saddle points at X = (π/a, 0)** — energy increases in one direction, decreases in the other; produces peaks in the DOS
- **Maximum at M = (π/a, π/a)** — highest energy of the first band, just below the first gap

The curvature of the surface at any point is proportional to the inverse effective mass of the electron: a flat surface means a heavy, hard-to-accelerate electron; a steep surface means a light, mobile one.

### High-symmetry path Γ → X → M → Γ (2D)

The conventional band structure plot is a one-dimensional cut through the 2D Brillouin zone along the path connecting the high-symmetry points:

```
ky
π/a  ·  M ──────────── M
         │            ↗
         │         ↗          M → Γ diagonal
         │      ↗
 0   ·  Γ ──── X
        0      π/a           kx
```

The horizontal axis shows cumulative distance along this path, so longer segments (like M→Γ, which has length √2·π/a) occupy proportionally more space.

---

## Sanity checks

The 1D solver prints a zone-boundary symmetry check at runtime:

```
Sanity check — zone boundary symmetry:
  Band 0 at k=-π/a: 2.285739
  Band 0 at k=+π/a: 2.285739  (should match)
```

The two endpoints of the BZ are physically equivalent, so band energies must be identical there. A mismatch indicates an error in the Bloch boundary conditions.

---

## Extending the code

**Change the potential:** replace the cosine in section 2 of either script with any function satisfying `V(x + a) = V(x)`. A double-well potential or a square-wave potential will produce qualitatively different band structures.

**Extract wavefunctions:** `eigh` returns eigenvectors as the second output (currently discarded with `_`). Save `eigenvectors[:, n]` to get the discretized wavefunction `ψₙ(x)` for band `n` at a given `k`.

**Convert to physical units:** multiply energies by `ℏ²/(2mₑa²)` evaluated with `ℏ = 1.055×10⁻³⁴ J·s`, `mₑ = 9.109×10⁻³¹ kg`, and your lattice constant `a` in meters. This gives energies in Joules; divide by `1.602×10⁻¹⁹` to get eV.

**Extend to graphene:** replace the finite-difference Hamiltonian with a 2×2 tight-binding model on the hexagonal lattice, sweep `k` along `Γ → M → K → Γ`, and observe the linear band touching (Dirac cone) at the `K` points.

