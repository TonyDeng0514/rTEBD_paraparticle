# rTEBD_paraparticle

Time-evolution by block decimation (TEBD) for a paraparticle chain, represented as a Matrix Product Density Operator (MPDO) in the operator-vectorization (Liouville) picture. Results are validated against exact diagonalization (ED).

---

## Project Structure

`paraparticles/` is a Python package. All scripts and notebooks must be run from the **repo root** (`rTEBD/`) so that `import paraparticles` resolves correctly.

```
rTEBD/
├── paraparticles/          # core simulation package
│   ├── __init__.py
│   ├── MPDO.py
│   ├── build_MPDO_from_mps.py
│   ├── Hamiltonian.py
│   ├── GellMann.py
│   ├── Umat.py
│   ├── ED.py
│   └── rTEBD_para.ipynb
└── bond_convergence/       # bond-dimension convergence study
    ├── bond_convergence.py
    ├── 01_bond_convergence.ipynb
    └── results/
```

---

## File List

| File | Purpose |
|---|---|
| `paraparticles/MPDO.py` | Main simulation class: stores the MPDO tensor network, runs TEBD sweeps, and measures observables |
| `paraparticles/build_MPDO_from_mps.py` | Builds the initial product-state MPDO dict from a classical configuration vector |
| `paraparticles/Hamiltonian.py` | Defines local operators, builds the two-site bond Hamiltonians and their time-evolution gates |
| `paraparticles/GellMann.py` | Defines the standard SU(3) 9-element operator basis {I, λ₁,…,λ₈} and the `gellmann_tilde`/`gellmann_bar` dual pair |
| `paraparticles/Umat.py` | Converts a 9×9 two-site unitary into its rank-4 superoperator tensor `U_all[i,j,k,l]` |
| `paraparticles/ED.py` | Exact diagonalization: assembles the full 3^L × 3^L Hamiltonian from the MPO for benchmark comparisons |
| `paraparticles/rTEBD_para.ipynb` | Driver notebook: runs TEBD, then runs ED on the same disorder realization and plots per-site n_j(t) and total particle number for both methods |
| `bond_convergence/bond_convergence.py` | Runs TEBD at a fixed bond dimension χ and saves per-site n_j(t), trace, and parameters to `results/` as a `.npz` file |
| `bond_convergence/01_bond_convergence.ipynb` | Analysis notebook for bond-dimension convergence study |

---

## MPDO Data Structure

**No separate ket/bra legs.** The density matrix is stored in the operator-vectorization (Liouville) picture, not as an MPO with two physical legs. The ket ⊗ bra d² = 9-dimensional space is collapsed into a single index over a 9-element operator basis.

**Local Hilbert space:** d = 3, basis {|vac⟩, |a⟩, |b⟩} at matrix indices 0, 1, 2.

**Operator basis** (defined in `GellMann.py`):

| Index j | Operator `gellmann_tilde(g)[j]` | Notes |
|---|---|---|
| 0 | I₃ | identity / trace channel |
| 1 | g·λ₁ | off-diagonal vac–a |
| 2 | g·λ₂ | off-diagonal vac–a (imaginary) |
| 3 | g·λ₃ = g·diag(1,−1,0) | standard SU(3) generator (traceless) |
| 4 | g·λ₄ | off-diagonal vac–b |
| 5 | g·λ₅ | off-diagonal vac–b (imaginary) |
| 6 | g·λ₆ | off-diagonal a–b |
| 7 | g·λ₇ | off-diagonal a–b (imaginary) |
| 8 | g·λ₈ = g·(1/√3)·diag(1,1,−2) | standard SU(3) generator (traceless) |

Physical number operators expressed in this basis (from `Hamiltonian.py`):

```
n_a   = I₃/3 − λ₃/2 + λ₈/(2√3)   = diag(0,1,0)
n_b   = I₃/3 − λ₈/√3              = diag(0,0,1)
n_loc = n_a + n_b                   = diag(0,1,1)
```

**Dual basis:** `gellmann_bar(g)[j]` = {I₃/3, λ₁/(2g), λ₂/(2g), …, λ₈/(2g)}.

Orthonormality: `Tr[λ̄_j · λ̃_k] = δ_{jk}` for all j, k. Verified by `check_orthonormality()` in `GellMann.py`.

**Each site tensor** `A_dict["Ai"]` has shape `(χ_left, 9, χ_right)`:

- Axis 0: left bond (dimension χ_left)
- Axis 1: operator-basis index j ∈ {0,…,8}
- Axis 2: right bond (dimension χ_right)

The component at index j is `A[j] = Tr(λ̄_j · ρ_local)`. The density matrix is reconstructed as ρ = Σ_{j₀,…,j_{L−1}} C(j₀,…) · λ̃_{j₀} ⊗ … ⊗ λ̃_{j_{L−1}}, where C is the MPS scalar given by the matrix product of the A tensors at those indices.

**Initial state** (see `build_MPDO_from_mps.py`): product state, χ = 1, shape (1,9,1). Local density matrices and their non-zero Gell-Mann coefficients (g=1):

| Config value | Physical state | ρ | Non-zero A components |
|---|---|---|---|
| 0 (vacuum) | \|vac⟩ | diag(1,0,0) | A[0]=1/3, A[3]=1/2, A[8]=1/(2√3) |
| 1 (flavor a) | \|a⟩ | diag(0,1,0) | A[0]=1/3, A[3]=−1/2, A[8]=1/(2√3) |
| 2 (flavor b) | \|b⟩ | diag(0,0,1) | A[0]=1/3, A[3]=0, A[8]=−1/√3 |

---

## TEBD Update

**Gate construction** (`Umat.py`):

```
U_all[i,j,k,l] = Tr( (λ̄_i ⊗ λ̄_j) · U₉ₓ₉ · (λ̃_k ⊗ λ̃_l) · U†₉ₓ₉ )
```

Indices: (i, j) = output operator basis on sites (left, right); (k, l) = input operator basis on sites (left, right). Shape (9,9,9,9). With U = I₉ this correctly reduces to δ_{ik}·δ_{jl}, verified by `check_Umat()` in `Umat.py`.

**`applyU(ind, dirc, U)`** step by step:

1. **Canonicalize**: call `lmbd_relocate` to move the Schmidt-value locus to `ind[0]` (for `'right'`) or `ind[1]` (for `'left'`). Done by applying identity gates via SVD.

2. **Fetch tensors**: `A1 = A_dict["A{ind[0]}"]` shape (χ₁, 9, χ_mid); `A2 = A_dict["A{ind[1]}"]` shape (χ_mid, 9, χ₂).

3. **Two-site contraction**:
   ```python
   s1 = einsum('ijkl, akb, blc -> aijc', U, A1, A2)
   ```
   - `k` (U's input-left) contracts with A1's physical index
   - `l` (U's input-right) contracts with A2's physical index
   - `b` = shared bond between A1 and A2
   - Result `s1` has shape (χ₁, 9, 9, χ₂): axes (left-bond, new-phys-left, new-phys-right, right-bond)

4. **Reshape for SVD**: `s2 = reshape(s1, (9·χ₁, 9·χ₂))` — C-order merges (χ₁, 9) → rows and (9, χ₂) → columns, so the SVD cut falls between `(left-bond, phys-left)` and `(phys-right, right-bond)`.

5. **SVD**: `Lp, λ, R = svd(s2, full_matrices=False)`. Falls back to `lapack_driver='gesvd'` on convergence failure (logged to `py_print.txt`).

6. **Truncate**: keep χ' = min(χ, 9χ₁, 9χ₂) singular values; `λ → diag(λ[:χ'])`, `Lp → Lp[:, :χ']`, `R → R[:χ', :]`.

7. **Write back**:
   - `'right'` (λ absorbed right): `A1 = reshape(Lp, (χ₁, 9, χ'))`, `A2 = reshape(λ·R, (χ', 9, χ₂))`, `lmbd_position = ind[1]`
   - `'left'` (λ absorbed left): `A1 = reshape(Lp·λ, (χ₁, 9, χ'))`, `A2 = reshape(R, (χ', 9, χ₂))`, `lmbd_position = ind[0]`

**Sweep order** (`sweepU`): odd bonds (0,1), (2,3), … then even bonds (1,2), (3,4), … all with `dirc='right'`. Both odd and even bond gates are built with the full step `τ` (`Hamiltonian.py`), giving a simple first-order Trotter decomposition exp(−iH_odd·τ) · exp(−iH_even·τ) per sweep.

---

## Particle Number Measurement

Called via `measure_TEBD` at the end of each `sweepU` and once at the end of `__init__` (so t=0 measurements are populated before the first sweep).

**Environment construction:**

`build_left` builds `left_trace[i]` as the iterated partial trace from the left up to (but not including) site i:

```python
left_trace[i] = 1 ⊗ (3·A_0[:,0,:]) ⊗ (3·A_1[:,0,:]) ⊗ … ⊗ (3·A_{i-1}[:,0,:])
```

Index 0 selects the identity/trace channel of each site tensor. The factor of 3 per site is `Tr[I₃] = 3`, the physical contribution from tracing out the identity channel. `build_right` builds `right_trace[i]` analogously from the right.

**Single-site measurement** `tensordot_n(ind)`:

```python
n_coeffs[j] = Tr[ n_loc · gellmann_tilde(g)[j] ]   (precomputed at init)

result = Σ_j  n_coeffs[j] · (left_trace[ind] @ A_ind[:,j,:] @ right_trace[ind])
```

For g=1 the non-zero coefficients are: `n_coeffs[0]=2`, `n_coeffs[3]=−1`, `n_coeffs[8]=−1/√3`. All other Gell-Mann matrices have zero overlap with `n_loc` and are skipped.

**Trace** (`tr_TEBD`): contracts `3·A_i[:,0,:]` across all sites — this is the physical trace `Tr[ρ]`, which equals 1 for a normalized state.

---

## Suspected Fragile Points

Items marked **FIXED** were resolved. Items marked **OPEN** remain outstanding.

---

**1. ~~`l_3` was not the standard Gell-Mann λ₃ and was not traceless.~~ — FIXED**
`l_3` was `diag(0,1,1)` (total number operator, `Tr = 2`). It is now `diag(1,−1,0)` (standard SU(3) λ₃, traceless). `l_8` was `diag(0,1,−1)` and is now `(1/√3)·diag(1,1,−2)`. All eight generators are now traceless, which is required for the trace and measurement formulas to work correctly.

---

**2. ~~Operator basis completeness was not verified.~~ — FIXED**
`gellmann_bar` previously used `I₃` (no 1/3 factor) and `(1/g)·λⱼ` for j≥1, giving `Tr[λ̄_j·λ̃_k] ≠ δ_{jk}`. It now uses `I₃/3` and `λⱼ/(2g)`, satisfying `Tr[λ̄_j·λ̃_k] = δ_{jk}` for all j,k. `check_orthonormality()` was added to `GellMann.py` and passes.

---

**3. ~~`build_left` / `build_right` were missing `Tr[I₃] = 3` per environment site.~~ — FIXED**
Each environment contraction now uses `3 · A_k[:,0,:]`. The factor of 3 = `Tr[I₃]` is the physical partial trace contribution from tracing out the identity channel of each environment site. The `tr_TEBD` loop applies the same factor. Measurements return integer values at t=0 and total particle number is conserved under TEBD, consistent with ED.

---

**4. `U_mat` index convention is implicit and not documented. — OPEN**
`U_all[i,j,k,l]` with `(i,j)` = output and `(k,l)` = input is assumed throughout `applyU`'s einsum `'ijkl,akb,blc->aijc'`. If `U_mat` were ever changed to swap input/output ordering, the evolution would silently apply the conjugate superoperator. There is no assertion or comment protecting this.

---

**5. Reshape index order in the SVD step is load-bearing. — OPEN**
`reshape(s1, (9·χ₁, 9·χ₂))` where s1 is `(χ₁, 9, 9, χ₂)` relies on C-order to place `(χ₁, 9)` as rows and `(9, χ₂)` as columns. The inverse reshapes of `Lp` and `R` must use the same order. If the axes of s1 were ever permuted before the reshape, the SVD would cut along the wrong axis, mixing bond and physical indices.

---

**6. ~~Initial measurement not populated before the first sweep.~~ — FIXED**
`__init__` calls `self.measure_TEBD()` after constructing the initial MPDO, so `ni_persite` and `tr_TEBD` are populated at t=0 before any sweep runs.

---

**7. ~~`GellMann.py` returned `np.matrix` objects, not `np.ndarray`.~~ — FIXED**
`l_1` through `l_8` previously used `np.matrix(…)`. They now use `np.array(…)` throughout, eliminating the risk of silent matrix-multiply semantics and the deprecation warning.
