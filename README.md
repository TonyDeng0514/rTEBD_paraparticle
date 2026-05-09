# rTEBD_paraparticle

Time-evolution by block decimation (TEBD) for a paraparticle chain, represented as a Matrix Product Density Operator (MPDO) in the operator-vectorization (Liouville) picture.

---

## File List

| File | Purpose |
|---|---|
| `paraparticles/MPDO.py` | Main simulation class: stores the MPDO tensor network, runs TEBD sweeps, and measures observables |
| `paraparticles/build_MPDO_from_mps.py` | Builds the initial product-state MPDO dict from a classical configuration vector |
| `paraparticles/Hamiltonian.py` | Defines local operators, builds the two-site bond Hamiltonians and their time-evolution gates |
| `paraparticles/GellMann.py` | Defines the custom 9-element operator basis {I, l_1,…,l_8} and the `gellmann_tilde`/`gellmann_bar` dual pair |
| `paraparticles/Umat.py` | Converts a 9×9 two-site unitary into its rank-4 superoperator tensor `U_all[i,j,k,l]` |
| `paraparticles/rTEBD_para.ipynb` | Driver notebook: instantiates `MPDO`, runs `sweepU` for `Nt` steps, stores and plots `n_i(t)` |

---

## MPDO Data Structure

**No separate ket/bra legs.** The density matrix is stored in the operator-vectorization (Liouville) picture, not as an MPO with two physical legs. The ket ⊗ bra d² = 9-dimensional space is collapsed into a single index over a 9-element operator basis.

**Local Hilbert space:** d = 3, basis {|vac⟩, |a⟩, |b⟩} at matrix indices 0, 1, 2.

**Operator basis** (defined in `GellMann.py`):

| Index j | Operator `gellmann_tilde(g)[j]` | Notes |
|---|---|---|
| 0 | I₃ | identity / trace channel |
| 1 | g·l₁ | off-diagonal vac–a |
| 2 | g·l₂ | off-diagonal vac–a (imaginary) |
| 3 | g·l₃ = g·diag(0,1,1) | total number n̂_a + n̂_b — **not traceless** |
| 4 | g·l₄ | off-diagonal vac–b |
| 5 | g·l₅ | off-diagonal vac–b (imaginary) |
| 6 | g·l₆ | off-diagonal a–b |
| 7 | g·l₇ | off-diagonal a–b (imaginary) |
| 8 | g·l₈ = g·diag(0,1,−1) | flavor magnetization n̂_a − n̂_b |

Dual basis: `gellmann_bar(g)[j]` = {I₃, (1/g)·l₁, …, (1/g)·l₈}.

**Each site tensor** `A_dict["Ai"]` has shape `(χ_left, 9, χ_right)`:

- Axis 0: left bond (dimension χ_left)
- Axis 1: operator-basis index j ∈ {0,…,8}
- Axis 2: right bond (dimension χ_right)

The component along j at site i is `A[i][j] = Tr(λ̄_j · ρ_i_local)`. The density matrix is notionally reconstructed as ρ = Σ_{j₀,…,j_{L−1}} C(j₀,…) · λ̃_{j₀} ⊗ … ⊗ λ̃_{j_{L−1}}, where C is the MPS scalar given by the matrix product of A tensors at those indices.

**Initial state** (see `build_MPDO_from_mps.py`): product state, χ = 1, shape (1,9,1). Local density matrices:

| Config value | Physical state | Non-zero components |
|---|---|---|
| 0 (vacuum) | ρ = I₃ − l₃ = diag(1,0,0) | A[0]=1, all others 0 |
| 1 (flavor a) | ρ = n_a = diag(0,1,0) | A[0]=1, A[3]=1, A[8]=1 |
| 2 (flavor b) | ρ = n_b = diag(0,0,1) | A[0]=1, A[3]=1, A[8]=−1 |

---

## TEBD Update

**Gate construction** (`Umat.py`):

```
U_all[i,j,k,l] = (1/9) · Tr( (λ̄_i ⊗ λ̄_j) · U₉ₓ₉ · (λ̃_k ⊗ λ̃_l) · U†₉ₓ₉ )
```

Indices: (i, j) = output operator basis on sites (left, right); (k, l) = input operator basis on sites (left, right). Shape (9,9,9,9).

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

**Sweep order** (`sweepU`): odd bonds (0,1), (2,3), … then even bonds (1,2), (3,4), … all with `dirc='right'`. The odd-bond gates are built with `τ/2` and even-bond gates with `τ` (`Hamiltonian.py`), implementing only the first half of Strang splitting — the second odd-bond pass (which would make it second-order) is absent.

---

## Particle Number Measurement

Called via `measure_TEBD` at the end of each `sweepU`.

**Environment construction:**

`build_left` builds `left_trace[i]` = matrix product of `A_0[:,0,:] @ A_1[:,0,:] @ … @ A_{i-1}[:,0,:]`. Index `0` selects the identity/trace channel. Shape of `left_trace[i]`: `(χ, χ)` where χ is the bond dimension at site i.

`build_right` builds `right_trace[i]` similarly from the right: `A_{i+1}[:,0,:] @ … @ A_{L-1}[:,0,:]`.

**Single-site measurement** `tensordot_l3(ind)`:

```python
result = left_trace[ind] · (g · A_ind[:,3,:]) · right_trace[ind]   →  scalar
```

- Index `3` selects the `l_3` component = Tr(λ̄₃ · ρ_ind) = (1/g)·Tr(l₃ · ρ_ind)
- Multiplying by `g` recovers Tr(l₃ · ρ_ind) = ⟨n̂_a + n̂_b⟩_ind
- Left and right environments supply the partial traces over all other sites

**Trace** (`tr_TEBD`): contracts `A_i[:,0,:]` across all sites — this is the coefficient of the all-identity term, not the physical trace (see Fragile Points §3 below).

---

## Normalization Bug Diagnosis (post-refactor)

After the refactor to standard SU(3) Gell-Mann matrices, the simulation gives ~10⁻⁵ particles instead of integer values. The following checks were run analytically to locate the cause.

### Check 1 — GellMann.py: Orthonormality PASSES

Current definitions:
- `l_3 = diag(1, −1, 0)` — standard SU(3) λ₃ (old `diag(0,1,1)` is commented out)
- `l_8 = (1/√3)·diag(1, 1, −2)` — standard SU(3) λ₈ (old `diag(0,1,−1)` is commented out)
- `gellmann_tilde(g)` = `{I₃, g·λ₁, …, g·λ₈}`
- `gellmann_bar(g)` = `{I₃/3, λ₁/(2g), …, λ₈/(2g)}`

`Tr[λ̄_j · λ̃_k] = δ_{jk}` holds for all 81 pairs:
- j=k=0: `Tr[(I₃/3)·I₃] = 1` ✓
- j=k≥1: `Tr[λⱼ/(2g) · g·λⱼ] = (1/2)·Tr[λⱼ²] = (1/2)·2 = 1` ✓
- j≠k: zero by tracelessness or Gell-Mann orthogonality ✓

`check_orthonormality()` passes.

### Check 2 — Umat.py: Identity check PASSES

Current formula (old `1/9` scalar prefactor removed):

```
U_all[i,j,k,l] = Tr( (λ̄_i ⊗ λ̄_j) · U · (λ̃_k ⊗ λ̃_l) · U† )
```

With U = I₉: `U_all[i,j,k,l] = Tr[λ̄_i·λ̃_k] · Tr[λ̄_j·λ̃_l] = δ_{ik}·δ_{jl}` ✓

`check_Umat()` passes.

### Check 3 — build_MPDO_from_mps.py: Reconstruction PASSES

For each state, `A[j] = Tr(gellmann_bar(g)[j] · ρ)` and `ρ_rec = Σ_j A[j] · gellmann_tilde(g)[j]`.

Checked for flavor-a (`ρ = diag(0,1,0)`, g=1):
- `A[0] = 1/3`,  `A[3] = −1/2`,  `A[8] = 1/(2√3)`,  all others 0
- `ρ_rec = (1/3)·I₃ + (−1/2)·diag(1,−1,0) + (1/(2√3))·(1/√3)·diag(1,1,−2)`
- `= diag(1/3−1/2+1/6,  1/3+1/2+1/6,  1/3+0−1/3) = diag(0,1,0)` ✓

Same holds for vacuum and flavor-b by symmetry. Reconstruction is correct.

### Check 4 — MPDO.py: tensordot_n FAILS

**n_coeffs** `n_coeffs[j] = Tr[n_loc · gellmann_tilde(g)[j]]` with g=1:

| j | value |
|---|---|
| 0 | 2 |
| 3 | −1 |
| 8 | −1/√3 |
| all others | 0 |

Single-site dot product for flavor-a: `2·(1/3) + (−1)·(−1/2) + (−1/√3)·(1/(2√3)) = 2/3 + 1/2 − 1/6 = 1.0` ✓

**The failure is in `build_left` / `build_right`.** Because `gellmann_bar[0] = I₃/3`, every normalized site contributes `A_k[0] = Tr[(I₃/3)·ρ_k] = 1/3`. The environment tensors therefore accumulate:

```
left_trace[i]  = (1/3)^i
right_trace[i] = (1/3)^(L−1−i)
```

So `tensordot_n(i) = (1/3)^(L−1) · 1.0`. For L=10: `(1/3)⁹ ≈ 5.08 × 10⁻⁵` — exactly the observed symptom.

### Root cause

The correct physical environment sum per traced-out site is:

```
Σ_{j_k} A_k[j_k] · Tr[λ̃_{j_k}]
```

All Gell-Mann matrices λ₁…λ₈ are traceless, so only j_k=0 survives, contributing `A_k[0] · Tr[I₃] = (1/3) · 3 = 1`. The missing factor per site is `Tr[I₃] = 3`; total missing factor is `3^(L−1)`.

The old code had `gellmann_bar[0] = I₃` (no 1/3), giving `A_k[0] = 1` and environments of 1 — the missing `3^(L−1)` and the wrong normalization cancelled accidentally. When the refactor correctly introduced `I₃/3` to make the dual basis orthonormal, that accidental cancellation broke.

**Fix required:** `build_left` and `build_right` must contract with `3 · A_k[:,0,:]` instead of `A_k[:,0,:]`. The same factor of 3 is missing from the `tr_TEBD` calculation (it currently computes `(1/3)^L` rather than 1).

---

## Suspected Fragile Points

Items marked **FIXED** were resolved in the refactor to standard SU(3) Gell-Mann matrices. Items marked **OPEN** remain outstanding.

---

**1. ~~`l_3` was not the standard Gell-Mann λ₃ and was not traceless.~~ — FIXED**
`l_3` was `diag(0,1,1)` (total number operator, `Tr = 2`). It is now `diag(1,−1,0)` (standard SU(3) λ₃, traceless). `l_8` was `diag(0,1,−1)` and is now `(1/√3)·diag(1,1,−2)`. All eight generators are now traceless, which is required for the trace and measurement formulas to work correctly.

---

**2. ~~Operator basis completeness was not verified.~~ — FIXED**
`gellmann_bar` previously used `I₃` (no 1/3 factor) and `(1/g)·λⱼ` for j≥1, giving `Tr[λ̄_j·λ̃_k] ≠ δ_{jk}`. It now uses `I₃/3` and `λⱼ/(2g)`, satisfying `Tr[λ̄_j·λ̃_k] = δ_{jk}` for all j,k. `check_orthonormality()` was added to `GellMann.py` and passes.

---

**3. `build_left` / `build_right` are missing `Tr[I₃] = 3` per environment site. — OPEN (active bug)**
See the Normalization Bug Diagnosis section for the full derivation. In brief: each environment contraction uses `A_k[:,0,:]` directly, but the correct physical partial trace requires `3 · A_k[:,0,:]` (the factor of 3 = `Tr[I₃]` comes from tracing out the identity channel). Because `gellmann_bar[0] = I₃/3` now correctly gives `A_k[0] = 1/3`, the missing 3 per site produces a net suppression of `(1/3)^(L−1)` in every measurement and `(1/3)^L` in `tr_TEBD`. This is the cause of the ~10⁻⁵ particle numbers.

Fix: change both `build_left` and `build_right` to contract with `3 * self.A_dict["Ak"][:,0,:]` instead of `self.A_dict["Ak"][:,0,:]`. Apply the same factor to the `tr_TEBD` loop.

---

**4. `U_mat` index convention is implicit and not documented. — OPEN**
`U_all[i,j,k,l]` with `(i,j)` = output and `(k,l)` = input is assumed throughout `applyU`'s einsum `'ijkl,akb,blc->aijc'`. If `U_mat` were ever changed to swap input/output ordering, the evolution would silently apply the conjugate superoperator. There is no assertion or comment protecting this.

---

**5. Reshape index order in the SVD step is load-bearing. — OPEN**
`reshape(s1, (9·χ₁, 9·χ₂))` where s1 is `(χ₁, 9, 9, χ₂)` relies on C-order to place `(χ₁, 9)` as rows and `(9, χ₂)` as columns. The inverse reshapes of `Lp` and `R` must use the same order. If the axes of s1 were ever permuted before the reshape, the SVD would cut along the wrong axis, mixing bond and physical indices.

---

**6. Initial measurement is never populated before the first sweep. — OPEN**
`ni_persite` is initialized to zeros in `__init__` and only updated inside `measure_TEBD`, which is called at the end of `sweepU`. The notebook stores `ni[j][0]` before the first sweep, so the t=0 column is always zero regardless of the initial configuration.

---

**7. ~~`GellMann.py` returned `np.matrix` objects, not `np.ndarray`.~~ — FIXED**
`l_1` through `l_8` previously used `np.matrix(…)`. They now use `np.array(…)` throughout, eliminating the risk of silent matrix-multiply semantics and the deprecation warning.

---

**8. Strang splitting is incomplete. — OPEN**
`build_bond_gates` builds odd gates with `τ/2` and even gates with `τ`, implying second-order Strang splitting. But `sweepU` applies only one odd pass (at `τ/2`) followed by one even pass (at `τ`), omitting the closing odd pass (at `τ/2`). Each time step accumulates first-order Trotter error, not second-order.
