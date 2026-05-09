import numpy as np
from GellMann import (
                        l_1, l_2, l_3, l_4,
                        l_5, l_6, l_7, l_8,
                        I_3
                    )
from Umat import U_mat

# def Hamiltonian
# local spin ladder operators
S_a_plus  = (l_1() - 1j * l_2()) / 2
S_a_minus = (l_1() + 1j * l_2()) / 2
S_b_plus  = (l_4() - 1j * l_5()) / 2
S_b_minus = (l_4() + 1j * l_5()) / 2

# local number and magnetization operators
n_a   = I_3()/3 - l_3()/2 + l_8()/(2*np.sqrt(3))
n_b   = I_3()/3             - l_8()/np.sqrt(3)
n_loc = n_a + n_b
m_loc = n_a - n_b

def draw_dis(L, mean, W, rng = None):
    rng = np.random.default_rng() if rng is None else rng
    return rng.normal(loc=mean, scale=W, size=L)


def draw_q(L, mean, W, rng=None):
    rng=np.random.default_rng() if rng is None else rng
    return rng.normal(loc=mean, scale=W, size=(L, 2))

def build_MPO(L, t, Omega_list, q_list, V_list):
    """
    V_list[j] is the coupling on the bond (j, j+1), so effectively length L-1.
    Omega_list and q_list have length L and shape (L, 2) respectively.
    """
    d = 3
    w = 7

    def make_W(Omega_j, q_a, q_b, V_j):
        W = np.zeros((w, w, d, d), dtype=complex)

        # --- Right column (finishing operator B_{j+1} of each two-site term) ---
        W[0, 0] = I_3()        # identity channel (term already completed to the right)
        W[1, 0] = S_a_plus     # finishes  y-_j y+_{j+1}   (hopping, flavor a)
        W[2, 0] = S_a_minus    # finishes  y+_j y-_{j+1}   (hopping, flavor a)
        W[3, 0] = S_b_plus     # finishes  y-_j y+_{j+1}   (hopping, flavor b)
        W[4, 0] = S_b_minus    # finishes  y+_j y-_{j+1}   (hopping, flavor b)
        W[5, 0] = m_loc        # finishes  m_j m_{j+1}     (V interaction)

        # --- Bottom row (starting operator A_j, carries the coefficient) ---
        W[6, 1] = -t * S_a_minus
        W[6, 2] = -t * S_a_plus
        W[6, 3] = -t * S_b_minus
        W[6, 4] = -t * S_b_plus
        W[6, 5] = V_j * m_loc
        W[6, 6] = I_3()         # identity channel (no term started yet from the left)

        # --- Bottom-right corner: purely on-site terms, born and die at site j ---
        W[6, 0] = (Omega_j * (S_a_plus @ S_b_minus + S_b_plus @ S_a_minus)
                   + q_a * n_a + q_b * n_b)
        return W

    v_L = np.zeros(w, dtype=complex); v_L[6] = 1.0
    v_R = np.zeros(w, dtype=complex); v_R[0] = 1.0

    # V_j lives on bond (j, j+1); site L-1 has no bond to its right, so V is 0 there.
    Ws = [make_W(Omega_list[j], q_list[j, 0], q_list[j, 1],
                 V_list[j] if j < L - 1 else 0.0) for j in range(L)]

    MPO = [None] * L
    MPO[0]  = np.einsum("a,abij->bij", v_L, Ws[0])
    MPO[-1] = np.einsum("abij,b->aij", Ws[-1], v_R)
    for j in range(1, L - 1):
        MPO[j] = Ws[j]

    return MPO

def build_onsite_omega(Omega_j, q_a, q_b):
    """3x3 on-site operator: omega_j = Omega_j (a+ b- + b+ a-) + q_a n_a + q_b n_b."""
    return (Omega_j * (S_a_plus @ S_b_minus + S_b_plus @ S_a_minus)
            + q_a * n_a + q_b * n_b)

def build_bond_hamiltonian_twosite(t, V_i):
    """9x9 bare two-site piece on bond (i, i+1): hopping (both flavors) and V_i m m."""
    h = np.zeros((9, 9), dtype=complex)
    h += -t * (np.kron(S_a_plus, S_a_minus) + np.kron(S_a_minus, S_a_plus))
    h += -t * (np.kron(S_b_plus, S_b_minus) + np.kron(S_b_minus, S_b_plus))
    h +=  V_i * np.kron(m_loc, m_loc)
    return h

def build_bond_hamiltonians_tilde(L, t, Omega_list, q_list, V_list):
    if L < 2:
        raise ValueError("Need at least L=2 to define bond Hamiltonians.")

    omegas = [build_onsite_omega(Omega_list[j], q_list[j, 0], q_list[j, 1])
              for j in range(L)]

    h_tilde_list = []
    for i in range(L - 1):
        # Weights on the two sites of this bond, aware of open-boundary edges.
        w_left  = 1.0 if i == 0         else 0.5
        w_right = 1.0 if (i + 1) == L-1 else 0.5
        h = build_bond_hamiltonian_twosite(t, V_list[i])
        h += w_left  * np.kron(omegas[i],     I_3())
        h += w_right * np.kron(I_3(), omegas[i + 1])
        h_tilde_list.append(h)
    return h_tilde_list

def _expm_hermitian(H, theta):
    # Symmetrize to kill roundoff drift before eigh.
    Hs = 0.5 * (H + H.conj().T)
    w, U = np.linalg.eigh(Hs)
    return (U * np.exp(-1j * theta * w)) @ U.conj().T

def build_bond_gates(L, t, Omega_list, q_list, V_list, tau, g=1):
    h_tilde_list = build_bond_hamiltonians_tilde(L, t, Omega_list, q_list, V_list)

    gates_odd  = []   # bonds (0,1), (2,3), (4,5), ...  ← bond index 2k
    gates_even = []   # bonds (1,2), (3,4), (5,6), ...  ← bond index 2k+1
    for i, h in enumerate(h_tilde_list):
        if i % 2 == 0:
            gates_odd.append(U_mat(_expm_hermitian(h, tau), g))
        else:
            gates_even.append(U_mat(_expm_hermitian(h, tau), g))
    return gates_odd, gates_even
