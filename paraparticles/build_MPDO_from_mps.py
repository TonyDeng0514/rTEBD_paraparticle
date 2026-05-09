import numpy as np

from GellMann import I_3, l_3, mat_dot2, gellmann_bar
from Hamiltonian import n_a, n_b

# build random initial state
def random_config(L, N, Na, rng=None):
    """
    Returns a random configuration of length L with exactly N particles,
    Na of which are flavor a (|a>=1) and N-Na are flavor b (|b>=2).
    The remaining L-N sites are |vac> (0).
    """
    if rng is None:
        rng = np.random.default_rng()
    config = np.zeros(L, dtype=int)
    particle_sites = rng.choice(L, size=N, replace=False)
    flavors = np.array([1] * Na + [2] * (N - Na))
    rng.shuffle(flavors)
    config[particle_sites] = flavors
    return config.tolist()


def build_product_state(L, psi0_config, chi_init=1):
    """
    psi0_config[j] in {0, 1, 2}, meaning the local basis index at site j:
      0 -> |vac>, 1 -> |a>, 2 -> |b>.
    """
    d = 3
    if len(psi0_config) != L:
        raise ValueError(f"psi0_config length {len(psi0_config)} != L={L}")
    
    MPS = [None] * L
    for i, occ in enumerate(psi0_config):
        if occ not in (0, 1, 2):
            raise ValueError(f"site {i}: occ must be 0, 1, or 2 (got {occ!r})")
        A = np.zeros((chi_init, d, chi_init), dtype=complex)
        A[0, occ, 0] = 1.0
        MPS[i] = A
    
    vL = np.zeros(chi_init); vL[0] = 1.0
    vR = np.zeros(chi_init); vR[0] = 1.0
    MPS[0]  = np.einsum('a, asb -> sb', vL, MPS[0])
    MPS[-1] = np.einsum('asb, b -> as', MPS[-1], vR)
    return MPS

def initial_MPDO_dict(L, psi_0_config, g=1):
    A_dict = {}
    rho_vac = np.diag([1., 0., 0.])
    rho_a   = np.diag([0., 1., 0.])
    rho_b   = np.diag([0., 0., 1.])
    
    for i in range(L):
        A_temp = np.zeros(9, dtype=np.complex128)
        if psi_0_config[i] == 0:
            rho = rho_vac
        elif psi_0_config[i] == 1:
            rho = rho_a
        elif psi_0_config[i] == 2:
            rho = rho_b
        for j in range(9):
            A_temp[j] = np.trace(mat_dot2(gellmann_bar(g)[j], rho))
        A_dict["A"+str(i)] = np.reshape(A_temp, (1, 9, 1))
    return A_dict

