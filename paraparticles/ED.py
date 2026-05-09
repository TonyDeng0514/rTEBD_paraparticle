import numpy as np
import string

def build_full_H_from_mpo(MPO):
    N = len(MPO)
    if N < 2:
        raise ValueError("Need at least N=2.")

    d = MPO[0].shape[1]

    letters = string.ascii_letters
    if 3 * N - 1 > len(letters):
        raise ValueError("System too large for this simple einsum label builder.")

    bond = list(letters[:N-1])

    ket = list(letters[N-1:2*N-1])

    bra = list(letters[2*N-1:3*N-1])

    subs = []

    subs.append(bond[0] + ket[0] + bra[0])

    for i in range(1, N - 1):
        subs.append(bond[i-1] + bond[i] + ket[i] + bra[i])

    subs.append(bond[-1] + ket[-1] + bra[-1])

    out = ''.join(ket) + ''.join(bra)

    eq = ','.join(subs) + '->' + out
    H_tensor = np.einsum(eq, *MPO)
    return H_tensor.reshape(d**N, d**N)