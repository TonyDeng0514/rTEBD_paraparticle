import numpy as np
from .GellMann import dagger, gellmann_bar, gellmann_normal, gellmann_tilde, mat_dot4



def U_mat(U, g=1):
    """Convert a 9x9 two-site unitary U into the rank-4 superoperator tensor U_all.

    U_all[i,j,k,l] = Tr((bar[i]xbar[j]) @ U @ (tilde[k]xtilde[l]) @ U†) / (N[i]*N[j])

    The division by N[i]*N[j] (N[0]=1, N[j]=2 for j=1..8) compensates for the norm-2
    dual pair convention Tr(bar[j]@tilde[k])=2*delta_{jk}, so that the einsum in applyU
    maps true expansion coefficients to true expansion coefficients. For U=identity,
    U_all[i,j,k,l] = delta_{ik}*delta_{jl} for all i,j,k,l in 1..8.
    """
    Ud = dagger(U)
    U_all = np.zeros((9,9,9,9),dtype=np.complex128)
    N = np.ones(9); N[1:] = 2.0   # Tr(bar[j]@tilde[j]): 1 for j=0, 2 for j=1..8
    for i in range(9):
        for j in range(9):
            for k in range(9):
                for l in range(9):
                    sg1 = np.kron(gellmann_bar(g)[i],gellmann_bar(g)[j])
                    sg2 = np.kron(gellmann_tilde(g)[k],gellmann_tilde(g)[l])
                    U_all[i][j][k][l] = np.trace(mat_dot4(sg1,U,sg2,Ud)) / (N[i]*N[j])
    return U_all

def check_Umat(U_all):
    """Verify U_all[i,j,k,l] = delta_{ik}*delta_{jl} for i,j,k,l=1..8 when built from U=identity."""
    for i in range(1,9):
        for j in range(1,9):
            for k in range(1,9):
                for l in range(1,9):
                    expected = (1.0 if (i==k and j==l) else 0.0)
                    assert abs(U_all[i,j,k,l] - expected) < 1e-10, \
                        f"Failed at i={i},j={j},k={k},l={l}: got {U_all[i,j,k,l]}"
    print("Umat identity check passed.")

if __name__ == "__main__":
    I9 = np.eye(9, dtype=complex)
    U_test = U_mat(I9, g=2)
    check_Umat(U_test)