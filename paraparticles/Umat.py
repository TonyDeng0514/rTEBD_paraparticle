import numpy as np
from .GellMann import dagger, gellmann_bar, gellmann_normal, gellmann_tilde, mat_dot4



# build U_mat from U
def U_mat(U, g=1):
#     Hi = transverse_ising(J,hx,hz)
#     U = expm(-1j*dt*Hi)
    Ud = dagger(U)
    U_all = np.zeros((9,9,9,9),dtype=np.complex128)
    for i in range(9):
        for j in range(9):
            for k in range(9):
                for l in range(9):
                    sg1 = np.kron(gellmann_bar(g)[i],gellmann_bar(g)[j])
                    sg2 = np.kron(gellmann_tilde(g)[k],gellmann_tilde(g)[l])
                    U_all[i][j][k][l] = np.trace(mat_dot4(sg1,U,sg2,Ud))
    return U_all

def check_Umat(U_all):
    for i in range(9):
        for j in range(9):
            for k in range(9):
                for l in range(9):
                    expected = (1.0 if (i==k and j==l) else 0.0)
                    assert abs(U_all[i,j,k,l] - expected) < 1e-10, \
                        f"Failed at i={i},j={j},k={k},l={l}: got {U_all[i,j,k,l]}"
    print("Umat identity check passed.")

if __name__ == "__main__":
    I9 = np.eye(9, dtype=complex)
    U_test = U_mat(I9, g=1)
    check_Umat(U_test)