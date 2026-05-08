import numpy as np

# define the vecrtors and Gellmann matrices
def vec_conj(a):
    return np.conjugate(a)

def vec_dot(a,b):
    # second vector is conjuagated
    return np.dot(a,vec_conj(b))

def dagger(M):
    return np.conjugate(np.transpose(M))

def mat_dot2(A,B):
    return np.dot(A,B)

def mat_dot4(A,B,C,D):
    return np.dot(A,np.dot(B,np.dot(C,D)))

def normalize(psi):
    return psi/np.sqrt(vec_dot(psi,psi))

def I_3():
    I3 = np.eye(3, dtype=complex)
    return I3

def l_1():
    l1 = np.matrix([[0,1.,0],[1.,0,0],[0,0,0]])
    return l1

def l_2():
    l2 = np.matrix([[0,-1j,0],[1j,0,0],[0,0,0]])
    return l2

def l_3():
    l3 = np.matrix([[0,0,0],[0,1.,0],[0,0,1.]])
    return l3

def l_4():
    l4 = np.matrix([[0,0,1.],[0,0,0],[1.,0,0]])
    return l4

def l_5():
    l5 = np.matrix([[0,0,-1j],[0,0,0],[1j,0,0]])
    return l5

def l_6():
    l6 = np.matrix([[0,0,0],[0,0,1.],[0,1.,0]])
    return l6

def l_7():
    l7 = np.matrix([[0,0,0],[0,0,-1j],[0,1j,0]])
    return l7

def l_8():
    l8 = np.matrix([[0,0,0],[0,1.,0],[0,0,-1.]])
    return l8

def gellmann_normal():
    return [np.eye(3),np.array(l_1()),np.array(l_2()),np.array(l_3()),np.array(l_4()),np.array(l_5()),np.array(l_6()),np.array(l_7()),np.array(l_8())]

def gellmann_tilde(g=1):
    return [np.eye(3),g*np.array(l_1()),g*np.array(l_2()),g*np.array(l_3()),g*np.array(l_4()),g*np.array(l_5()),g*np.array(l_6()),g*np.array(l_7()),g*np.array(l_8())]

def gellmann_bar(g=1):
    return [np.eye(3),(1/g)*np.array(l_1()),(1/g)*np.array(l_2()),(1/g)*np.array(l_3()),(1/g)*np.array(l_4()),(1/g)*np.array(l_5()),(1/g)*np.array(l_6()),(1/g)*np.array(l_7()),(1/g)*np.array(l_8())]
