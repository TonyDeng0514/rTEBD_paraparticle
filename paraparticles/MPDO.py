import numpy as np
from scipy.linalg import svd

from .Hamiltonian import draw_dis, draw_q, build_bond_gates, n_loc
from .build_MPDO_from_mps import random_config, initial_MPDO_dict
from .GellMann import gellmann_tilde

class MPDO:
    # Hamiltonian parameters
    
    
    def __init__(self,L,N,Na,t_hop,W,dt,T,chi,seed,g):
        self.L      = L
        self.N      = N
        self.Na     = Na
        self.t_hop  = t_hop
        self.W      = W
        self.chi    = chi
        self.T      = T
        self.N      = N
        # self.dt = self.T/self.N
        self.dt     = dt        
        self.seed   = seed
        self.rng    = np.random.default_rng(self.seed)
        self.g      = g
        tilde = gellmann_tilde(self.g)
        self.n_coeffs = np.array([np.trace(n_loc @ tilde[j]) for j in range(9)])

        self.Omega_list = draw_dis(self.L, 0, self.W, self.rng)
        self.V_list = draw_dis(self.L - 1, 0, self.W, self.rng)
        self.q_list = draw_q(self.L, 0, self.W, self.rng)

        # Initialize U
        self.odd_bonds_U, self.even_bonds_U = build_bond_gates(self.L,
                                                               self.t_hop,
                                                               self.Omega_list,
                                                               self.q_list,
                                                               self.V_list,
                                                               self.dt,
                                                               self.g)
        # Initial state
        self.config = random_config(self.L, self.N, self.Na, self.rng) # if we want to run different random disorder in H 
                                                                        # but keep the same initial state, change self.seed
                                                                        # to a fixed seed
        
        self.A_dict = initial_MPDO_dict(self.L, self.config, self.g)
        self.lmbd_position = 0

        # initialize all your measurements
        self.tr_TEBD = 0
        self.ni_persite = np.zeros(self.L, dtype=np.complex128)
        self.E_persite = np.zeros(self.L-1, dtype=np.complex128)
        self.E_total_TEBD = 0

        self.measure_TEBD()
            
    def applyU(self,ind,dirc,U):
        
        # This part relocates lmbd to the right position
        if dirc == 'left':
            self.lmbd_relocate(ind[1])
        elif dirc == 'right':
            self.lmbd_relocate(ind[0])
        
        
            
        A1 = self.A_dict["A"+str(ind[0])]
        A2 = self.A_dict["A"+str(ind[1])]
        chi1 = np.shape(A1)[0]
        chi2 = np.shape(A2)[2]
        
        s1 = np.einsum('ijkl,akb,blc->aijc',U,A1,A2,optimize='optimal')
        
        s2 = np.reshape(s1,(9*chi1,9*chi2))
        try:
            Lp,lmbd,R=np.linalg.svd(s2,full_matrices=False)
        except np.linalg.LinAlgError as err:
            if "SVD did not converge" in str(err):
                Lp,lmbd,R=svd(s2,full_matrices=False,lapack_driver='gesvd')
                f = open("py_print.txt","a")
                f.write("SVD convergence issue")
                f.close()
            else:
                raise
        chi12 = np.min([9*chi1,9*chi2])
        chi12_p = np.min([self.chi,chi12])
        lmbd = np.diag(lmbd)
    
        # Truncation step
        lmbd = lmbd[:chi12_p,:chi12_p]
        Lp = Lp[:,:chi12_p]
        R = R[:chi12_p,:]
    
        if (dirc == 'left'):
            A1 = np.reshape(np.dot(Lp,lmbd),(chi1,9,chi12_p))
            A2 = np.reshape(R,(chi12_p,9,chi2))
            self.lmbd_position = ind[0]
            
        elif (dirc == 'right'):
            A1 = np.reshape(Lp,(chi1,9,chi12_p))
            A2 = np.reshape(np.dot(lmbd,R),(chi12_p,9,chi2))
            self.lmbd_position = ind[1]
            
        self.A_dict["A"+str(ind[0])] = A1
        self.A_dict["A"+str(ind[1])] = A2
        
    # Function to move lmbd right
    def move_lmbd_right(self,ind):
        I = np.reshape(np.eye(81),(9,9,9,9))
        self.applyU([ind,ind+1],'right',I)
    
    # Function to move lmbd left
    def move_lmbd_left(self,ind):
        I = np.reshape(np.eye(81),(9,9,9,9))
        self.applyU([ind,ind+1],'left',I)
        
    def sweepU(self):

        for i,U in enumerate(self.odd_bonds_U):
            self.applyU([2*i,2*i+1],'right',U)
        for i, U in enumerate(self.even_bonds_U):
            self.applyU([2*i+1,2*i+2],'right',U)
    
        
        self.measure_TEBD()
    
    # Relocates lmbd from lmbd_position to ind
    def lmbd_relocate(self,ind):
        step = ind - self.lmbd_position
        for i in range(np.abs(step)):
            if step > 0:
                self.move_lmbd_right(self.lmbd_position)
            elif step < 0:
                self.move_lmbd_left(self.lmbd_position-1)
                
    def measure_TEBD(self):
        
        self.left_trace = []
        self.right_trace = []
        self.build_left()
        self.build_right()
        
        #Measure n_i
        for i in range(self.L):
            self.ni_persite[i] = self.tensordot_n(i)

        # Measure trace
        temp = 3*self.A_dict["A0"][:,0,:]
        for i in range(1,self.L):
            temp = np.tensordot(temp, 3*self.A_dict["A"+str(i)][:,0,:], axes=1)
        self.tr_TEBD = temp.flatten()[0]
    
    def build_left(self):
        temp = np.reshape(1.+0.*1j,(1,1))
        self.left_trace.append(temp)
        for i in range(1,self.L):
            temp = np.tensordot(temp, 3*self.A_dict["A"+str(i-1)][:,0,:], axes=1)
            self.left_trace.append(temp)
        
    def build_right(self):
        temp = np.reshape(1.+0.*1j,(1,1))
        self.right_trace.append(temp)
        loop_arr = np.arange(self.L-2,-1,-1)
        for i in loop_arr:
            temp = np.tensordot(3*self.A_dict["A"+str(i+1)][:,0,:], temp, axes=1)
            self.right_trace.append(temp)
        self.right_trace.reverse()
    
    def tensordot_n(self, ind):
        A_ind = self.A_dict["A"+str(ind)]
        L_env = self.left_trace[ind]
        R_env = self.right_trace[ind]
        result = 0.
        for j in range(9):
            if abs(self.n_coeffs[j]) > 1e-12:  # skip zero terms
                result += self.n_coeffs[j] * (L_env @ A_ind[:,j,:] @ R_env).flatten()[0]
        return result