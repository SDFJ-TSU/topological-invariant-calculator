# -*- coding: utf-8 -*-
"""
Topological invariant calculator

@author: Ziyi Chen
1060140508324302621264569262251643506286332804535242424320936009244396687298197422217070322818885159203129600
"""

#%% Import
import numpy as np
import numpy.linalg as la

# Pauli Matrix
SIGMA_X = np.array([[0, 1 + 0j], [1 + 0j, 0]])
SIGMA_Y = np.array([[0, 0 - 1j], [0 + 1j, 0]])
SIGMA_Z = np.array([[1 + 0j, 0], [0, -1 + 0j]])
SIGMA_0 = np.array([[1, 0], [0, 1]])

#%% Main Function
def calc_SSH(v, w = 1):
    # Winding number and Chern Simmon invariant for SSH model
    SSH_set = lambda k: SSH(k, v, w)
    W = winding_number(SSH_set, [0], 2000)
    CS = Chern_Simons(SSH_set, [0], 2000)
    print('Winding number of SSH (v/w = {0}) = {1}'.format(v/w, W))
    print('Chern-Simons invariant of SSH (v/w = {0}) = {1}'.format(v/w, CS))
    
def calc_modified_SSH(v, t, w = 1):
    # Winding number and Chern Simmon invariant for modified SSH model (see figure 6(c))
    modified_SSH_set = lambda k: modified_SSH(k, v, t, w)
    W = winding_number(modified_SSH_set, [0], 2000)
    CS = Chern_Simons(modified_SSH_set, [0], 2000)
    print('Winding number of modified SSH (v/w = {0}, t/w = {1}) = {2}'.format(v/w, t/w, W))
    print('Chern-Simons invariant of modified SSH (v/w = {0}, t/w = {1}) = {2}'.format(v/w, t/w, CS))
    
def calc_QWZ(u, v = 1):
    # Chern number for QWZ model
    QWZ_set = lambda k: QWZ(k, u, v)
    Ch = Chern_number(QWZ_set, [0], 100, 100)
    print('Chern number of QWZ (u = {0}, v = {1}) = {2}'.format(u, v, Ch))
    
def calc_BHZ(u):
    BHZ_set = lambda k: BHZ(k, u, 1)
    FKM = FKM_number(BHZ_set, [0, 1], 100, 100)
    print('Fu-Kane invariant of BHZ (u = {0}) = {1}'.format(u, FKM))

#%% Model
def SSH(k, v, w):
    # SSH model
    hamiltonian = (v + w * np.cos(k)) * SIGMA_X + w * np.sin(k) * SIGMA_Y
    chiral = SIGMA_Z
    return hamiltonian, chiral

def modified_SSH(k, v, t, w = 1):
    # Modified SSH model (see figure 6(c))
    hamiltonian = (v + w * np.cos(k) + t * np.cos(2*k)) * SIGMA_X + (w * np.sin(k) + t * np.sin(2*k)) * SIGMA_Y
    chiral = SIGMA_Z
    return hamiltonian, chiral

def QWZ(k, u, v):
    # QWZ model
    # k is a 2D vector [kx, ky]
    kx = k[0]
    ky = k[1]
    hamiltonian = np.sin(kx) * SIGMA_X + np.sin(ky) * SIGMA_Y + (u + v*np.cos(kx) + v*np.cos(ky)) * SIGMA_Z
    return hamiltonian

def BHZ(k, u, v = 1):
    # BHZ model
    # k is a 2D vector [kx, ky]
    kx = k[0]
    ky = k[1]
    
    hamiltonian = np.kron(SIGMA_0, (u + v * np.cos(kx) + v * np.cos(ky)) * SIGMA_Z + np.sin(ky) * SIGMA_Y) + \
        np.kron(SIGMA_Z, np.sin(kx) * SIGMA_X)
    
    return hamiltonian


#%% Winding number
def winding_number(hamiltonian, nf, J):
    # nf is the valence band, starting from 0
    J = int(J)
        
    # Index 2 momentum
    def momentum(k):
        return k * np.pi * 2 / J
    
    # braket matrix
    def braket_matrix(k, chiral):
        return np.conjugate(eigen_vect[k][:, nf].T) @ \
            chiral @ eigen_vect[k + 1][:, nf]
    
    # Eigen vector
    eigen_vect = []
    k_list = np.linspace(0, J, J + 1)
    for _k in k_list:
        _, vect = la.eigh(hamiltonian(momentum(_k))[0])
        eigen_vect.append(vect)
    chiral = hamiltonian(momentum(0))[1]

    # Calculation
    W = 0
    for _k in k_list[:-1]:
        _k = int(_k)
        W += np.trace(braket_matrix(_k, chiral))
    W = round(np.real(W * 1j / np.pi))
    
    return W

#%% Chern Simons number
def Chern_Simons(hamiltonian, nf, J):
    J = int(J)
    
    # Index 2 momentum
    def momentum(k):
        return k * np.pi * 2 / J
    
    # braket matrix
    def braket_matrix(k):
        return np.conjugate(eigen_vect[k][:, nf].T) @ eigen_vect[k + 1][:, nf]

    # Eigen vector
    eigen_vect = []
    k_list = np.linspace(0, J, J + 1)
    for _k in k_list:
        _, vect = la.eigh(hamiltonian(momentum(_k))[0])
        eigen_vect.append(vect)
    
    # Calculation
    CS = 0
    for _k in k_list[:-1]:
        _k = int(_k)
        CS += np.angle(la.det(braket_matrix(_k)))
    CS = round(CS * 1 / np.pi)%2
    
    return CS

#%% Chern number
def Chern_number(hamiltonian, nf, Jx, Jy):
    # nf is the band you cancered about. It usually is the valance band, starting from 0
    Jx = int(Jx)
    Jy = int(Jy)
    
    # Index 2 momentum
    def momentum(k):
        kx = (k % int(Jx + 1)) * np.pi * 2 / Jx - np.pi
        ky = (int(k) / int(Jx + 1)) * np.pi * 2 / Jy - np.pi
        return [kx, ky]
    
    def k_inline(kx, ky):
        return int(kx + ky * (Jx + 1))
    
    # braket matrix
    def braket_matrix(k):
        loop_edge1 = np.conjugate(eigen_vect[k][:, nf].T) @ eigen_vect[k+1][:, nf]
        loop_edge2 = np.conjugate(eigen_vect[k+1][:, nf].T) @ eigen_vect[k+1+(Jx+1)][:, nf]
        loop_edge3 = np.conjugate(eigen_vect[k+1+(Jx+1)][:, nf].T) @ eigen_vect[k+(Jx+1)][:, nf]
        loop_edge4 = np.conjugate(eigen_vect[k+(Jx+1)][:, nf].T) @ eigen_vect[k][:, nf]
        
        return loop_edge1 @ loop_edge2 @ loop_edge3 @ loop_edge4
    
    # Eigen vector
    eigen_vect = []
    kx_list = np.linspace(0, Jx, Jx + 1)
    ky_list = np.linspace(0, Jy, Jy + 1)
    for _ky in ky_list:
        for _kx in kx_list:
            _k = k_inline(_kx, _ky)
            _, vect = la.eigh(hamiltonian(momentum(_k)))
            eigen_vect.append(vect)
    
    # Calculation
    #ch_list = [] 
    Ch = 0
    for _ky in ky_list[:-1]:
        for _kx in kx_list[:-1]:
            _k = k_inline(_kx, _ky)
            ch = np.angle(la.det(braket_matrix(_k)))
            Ch += ch
            #ch_list.append(Ch * 5000 / np.pi)
    Ch = round(-Ch * 1 / (2 * np.pi))
    #np.savetxt('ch.csv', ch_list, delimiter = ',')
    
    return Ch

#%% Fu Kane Mele number
def FKM_number(hamiltonian, nf, Jx, Jy):
    # nf is the band you cancered about. It usually is the valance band, starting from 0
    Jx = int(Jx)
    Jy = int(int(Jy) / 2 * 2) # Ensure Jy is even
    
    # Index 2 momentum
    def momentum(k):
        # Return [kx, ky]
        kx = (k % int(Jx + 1)) * np.pi * 2 / Jx - np.pi
        ky = (int(k) / int(Jx + 1)) * np.pi * 2 / Jy - np.pi
        return [kx, ky]
    
    def k_inline(kx, ky):
        return int(kx + ky * (Jx + 1))
    
    # braket matrix
    def braket_matrix(k):
        loop_edge1 = np.conjugate(eigen_vect[k][:, nf].T) @ eigen_vect[k+1][:, nf]
        loop_edge2 = np.conjugate(eigen_vect[k+1][:, nf].T) @ eigen_vect[k+1+(Jx+1)][:, nf]
        loop_edge3 = np.conjugate(eigen_vect[k+1+(Jx+1)][:, nf].T) @ eigen_vect[k+(Jx+1)][:, nf]
        loop_edge4 = np.conjugate(eigen_vect[k+(Jx+1)][:, nf].T) @ eigen_vect[k][:, nf]
        
        return loop_edge1 @ loop_edge2 @ loop_edge3 @ loop_edge4
    
    # Eigen vector
    eigen_vect = []
    kx_list = np.linspace(0, Jx, Jx + 1)
    ky_list = np.linspace(0, Jy, Jy + 1)
    for _ky in ky_list:
        for _kx in kx_list:
            _k = k_inline(_kx, _ky)
            _, vect = la.eigh(hamiltonian(momentum(_k)))
            eigen_vect.append(vect)
    
    # Calculation
    W_matrix = np.identity(np.size(nf))
    #w_list = []
    for _ky in ky_list[:-1]:
        for _kx in kx_list[int(Jy/2):-1]:
            _k = k_inline(_kx, _ky)
            W_matrix = W_matrix @ braket_matrix(_k)
            exp_Fmk, _ = la.eig(W_matrix)
            Fmk = np.angle(exp_Fmk)
            #w_list.append(Fmk * 5000 / np.pi)
    
    #np.savetxt('fkm.csv', w_list, delimiter = ',')
    return round((Fmk[0]-Fmk[1]) / (2 * np.pi))%2


#%% Main
# SSH model
calc_SSH(v=0.5, w=1)
# modified SSH model
calc_modified_SSH(v=0.5, t=0.8, w=1)
# QWZ model
calc_QWZ(u=-1.2, v=1)
# BHZ model
calc_BHZ(u=-1.2)
