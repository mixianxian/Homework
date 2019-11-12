import numpy as np

Sx = 0.5 * np.array([[0,1],[1,0]])
Sy = 0.5 * np.array([[0,-1j],[1j,0]])
Sz = 0.5 * np.array([[1,0],[0,-1]])
I = np.eye(2)

# Express Hamiltonian with direct product of four spin operator
H = np.kron(np.kron(Sx,I),np.kron(I,Sx)) + np.kron(np.kron(Sy,I),np.kron(I,Sy)) + np.kron(np.kron(Sz,I),np.kron(I,Sz)) +\
    np.kron(np.kron(Sx,Sx),np.kron(I,I)) + np.kron(np.kron(Sy,Sy),np.kron(I,I)) + np.kron(np.kron(Sz,Sz),np.kron(I,I)) +\
    np.kron(np.kron(I,Sx),np.kron(Sx,I)) + np.kron(np.kron(I,Sy),np.kron(Sy,I)) + np.kron(np.kron(I,Sz),np.kron(Sz,I)) +\
    np.kron(np.kron(I,I),np.kron(Sx,Sx)) + np.kron(np.kron(I,I),np.kron(Sy,Sy)) + np.kron(np.kron(I,I),np.kron(Sz,Sz))

# Our problem contains only real numbers
H = np.real(H)

# Eigenvalue of Hamiltonian matrix
evalue,evector = np.linalg.eig(H)

# Save our results in file
np.savetxt('Result_Direct_Product.txt',H,fmt='%.1f')
with open('Result_Direct_Product.txt','r+') as f:
    old = f.read()
    f.seek(0)
    f.write('Matrix of Hamiltonian:\n')
    f.write(old)
    f.write('\nEigenvalue:\n')
    for i in evalue:
        f.write('{:.1f}'.format(i)+' ')
