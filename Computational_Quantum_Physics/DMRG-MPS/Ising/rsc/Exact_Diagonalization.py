import numpy as np
import sys
import matplotlib.pyplot as plt

sigma_x = np.array([[0,1],[1,0]])
sigma_z = np.array([[1,0],[0,-1]])

def Get_H(Ns,g): 
    H = [None]*Ns
    for i in range(Ns):
        h_x = [np.eye(2)]*Ns
        h_x[i] = -g*sigma_x
        A = 1
        for j in range(Ns):
            A = np.kron(A,h_x[j])
        H[i] = A

    for i in range(Ns-1):
        h_z = [np.eye(2)]*Ns
        h_z[i] = -sigma_z
        h_z[i+1] = sigma_z
        B = 1
        for j in range(Ns):
            B = np.kron(B,h_z[j])
        H[i] = H[i] + B
    return sum(H)

def Get_operator(Ns,site,op):
    A = 1
    for i in range(Ns):
        if i != site:
            A = np.kron(A,np.eye(2))
        else:
            A = np.kron(A,op)
    return A

def Get_Value_per_Site(Ns,psi):
    sigmaz = [None]*Ns
    sigmax = [None]*Ns

    for i in range(Ns):
        sz = Get_operator(Ns,i,sigma_z)
        sx = Get_operator(Ns,i,sigma_x)

        sigmaz[i] = np.dot(psi.transpose(),np.dot(sz,psi))
        sigmax[i] = np.dot(psi.transpose(),np.dot(sx,psi))
    
    return sigmaz, sigmax


def Get_Value(Ns,g):
    H = Get_H(Ns,g)  
    
    E,V = np.linalg.eig(H)
    E = np.real(E)
    V = np.real(V)
    order = np.argsort(E)
    
    E0 = E[order[0]]/Ns
    psi0 = V[:,order[0]]
    SZ0, SX0 = Get_Value_per_Site(Ns,psi0)

    return E0, SZ0, SX0

def plot(result,g):
	fig,ax = plt.subplots()

	ax.plot(g,result[0],'k-',label='Energy')
	ax.plot(g,result[1],'b-',label='<Sx> site 1')
	ax.plot(g,result[2],'b--',label='<Sx> site 2')
	ax.plot(g,result[3],'r-',label='<Sz> site 1')
	ax.plot(g,result[4],'r--',label='<Sz> site 2')

	legend = ax.legend(loc='lower left', bbox_to_anchor=(0.7,0.25), shadow=False, fontsize='small')

	plt.savefig(sys.argv[0][:-3]+".png")

def main(Ns,g):
    E0, SZ0, SX0 = Get_Value(Ns,g)

    print('Energy: %f' % E0)
    print('<sigmaZ>: ',  np.abs(SZ0))
    print('<sigmaX>: ', np.abs(SX0))

    return E0,SX0[0:2],SZ0[0:2]    

if __name__ == "__main__":
    np.set_printoptions(precision=10, suppress=True, threshold=10000, linewidth=300)

    for Ns in [10]:
        result = np.zeros([5,60])
        for i,g in enumerate(np.arange(0,1.2,0.02)):
            E,SX,SZ = main(Ns,g)
            result[:,i] = [E,SX[0],SX[1],np.abs(SZ[0]),np.abs(SZ[1])]
        plot(result,np.arange(0,1.2,0.02))