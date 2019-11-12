import numpy as np

def Get_H(Ns):

    sigma_p = np.array([[0,1],[0,0]])
    sigma_m = np.array([[0,0],[1,0]])
    sigma_z = np.array([[0.5,0],[0,-0.5]])

    
    H = [None]*(Ns-1)
    
    for i in range(Ns-1):
        Hpm = [np.eye(2)]*Ns
        Hmp = [np.eye(2)]*Ns
        Hz = [np.eye(2)]*Ns

        Hpm[i] = sigma_p*0.5
        Hpm[i+1] = sigma_m
        Hmp[i] = sigma_m*0.5
        Hmp[i+1] = sigma_p
        Hz[i] = sigma_z
        Hz[i+1] = sigma_z

        A =1;B=1;C=1

        for j in range(Ns):
        	A = np.kron(A,Hpm[j])
        	B = np.kron(B,Hmp[j])
        	C = np.kron(C,Hz[j])

        H[i] = A+B+C
    
    return sum(H)

def Get_E(Ns):
    H = Get_H(Ns)
    print(H)
    
    E,V = np.linalg.eig(H)
    order = np.argsort(E)
    print(E[order])
    E0 = E[order[0]]/Ns
    print('%f' % E0)

if __name__ == "__main__":
    Get_E(8)