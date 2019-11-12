"""
Apdated from MPS_OneSite to calculate transverse field Ising model on an 1D chain
For homework of Yang's class

By Li Xiang, xiang-li19@mails.tsinghua.edu.cn
Nov 09, 2019, Tsinghua, Beijing, China
"""

import numpy as np
import numpy.linalg as LA
import scipy.sparse.linalg as LAs
import Sub180221 as Sub
import math,sys,subprocess,glob,copy
import matplotlib.pyplot as plt
"""
Mi = 
S0	-Sz	-g*Sx
0	0	Sz
0	0	S0
"""

Sx = np.array([[0,1],[1,0]])
Sz = np.array([[1,0],[0,-1]])

def GetMpo_Ising_Obc(g,Dp):
	S0 = np.eye(Dp)
	
	Dmpo = 3
	Mpo = np.zeros((Dmpo,Dp,Dmpo,Dp))
	
	Mpo[0,:,0,:] = S0
	Mpo[0,:,1,:] = -Sz
	Mpo[0,:,2,:] = -g*Sx
	Mpo[1,:,2,:] = Sz
	Mpo[2,:,2,:] = S0
	
	return Mpo

def GetMpo_Identity(Dp):
	Dmpo = 2
	Mpo = np.zeros((Dmpo,Dp,Dmpo,Dp))

	Mpo[0,:,0,:] = np.eye(Dp)
	Mpo[1,:,1,:] = np.eye(Dp)

	return Mpo

def GetMpo_OneSite(op,Dp):
	Dmpo = 2
	Mpo = np.zeros((Dmpo,Dp,Dmpo,Dp))

	Mpo[0,:,0,:] = np.eye(Dp)
	Mpo[0,:,1,:] = op
	Mpo[1,:,1,:] = np.eye(Dp)

	return Mpo

def InitMps(Ns,Dp,Ds):
	T = [None]*Ns
	# Refer to page 17 on 20191021 slides
	# Bond dimension of T varies along the chain
	for i in range(Ns):
		Dl = min(Dp**i,Dp**(Ns-i),Ds)
		Dr = min(Dp**(i+1),Dp**(Ns-1-i),Ds)
		T[i] = np.random.rand(Dl,Dp,Dr)
	
	U = np.eye(np.shape(T[-1])[-1])
	for i in range(Ns-1,0,-1):
		U,T[i] = Sub.Mps_LQP(T[i],U)
	
	return T

def InitH(Mpo,T):
	Ns = len(T)
	Dmpo = np.shape(Mpo)[0]
	
	HL = [None]*Ns
	HR = [None]*Ns
	
	HL[0] = np.zeros((1,Dmpo,1))
	HL[0][0,0,0] = 1.0
	HR[-1] = np.zeros((1,Dmpo,1))
	HR[-1][0,-1,0] = 1.0
	
	for i in range(Ns-1,0,-1):
		HR[i-1] = Sub.NCon([HR[i],T[i],Mpo,np.conj(T[i])],[[1,3,5],[-1,2,1],[-2,2,3,4],[-3,4,5]])
	for i in range(0,Ns-1):
		HL[i+1] = Sub.NCon([HL[i],np.conj(T[i]),Mpo,T[i]],[[1,3,5],[1,2,-1],[3,4,-2,2],[5,4,-3]])
	
	return HL,HR

def Operator_Identity(Mpo,T):
	Ns = len(T)
	Dmpo = np.shape(Mpo)[0]
	Dp = np.shape(Mpo)[1]

	opL = [None]*Ns
	opR = [None]*Ns

	opL[0] = np.zeros((1,Dmpo,1))
	opL[0][0,0,0] = 1.0
	opR[-1] = np.zeros((1,Dmpo,1))
	opR[-1][0,-1,0] = 1.0

	I_mpo = GetMpo_Identity(Dp)

	for i in range(Ns-1,0,-1):
			opR[i-1] = Sub.NCon([opR[i],T[i],I_mpo,np.conj(T[i])],[[1,3,5],[-1,2,1],[-2,2,3,4],[-3,4,5]])	
	for i in range(0,Ns-1):
			opL[i+1] = Sub.NCon([opL[i],np.conj(T[i]),I_mpo,T[i]],[[1,3,5],[1,2,-1],[3,4,-2,2],[5,4,-3]])
	
	return opL, opR

def Get_Value(op,T,Dp):
	Ns = len(T)
	Mpo = GetMpo_OneSite(op,Dp)
	opL, opR = Operator_Identity(Mpo,T)
	value = [None]*Ns
	for site in range(Ns):
		value[site] = Sub.NCon([opL[site],Mpo,opR[site],T[site],np.conj(T[site])],[[5,3,1],[3,2,7,4],[8,7,6],[1,2,8],[5,4,6]])
	
	return value

def OptT_TwoSite(Mpo,HL,HR,T1,T2,Method=1):
    Dl = T1.shape[0]
    Dm = T1.shape[2]
    Dr = T2.shape[2]
    Dp = Mpo.shape[1]

    T = Sub.NCon([T1,T2],[[-1,-2,1],[1,-3,-4]])
    DT = np.shape(T)
    DT_V = np.prod(DT)

    if Method == 1:
        def UpdateV(V):
            V = np.reshape(V,DT)
            V = Sub.NCon([HL,V,Mpo,Mpo,HR],[[-1,3,1],[1,2,6,5],[3,2,4,-2],[4,6,7,-3],[5,7,-4]])
            V = np.reshape(V,[DT_V])
            return V
		
        V0 = np.reshape(T,[DT_V])

        MV = LAs.LinearOperator((DT_V,DT_V),matvec=UpdateV)
        Eig,V = LAs.eigsh(MV,k=1,which='SA',v0=V0)

        T_mix = np.reshape(V,[Dl*Dp,Dp*Dr])
        T1,s,T2,Dm = Sub.SplitSvd_Lapack(T_mix,Dm,0)
        
        T1 = T1.reshape(Dl,Dp,-1)
        T2 = T2.reshape(-1,Dp,Dr)

    return T1,T2,s,Eig

def Get_Entropy(s):
	S = 0
	s1 = s**2
	s2 = np.log(s1)

	return -sum(s1*s2)

def OptT(Mpo,HL,HR,T):
    Ns = len(T)
    Eng0 = np.zeros(Ns)
    Eng1 = np.zeros(Ns)

    File_Entopy = open('Entropy.txt','w')

    for r in range(100):
        Entropy = [None]*Ns
        for i in range(Ns-1):
            T[i],T[i+1],s,Eng1[i] = OptT_TwoSite(Mpo,HL[i],HR[i+1],T[i],T[i+1])
            T[i],U = Sub.Mps_QR0P(T[i])
            HL[i+1] = Sub.NCon([HL[i],np.conj(T[i]),Mpo,T[i]],[[1,3,5],[1,2,-1],[3,4,-2,2],[5,4,-3]])
            T[i+1] = Sub.NCon([U,np.diag(s),T[i+1]],[[-1,1],[1,2],[2,-2,-3]])
            Entropy[i] = Get_Entropy(s)
        File_Entopy.write(str(Entropy)[1:-1]+'\n')
        Entropy = [None]*Ns	
        for i in range(Ns-1,0,-1):
            T[i-1],T[i],s,Eng1[i] = OptT_TwoSite(Mpo,HL[i-1],HR[i],T[i-1],T[i])
            U,T[i] = Sub.Mps_LQ0P(T[i])
            HR[i-1] = Sub.NCon([HR[i],T[i],Mpo,np.conj(T[i])],[[1,3,5],[-1,2,1],[-2,2,3,4],[-3,4,5]])
            T[i-1] = Sub.NCon([U,np.diag(s),T[i-1]],[[2,-3],[1,2],[-1,-2,1]])
            Entropy[i] = Get_Entropy(s)
        File_Entopy.write(str(Entropy)[1:-1]+'\n')

        
        if abs(Eng1[1]-Eng0[1]) < 1.0e-7:
            break
        Eng0 = copy.copy(Eng1)
    
    File_Entopy.close()    
    print (Eng1/float(Ns))
    
    return T,Eng1[0]/float(Ns)

def plot(result,g,Ds):
	fig,ax = plt.subplots()

	ax.plot(g,result[0],'k-',label='Energy')
	ax.plot(g,result[1],'b-',label='<Sx> site 1')
	ax.plot(g,result[2],'b--',label='<Sx> site 2')
	ax.plot(g,result[3],'r-',label='<Sz> site 1')
	ax.plot(g,result[4],'r--',label='<Sz> site 2')

	legend = ax.legend(loc='lower left', bbox_to_anchor=(0.7,0.25), shadow=False, fontsize='small')

	plt.savefig(sys.argv[0][:-3]+"_{}.png".format(Ds))

def main(Ns,g,Dp,Ds):
	Mpo = GetMpo_Ising_Obc(g,Dp)
	T = InitMps(Ns,Dp,Ds)
	HL,HR = InitH(Mpo,T)
	T,E = OptT(Mpo,HL,HR,T)

	SX0 = Get_Value(Sx,T,Dp)
	SZ0 = Get_Value(Sz,T,Dp)
	print('<sigmaZ>: ',  np.abs(SZ0))
	print('<sigmaX>: ', np.abs(SX0))

	return E,SX0[0:2],SZ0[0:2]

#-------------------------------------------
if __name__ == "__main__":
    np.set_printoptions(precision=10, suppress=True, threshold=10000, linewidth=300)
    Dp = 2
    '''
    for Ns in [10]:
        for Ds in [4]:
            result = np.zeros([5,60])
            for i,g in enumerate(np.arange(0,1.2,0.02)):
                E,SX,SZ = main(Ns,g,Dp,Ds)
                result[:,i] = [E,SX[0],SX[1],np.abs(SZ[0]),np.abs(SZ[1])]
            plot(result,np.arange(0,1.2,0.02),Ds)
	'''
    main(10,1,Dp,4)      
