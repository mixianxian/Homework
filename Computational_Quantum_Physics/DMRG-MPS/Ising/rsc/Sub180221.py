"""
Subroutine

Methods:
(1) Group(A,shapeA)
(2) NCon(Tensor,Index)
(3) SpinOper(ss)
(4) OutputT(T,prec=1.0e-10)
(5) OutputTFile(file,T,prec=1.0e-10)
(6) SplitEig(A,Dcut,prec=1.0e-12,safe=3)
(7) SplitEigh(A,Dcut,mode='P',prec=1.0e-12,safe=3,icheck=0)
(8) SplitSvd_Lapack(A,Dcut,iweight,mode='P',prec=1.0e-12,icheck=0)
(9) Mps_QR0P(T,icheck=0)
(10) Mps_LQ0P(T,icheck=0)

Copyright:
Shuo Yang, shuoyang@tsinghua.edu.cn
Nov 24, 2018, Tsinghua, Beijing, China
"""

import numpy as np
import numpy.linalg as LA
import scipy.sparse.linalg as LAs
from scipy import linalg
import itertools
#----------------------------------------------------------------
def Group(A,shapeA):
	""" transpose + reshape """
	dimA = np.asarray(np.shape(A))
	rankA = len(shapeA) # 自组合角标对数
	
	shapeB = []
	for i in range(0,rankA):
		shapeB += [np.prod(dimA[shapeA[i]])] # 自组合后角标维数
	
	orderB = sum(shapeA,[])
	A = np.reshape(np.transpose(A,orderB),shapeB) # 角标换次序，组合
	return A

def NCon(Tensor,Index):
	ConList = range(1,max(sum(Index,[]))+1) # 消除次序
	
	while len(ConList) > 0:
	
		Icon = [] # 本次消除的张量
		for i in range(len(Index)):
			if ConList[0] in Index[i]:
				Icon.append(i)
				if len(Icon) == 2:
					break
		
		if len(Icon) == 1: # 张量自身消除
			IndCommon = list(set([x for x in Index[Icon[0]] if Index[Icon[0]].count(x)>1])) # 本次消除张量的轴的次序代号
			
			for icom in range(len(IndCommon)):
				Pos = sorted([i for i,x in enumerate(Index[Icon[0]]) if x==IndCommon[icom]]) # 轴的位置
				Tensor[Icon[0]] = np.trace(Tensor[Icon[0]],axis1=Pos[0],axis2=Pos[1])
				Index[Icon[0]].pop(Pos[1]) # 消除张量求迹后的不对应
				Index[Icon[0]].pop(Pos[0])
		
		else: # 张量之间消除
			IndCommon = list(set(Index[Icon[0]]) & set(Index[Icon[1]]))		
			Pos = [[],[]]
			for i in range(2):
				for ind in range(len(IndCommon)):
					Pos[i].append(Index[Icon[i]].index(IndCommon[ind]))
			A = np.tensordot(Tensor[Icon[0]],Tensor[Icon[1]],(Pos[0],Pos[1]))
			
			for i in range(2):
				for ind in range(len(IndCommon)):
					Index[Icon[i]].remove(IndCommon[ind])
			Index[Icon[0]] = Index[Icon[0]]+Index[Icon[1]]
			Index.pop(Icon[1])
			Tensor[Icon[0]] = A
			Tensor.pop(Icon[1])
		
		ConList = list(set(ConList)^set(IndCommon))
	
	while len(Index) > 1: # 处理不参与消除的负数指标
		
		Tensor[0] = np.multiply.outer(Tensor[0],Tensor[1])
		Tensor.pop(1)
		Index[0] = Index[0]+Index[1]
		Index.pop(1)
	
	Index = Index[0]
	if len(Index) > 0:
		Order = sorted(range(len(Index)),key=lambda k:Index[k])[::-1]	
		Tensor = np.transpose(Tensor[0],Order)
	else:
		Tensor = Tensor[0]
	
	return Tensor

def SpinOper(ss):
	spin = (ss-1)/2.0 # 2s+1，自旋多重度ss
	dz = np.zeros(ss) # 自旋取值
	mp = np.zeros(ss-1)
	
	for i in range(ss):
		dz[i] = spin-i
	for i in range(ss-1):
		mp[i] = np.sqrt((2*spin-i)*(i+1))
	
	S0 = np.eye(ss)
	Sp = np.diag(mp,1) # s^+
	Sm = np.diag(mp,-1) # s^-
	Sx = 0.5*(Sp+Sm)
	Sy = -0.5j*(Sp-Sm)
	Sz = np.diag(dz)
	
	return S0,Sp,Sm,Sz,Sx,Sy
#----------------------------------------------------------------
def OutputT(T,prec=1.0e-10):
	""" output nonzero elements """
	pos = np.nonzero(np.abs(T)>prec)
	val = T[pos]
	pos = np.transpose(pos)
	for i in range(len(val)):
		print (pos[i],val[i])

def OutputTFile(file,T,prec=1.0e-10):
	""" output nonzero elements to file """
	pos = np.nonzero(np.abs(T)>prec)
	val = T[pos]
	pos = np.transpose(pos)
	val = np.reshape(val,[len(val),1])
	
	if LA.norm(np.imag(T)) > prec:
		out = np.concatenate((pos+1,np.real(val),np.imag(val)),axis=1)
		np.savetxt(file,out,fmt = '%d\t'*np.shape(pos)[1] + '%0.6f \t %0.6f')
	else:
		out = np.concatenate((pos,np.real(val)),axis=1)
		np.savetxt(file,out,fmt = '%d\t'*np.shape(pos)[1] + '%0.6f')
#----------------------------------------------------------------
def SplitEig(A,Dcut,prec=1.0e-12,safe=3):
	""" Eig, Lapack + Arpack """
	D = np.min(np.shape(A))
	
	if Dcut >= D-1:
		S,V = LA.eig(A)
	else:
		k_ask = min(Dcut+safe,D-2)
		S,V = LAs.eigs(A,k=k_ask)
	
	idx = np.isfinite(S)
	if idx.any() != False:
		S = S[idx]
		V = V[:,idx]
	order = np.argsort(abs(S))[::-1]
	S = S[order]
	V = V[:,order]
	
	S = S[abs(S/S[0])>prec]
	Dc = min(len(S),Dcut)
	S = S[:Dc]
	V = V[:,:Dc]
	
	return V,S,Dc

def SplitEigh(A,Dcut,mode='P',prec=1.0e-12,safe=3,icheck=0):
	""" Eig, Hermitian, Lapack + Arpack """
	""" P = preserve, C = cut """
	D = np.min(np.shape(A))
	
	if Dcut >= D-1:
		S,V = LA.eigh(A)
	else:
		k_ask = min(Dcut+safe,D-2)
		S,V = LAs.eigsh(A,k=k_ask)
	
	idx = np.isfinite(S)
	if idx.any() != False:
		S = S[idx]
		V = V[:,idx]
	order = np.argsort(abs(S))[::-1]
	S = S[order]
	V = V[:,order]
	
	if mode == 'P':
		S = S[abs(S/S[0])>1.0e-15]
		Dc = min(len(S),Dcut)
		S = S[:Dc]
		V = V[:,:Dc]
		S[abs(S/S[0])<prec] = prec*np.sign(S[abs(S/S[0])<prec])
		# 绝对值小于prec的值放大到prec
	
	if mode == 'C':
		S = S[abs(S/S[0])>prec]
		Dc = min(len(S),Dcut)
		S = S[:Dc]
		V = V[:,:Dc]
	
	if icheck == 1:
		print ( LA.norm(np.dot(V,np.dot(np.diag(S),np.transpose(np.conj(V))))-A)/LA.norm(A) )
	
	return V,S,Dc
#----------------------------------------------------------------
def SplitSvd_Lapack(A,Dcut,iweight,mode='P',prec=1.0e-12,icheck=0):
	""" SVD, Lapack only """
	""" P = perserve, C = cut """
	
	U,S,V = LA.svd(A,full_matrices=0)
	
	S = np.abs(S)
	idx = np.isfinite(S)
	if idx.any() != False:
		S = S[idx]
		U = U[:,idx]
		V = V[idx,:]
	order = np.argsort(S)[::-1]
	S = S[order]
	U = U[:,order]
	V = V[order,:]
	
	if mode == 'P':
		S = S[S/S[0]>1.0e-15]
		Dc = min(len(S),Dcut)
		S = S[:Dc]
		U = U[:,:Dc]
		V = V[:Dc,:]
		S[S/S[0]<prec] = prec
		
	if mode == 'C':
		S = S[S/S[0]>prec]
		Dc = min(len(S),Dcut)
		S = S[:Dc]
		U = U[:,:Dc]
		V = V[:Dc,:]
	
	if iweight == 1:
		U = np.dot(U,np.diag(np.sqrt(S)))
		V = np.dot(np.diag(np.sqrt(S)),V)
	
	if icheck == 1:
		if iweight == 0:
			print ( LA.norm(np.dot(U,np.dot(np.diag(S),V))-A)/LA.norm(A) )
		else:
			print ( LA.norm(np.dot(U,V)-A)/LA.norm(A) )
	return U,S,V,Dc
#----------------------------------------------------------------
def Mps_QRP(UL,T,icheck=0):
	""" (0-UL-1)(0-T-L) -> (0-Tnew-L)(0-UR-1) """
	shapeT = np.asarray(np.shape(T))
	rankT = len(shapeT)
	
	A = np.tensordot(UL,T,(1,0))
	A = np.reshape(A,[np.prod(shapeT[:-1]),shapeT[-1]])
	Tnew,UR = linalg.qr(A,mode = 'economic')
	Sign = np.diag(np.sign(np.diag(UR)))
	Tnew = np.dot(Tnew,Sign)
	UR = np.dot(Sign,UR)
	Tnew = np.reshape(Tnew,shapeT)
	
	if icheck == 1:
		A = np.reshape(A,shapeT)
		B = np.tensordot(Tnew,UR,(rankT-1,0))
		print ( LA.norm(A-B)/LA.norm(A) )
		A = np.tensordot(np.conj(Tnew),Tnew,(range(0,rankT-1),range(0,rankT-1)))
		print ( LA.norm(A-np.eye(shapeT[-1])) )
	
	return Tnew,UR

def Mps_LQP(T,UR,icheck=0):
	""" (0-T-L)(0-UR-1) -> (0-UL-1)(0-Tnew-L) """
	shapeT = np.asarray(np.shape(T))
	rankT = len(shapeT)
	
	A = np.tensordot(T,UR,(rankT-1,0))
	A = np.reshape(A,[shapeT[0],np.prod(shapeT[1:])])
	UL,Tnew = linalg.rq(A,mode = 'economic')
	Sign = np.diag(np.sign(np.diag(UL)))
	UL = np.dot(UL,Sign)
	Tnew = np.dot(Sign,Tnew)
	Tnew = np.reshape(Tnew,shapeT)
	
	if icheck == 1:
		A = np.reshape(A,shapeT)
		B = np.tensordot(UL,Tnew,(1,0))
		print ( LA.norm(A-B)/LA.norm(A) )
		A = np.tensordot(np.conj(Tnew),Tnew,(range(1,rankT),range(1,rankT)))
		print ( LA.norm(A-np.eye(shapeT[0])) )
	
	return UL,Tnew

def Mps_QR0P(T,icheck=0):
	""" (0-T-L) -> (0-Tnew-L)(0-UR-1) """
	shapeT = np.asarray(np.shape(T))
	rankT = len(shapeT)
	
	A = np.reshape(T,[np.prod(shapeT[:-1]),shapeT[-1]])
	Tnew,UR = linalg.qr(A,mode = 'economic')
	Sign = np.diag(np.sign(np.diag(UR)))
	Tnew = np.dot(Tnew,Sign)
	UR = np.dot(Sign,UR)
	Tnew = np.reshape(Tnew,shapeT)
	
	if icheck == 1:
		B = np.tensordot(Tnew,UR,(rankT-1,0))
		print ( LA.norm(T-B)/LA.norm(T) )
		A = np.tensordot(np.conj(Tnew),Tnew,(range(0,rankT-1),range(0,rankT-1)))
		print ( LA.norm(A-np.eye(shapeT[-1])) )
	
	return Tnew,UR

def Mps_LQ0P(T,icheck=0):
	""" (0-T-L) -> (0-UL-1)(0-Tnew-L) """
	shapeT = np.asarray(np.shape(T))
	rankT = len(shapeT)
	
	A = np.reshape(T,[shapeT[0],np.prod(shapeT[1:])])
	UL,Tnew = linalg.rq(A,mode = 'economic')
	Sign = np.diag(np.sign(np.diag(UL)))
	UL = np.dot(UL,Sign)
	Tnew = np.dot(Sign,Tnew)
	Tnew = np.reshape(Tnew,shapeT)
	
	if icheck == 1:
		B = np.tensordot(UL,Tnew,(1,0))
		print (LA.norm(T-B)/LA.norm(T))
		A = np.tensordot(np.conj(Tnew),Tnew,(range(1,rankT),range(1,rankT)))
		print (LA.norm(A-np.eye(shapeT[0])))
	
	return UL,Tnew
#----------------------------------------------------------------
if __name__ == "__main__":
	Test = {}
	Test['Group'] = 1
	Test['NCon'] = 1
	Test['SplitEig'] = 1
	Test['SplitSvd'] = 1
	Test['Mps_QRP'] = 1
	
	if Test['Group'] == 1:
		A = np.random.rand(3,2,4,3)
		A = Group(A,[[0,2],[3,1]])
		print (A)
	
	if Test['NCon'] == 1:
		T1 = np.random.rand(2,3,4,3)
		T2 = np.random.rand(4,2,3,5)
		T3 = np.random.rand(3,3,2,4)
	
		A = np.tensordot(T1,T2,([0,3],[1,2]))
		A = np.transpose(A,[2,0,3,1])
		B = NCon([T1,T2],[[1,-2,-4,2],[-1,1,2,-3]])
		print (LA.norm(A-B)/LA.norm(A))
	
		A = np.tensordot(T1,T2,(3,2))
		A = np.transpose(A,[1,3,4,2,0,5])
		B = NCon([T1,T2],[[-5,-1,-4,1],[-2,-3,1,-6]])
		print (LA.norm(A-B)/LA.norm(A))
	
		A = np.tensordot(T1,T2,([0,3],[1,2]))
		A = np.tensordot(A,T3,([0,1],[1,3]))
		A = np.transpose(A,[3,0,1,2])
		B = NCon([T1,T2,T3],[[1,3,4,2],[-2,1,2,-3],[-4,3,-1,4]])
		print (LA.norm(A-B)/LA.norm(A))
		
		T1 = np.random.rand(2,3)
		T2 = np.random.rand(3,4)
		T3 = np.random.rand(4,2)
	
		A = np.kron(T1,T2)
		A = np.reshape(A,[2,3,3,4])
		B = NCon([T1,T2],[[-1,-3],[-2,-4]])
		print (LA.norm(A-B)/LA.norm(A))
	
		A = np.kron(T1,T2)
		A = np.kron(A,T3)
		A = np.reshape(A,[2,3,4,3,4,2])
		A = np.transpose(A,[1,0,5,4,2,3])
		B = NCon([T1,T2,T3],[[-2,-6],[-1,-4],[-5,-3]])
		print (LA.norm(A-B)/LA.norm(A))
		
		T = np.random.rand(3,2,4,3,2)
	
		A = np.eye(3)
		A = np.tensordot(T,A,([0,3],[0,1]))
		A = np.transpose(A,[2,0,1])
		B = NCon([T],[[1,-2,-3,1,-1]])
		print (LA.norm(A-B)/LA.norm(A))
	
		A = np.eye(3*2)
		A = np.reshape(A,[3,2,3,2])
		A = np.tensordot(T,A,([0,1,3,4],[0,1,2,3]))
		B = NCon([T],[[1,2,-1,1,2]])
		print (LA.norm(A-B)/LA.norm(A))
		
		T1 = np.random.rand(2,2)
		T2 = np.random.rand(2,2)
		T3 = np.random.rand(2,2)
	
		A = np.tensordot(T1,T2,(1,0))
		A = np.kron(A,T3)
		A = np.reshape(A,[2,2,2,2])
		B = NCon([T1,T2,T3],[[-1,1],[1,-3],[-2,-4]])
		print (LA.norm(A-B)/LA.norm(A))
	
		T1 = np.random.rand(3,3)
		T2 = np.random.rand(3,1,3)
		T3 = np.random.rand(3,3,1)
		T4 = np.random.rand(3,3,3,3)
	
		A = np.tensordot(T1,T2,(1,0))
		A = np.tensordot(A,T3,(1,2))
		A = np.tensordot(A,T4,([0,1,2,3],[0,1,2,3]))
		B = NCon([T1,T2,T3,T4],[[3,1],[1,2,4],[5,6,2],[3,4,5,6]])
		print (LA.norm(A-B)/LA.norm(A))
	
	if (Test['SplitEig'] == 1) | (Test['SplitSvd'] == 1):
		T = np.zeros((3,3,3))
		T[1,1,1] = 1
		T[1,0,2] = 1
		T[0,2,1] = 1
		T[2,1,0] = 1
		T[1,2,0] = -1
		T[2,0,1] = -1
		T[0,1,2] = -1
		
		G = np.zeros((3,2,3))
		G[1,1,0] = 1
		G[0,1,1] = 1
		G[1,0,2] = 1
		G[2,0,1] = 1
	
		A = np.tensordot(T,G,(0,0))
		A = np.tensordot(A,G,(0,0))
		A = np.tensordot(A,G,(0,0))
		A = np.tensordot(A,np.conj(A),([0,2,4],[0,2,4]))
		STA = Group(A,[[0,3],[1,4],[2,5]])
	
		A = np.reshape(T,[1,3,3,3])
		A = np.tensordot(A,np.conj(A),(0,0))
		STB = Group(A,[[0,3],[1,4],[2,5]])
	
		A = np.tensordot(STA,STB,([0],[0]))
		ST0 = np.transpose(A,[3,0,1,2])
	
	if Test['SplitEig'] == 1:
		ST = Group(ST0,[[3,0],[2,1]])
		STh = (ST+np.transpose(ST))/2
		ST0h = np.reshape(STh,[9,9,9,9])
		Dfull = np.shape(ST)[0]
		print (Dfull)
		
		print ('full eig')
		V,S,Dc = SplitEig(STh,Dfull)
		Sfull = S
		print (Sfull,len(Sfull))
		
		print ('partial eig')
		for Dcut in range(1,Dfull+1):
			V,S,Dc = SplitEig(STh,Dcut)
			print (Dcut,Dc,LA.norm(abs(S)-abs(Sfull[:Dc])))
		
		print ('full eigh')
		V,S,Dc = SplitEigh(STh,Dfull,icheck=1)
		Sfull = S
		print (Sfull,len(Sfull))
		
		print ('partial eigh')
		for Dcut in range(1,Dfull+1):
			V,S,Dc = SplitEigh(STh,Dcut,icheck=1)
			print (Dcut,Dc,LA.norm(abs(S)-abs(Sfull[:Dc])))

	if Test['SplitSvd'] == 1:
		ST = Group(ST0,[[0,1],[2,3]])
		ST1 = Group(ST0,[[0],[1],[2,3]])
		ST2 = Group(ST0,[[0,1],[2],[3]])
		Dfull = np.shape(ST)[0]
		print (Dfull)

		print ('full svd')
		for iweight in range(2):
			U,S,V,Dc = SplitSvd_Lapack(ST,Dfull,iweight,icheck=1)
		Sfull = S
		print (Sfull,len(Sfull))
		
		print ('partial svd')
		for Dcut in range(1,Dfull+1):
			for iweight in range(2):
				U,S,V,Dc = SplitSvd_Lapack(ST,Dcut,iweight,mode='C',icheck=1)
				print (Dcut,iweight,Dc,LA.norm(S-Sfull[:Dc]))
		
	if Test['Mps_QRP'] == 1:
		print ('no Z2, rank 3')
		for D1,D2,D3 in itertools.product(range(3,5),range(3,5),range(3,5)):
			D = [D1,D2,D3]
			T0 = np.random.rand(D[0],D[1],D[2]) + 1j*np.random.rand(D[0],D[1],D[2])
			UL = np.random.rand(D[0],D[0]) + 1j*np.random.rand(D[0],D[0])
			UR = np.random.rand(D[2],D[2]) + 1j*np.random.rand(D[2],D[2])
			
			print (D1,D2,D3)
			T,UR = Mps_QRP(UL,T0,icheck=1)
			UL,T = Mps_LQP(T0,UR,icheck=1)
			T,UR = Mps_QR0P(T0,icheck=1)
			UL,T = Mps_LQ0P(T0,icheck=1)
		
		print ('no Z2, rank 4')
		for D1,D2,D3,D4 in itertools.product(range(3,5),range(3,5),range(3,5),range(3,5)):
			D = [D1,D2,D3,D4]
			T0 = np.random.rand(D[0],D[1],D[2],D[3]) + 1j*np.random.rand(D[0],D[1],D[2],D[3])
			UL = np.random.rand(D[0],D[0]) + 1j*np.random.rand(D[0],D[0])
			UR = np.random.rand(D[3],D[3]) + 1j*np.random.rand(D[3],D[3])
			
			print (D1,D2,D3,D4)
			T,UR = Mps_QRP(UL,T0,icheck=1)
			UL,T = Mps_LQP(T0,UR,icheck=1)
			T,UR = Mps_QR0P(T0,icheck=1)
			UL,T = Mps_LQ0P(T0,icheck=1)
	