import numpy as np

class Sigmax:
	def __init__(self,nspins:int):
		self.nspins = nspins
		self.minflips = 1

	def H_state(self,state):
		mel = np.ones(self.nspins)
		flipsh = np.reshape(np.arange(self.nspins,dtype=float), (self.nspins,1))
		return mel, flipsh

class Sigmaz:
	def __init__(self,nspins:int):
		self.nspins = nspins
		self.minflips = 1

	def H_state(self,state):
		mel = [state[i] for i in range(self.nspins)]
		flipsh = np.array([[np.nan]]*self.nspins)
		return mel, flipsh