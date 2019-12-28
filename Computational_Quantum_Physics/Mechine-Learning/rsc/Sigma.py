import numpy as np

class Sigmax:
	def __init__(self,nspins:int,which_spin:int):
		self.nspins = nspins
		self.which = which_spin
		self.minflips = 1

	def H_state(self,state):
		mel = [1]
		flipsh = np.array([[self.which]])
		return mel, flipsh

class Sigmaz:
	def __init__(self,nspins:int,which_spin:int):
		self.nspins = nspins
		self.which = which_spin
		self.minflips = 1

	def H_state(self,state):
		mel = [state[self.which]]
		flipsh = np.array([[np.nan]])
		return mel, flipsh