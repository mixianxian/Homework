import numpy as np

class Ising1d:
	"""docstring for Ising1d"""
	def __init__(self, nspins:int, gfield:float, pbc:bool=False):
		self.nspins = nspins
		self.gfield = gfield
		self.pbc = pbc
		self.minflips =1
		print("# Using the 1d TF Ising model with {0} spins and h = {1}".format(self.nspins,self.gfield))

	def H_state(self,state):
		# State is a vector of 1/-1

		# mel is matrix elements of H on the basis of all states
		# mel[0] is the diagonal one

		# sigma_x|1/-1> = |-1/1>
		# sigma_z|1> = |1>; sigma_z|-1> = -|-1>
		mel = -1 * self.gfield * np.ones(self.nspins + 1)
		mel[0] = 0
		mel[0] -= sum([state[i]*state[i+1] for i in range(self.nspins -1)])
		if self.pbc:
			mel[0] -= state[-1] * state[0]

		# flips is the spin flip site between |state> and H|state>
		# NaN indicates no flips, which nqs can handle that
		flips = np.reshape(np.arange(self.nspins+1,dtype=float), (self.nspins+1,1)) - 1
		flips[0] = np.array([np.nan])

		return mel, flips