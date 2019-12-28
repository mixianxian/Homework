import numpy as np
from cmath import *
from scipy.linalg import circulant

class Nqs:
	"""docstring for Nqs"""
	def __init__(self, nspins, alpha):
		# alpha = number of hidden variables M / numnber of spins N
		# alpha is similar to bond dimension in MPS
		self.M = nspins * alpha
		self.N = nspins
		self.W = np.zeros((self.N,self.M))
		self.a = np.zeros(self.N)
		self.b = np.zeros(self.M)

		# theta = bi + \sum_j(Wij*sigma_zj)
		self.theta = np.zeros(self.M)

	def log_phi(self,state):
		# return log(|state>)
		logphi = np.dot(self.a,state)
		logphi += np.sum(np.log(2*np.cosh((self.b + np.dot(state,self.W)))))

		return logphi

	def log_pop(self,state,f):
		# return log(|state'>/|state>)
		if np.all(np.isnan(f)):
			return 0

		flips = self.reduce_flips(f)

		logpop = 0 + 0j # Use complex number
		logpop -= 2 * np.dot(self.a[flips], state[flips])
		# RuntimeWarning: overflow encountered in cosh
		
		logpop += np.sum(np.log(2*np.cosh((self.theta - 2 * np.dot(state[flips],self.W[flips]))))\
		        - np.log(2*np.cosh(self.theta)))

		return logpop

	def pop(self,state,flips):
		return np.exp(self.log_pop(state,flips))

	def reduce_flips(self,f):
		# strip all the NaN flip
		return f[np.invert(np.isnan(f))].astype(int)

	def init_theta(self,state):
		self.state = state
		self.theta = np.zeros(self.M)
		self.theta = self.b + np.dot(state,self.W)

		return None

	def update_theta(self,state,flips):
		if len(flips) == 0:
			return None

		self.theta -= 2 * np.dot(state[flips],self.W[flips])
		return None

	def load_parameters(self, filename):
		tmp = np.load(filename)
		self.a = tmp['a']
		self.b = tmp['b']
		self.W = tmp['W']
		self.N = len(self.a)
		self.M = len(self.b)

	def save_parameters(self,filename):
		np.savez(filename, a=self.a, b=self.b, W=self.W)