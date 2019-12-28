import numpy as np
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import LinearOperator
import Sampler
import random
from joblib import Parallel, delayed

class Trainer:
	def __init__(self, h, reg_list=(100,0.9,1e-4), cores=2, mag0=False):
		self.h = h # Hamiltonian
		self.nspins = h.nspins
		self.reg_list = reg_list # Parameters for regularization
		self.step_count = 0
		self.nvar = 0 # number of RBM(restrict boltzmann machine) variables
		self.parallel_cores = cores
		self.m = mag0

	def update_vector(self, wf, init_state, batch_size, gamma, step, therm=False):
		self.nvar = wf.N + wf.M + wf.N*wf.M
		wf.init_theta(init_state)
		samp = Sampler.Sampler(wf,self.h,mag0=self.m)
		# if init_state = np.array([]), Sampler.run() will do init_random_state()
		samp.state = np.copy(init_state)
		if therm == True:
			# where change the samp.state
			samp.thermalize(batch_size)
		#print(samp.state)

		results = Parallel(n_jobs=self.parallel_cores)(\
			delayed(get_sampler)(samp,self) for i in range(batch_size))

		#print(samp.state)
		#print("parallel_process\n")

		# three kinds of results with extra dimension "batch_size for" statistic average
		elocals = np.array([i[0] for i in results])
		deriv_vectors = np.array([i[1] for i in results])
		states = np.array([i[2] for i in results])

		# v = S^(-1)*F; W(p+1) = W(p) - gamma(p)*v; W are all the wf variables
		# S*v = F; So this is a problem to solve the quation and find v
		# Thus the cov_operator*v represent S*v, i.e. self.cov_operator
		'''
		cov_operator = LinearOperator((wf.N,wf.N), dtype=complex,\
			matvec=lambda v: self.cov_operator(v,deriv_vectors,step))
			'''
		forces = self.get_forces(elocals, deriv_vectors)
		vec, info = cg(self.cov_operator(deriv_vectors,step), forces)
		updates = -gamma * vec

		self.step_count += batch_size
		return updates, samp.state, np.mean(elocals) / self.nspins

	def get_elocal(self,state,wf):
		# E = (H|phi>)/(|phi>) = mel * (|phi'> / |phi>)
		if not all(state == wf.state):
			wf.init_theta

		mel,flips = self.h.H_state(state)
		eloc = sum([m*wf.pop(state,f) for m,f in zip(mel,flips)])

		return eloc

	def get_deriv_vector(self, state, wf):
		vector = np.zeros(self.nvar, dtype=complex)
		# refer equation S10 in SM of original paper
		for bias in range(wf.N):
			vector[bias] = state[bias]
		# RuntimeWarning: overflow encountered in tanh
		for bias in range(wf.M):
			vector[wf.N + bias] = np.tanh(wf.theta[bias])
		for n in range(wf.N):
			for m in range(wf.M):
				vector[wf.N + wf.M + wf.M*n + m] = state[n] * np.tanh(wf.theta[m])
		return vector

	def get_forces(self,elocals,deriv_vectors):
		# F = <E*deriv> - <E><deriv>
		emean = np.mean(elocals)
		omean = np.mean(deriv_vectors,axis=0)
		correlator = np.mean([i[0] * np.conj(i[1]) for i in zip(elocals,deriv_vectors)],axis=0)

		return correlator - emean*np.conj(omean)

	def cov_operator(self,deriv_vectors,step):
		# term1 = sum(<deriv_i * deriv_j> * vec_i)
		# term2 = sum(<deriv_j><deriv_i> * vec_i)
		# reg = max(lambda_0 * b**p,lambda_min) * (term1_i - term2_i) *vec
		corr_term = np.dot(deriv_vectors.T.conj(),deriv_vectors) / deriv_vectors.shape[0]
		#term1 = np.dot(corr_term,vec)
		dire_term = np.dot(np.mean(deriv_vectors.conj(), axis=0).reshape(deriv_vectors.shape[1],1),\
		            np.mean(deriv_vectors, axis=0).reshape(1,deriv_vectors.shape[1]))
		#term2 = np.dot(dire_term,vec)
		reg = max(self.reg_list[0] * self.reg_list[1] ** step, self.reg_list[2]) * (np.diag(corr_term) - np.diag(dire_term))
		'''
		tvec = np.dot(deriv_vectors, vec)
		term1 = np.dot(deriv_vectors.T.conj(), tvec) / deriv_vectors.shape[0]
		term2 = np.mean(deriv_vectors.conj(), axis=0) * np.mean(tvec)
		reg = max(self.reg_list[0] * self.reg_list[1] ** step, self.reg_list[2]) *\
		(np.mean(deriv_vectors.T.conj(),deriv_vectors,axis=0) - mean(deriv_vectors.conj(),axis=0)*mean(deriv_vectors,axis=0)) *vec
		'''
		return corr_term - dire_term + np.diag(reg)

	def train(self, wf, batch_size, num_steps, init_state=np.array([]), print_freq=20, file=' ', out_freq=0):
		state = init_state
		# Energy list
		elist = np.zeros(num_steps,dtype=complex)
		for step in range(num_steps):
			updates, state, elist[step] = self.update_vector(wf, state, batch_size, gamma_fun(step), step)
			self.apply_update(updates,wf)

			if step % print_freq == 0:
				print("Completed training step {}".format(step))
				print("Current energy per spin: {}".format(elist[step]))

			if out_freq > 0 and step % out_freq == 0:
				wf.save_parameters(file + str(step))

		return wf, elist

	def apply_update(self,updates,wf):
		wf.a += updates[0:wf.N]
		wf.b += updates[wf.N:wf.M + wf.N]
		wf.W += np.reshape(updates[wf.N + wf.M:], wf.W.shape)


def get_sampler(sampler,trainer):
	for i in range(sampler.nspins):
		sampler.move()
	return trainer.get_elocal(sampler.state, sampler.wf),\
	       trainer.get_deriv_vector(sampler.state, sampler.wf),\
	       sampler.state

def gamma_fun(p):
	return .01
