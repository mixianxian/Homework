import math
from Nqs import *

class Sampler:
	def __init__(self,wf,op,quiet=False,opname ='energy',mag0=False):
		self.wf = wf # wave function Nqs
		self.nspins = self.wf.N
		self.operator = op # Hamiltonian operator H_Ising1d
		self.observable = []
		self.writestates = True
		self.flips = []
		self.nmoves = 0
		self.accepted = 0
		self.state = np.ones(self.nspins)
		self.nflips = op.minflips
		self.quiet = quiet
		self.opname = opname
		self.mag0 = mag0

	def rand_spins(self):
		# randomly flip 1/2(self.nflips) spins
		# return whether this random spin is meaningful
		self.flips = np.random.randint(0, self.nspins, self.nflips)
		if self.nflips == 2:
			if not self.mag0:
				return self.flips[0] != self.flips[1]
			else:
				return self.state[self.flips[0]] != self.state[self.flips[1]]

		return True

	def init_random_state(self):
		if not self.mag0:
			self.state = np.random.choice([-1,1],self.nspins)
		else:
			if self.nspins % 2 != 0:
				raise ValueError('Need even number of spins to have zero magnetization!')
			base_array = np.concatenate(\
				(np.ones(int(self.nspins / 2)), -1 * np.ones(int(self.nspins / 2))))
			self.state = np.random.permutation(base_array)

	def reset_av(self):
		self.nmoves = 0
		self.accepted = 0

	def acceptance(self):
		return self.accepted / self.nmoves

	def move(self,filename=None):
		# try to get next spin configuration S
		if self.rand_spins():
			accept_prob = np.abs(self.wf.pop(self.state,self.flips)) ** 2
			accept_prob = min(1,accept_prob)
			if accept_prob > np.random.random():
				self.wf.update_theta(self.state,self.flips)
				# update state
				for flip in self.flips:
					self.state[flip] *= -1
					self.accepted += 1						
		self.nmoves += 1

		if (self.writestates) and (filename != None):
			with open(filename,'a') as fout:
				fout.write(' '.join(map(str,self.state))+'\n')

	def measure_operator(self):
		# E = (H|phi>)/(|phi>) = mel * (|phi'> / |phi>)
		mel,flips = self.operator.H_state(self.state)
		en = sum([m*self.wf.pop(self.state,f) for m, f in zip(mel, flips)])
		self.observable.append(en)

	def thermalize(self,thermmoves):
		for sweep in range(thermmoves):
			self.move()

	def run(self,nsweeps,filename=None,thermfactor=0.1,sweepfactor=1,init_state=np.array([])):
		if thermfactor > 1 or thermfactor < 0:
			raise ValueError('Invalid thermfactor (0~1)')
		if nsweeps < 50:
			raise ValueError('Use more sweeps (>50)')
		if not self.quiet:
			print("Starting Monte Carlo sampling, nsweeps = {}".format(nsweeps))
		
		if init_state.size == 0:
			self.init_random_state()
		else:
			self.state = init_state
		self.wf.init_theta(self.state)

		self.reset_av()
		if not self.quiet:
			print("Beginning thermalization...")
		if filename != None:
			with open(filename,'a') as fout:
				fout.write("Thermalization:\n")
		for thermsweep in range(math.floor(thermfactor * nsweeps)):
			for spinmove in range(sweepfactor * self.nspins):
				self.move(filename)
		if not self.quiet:
			print("Thermalization done.")

		self.reset_av()
		if not self.quiet:
			print("Sweeping...")
		if filename != None:
			with open(filename,'a') as fout:
				fout.write("\nSweeping:\n")
		for sweep in range(nsweeps):
			for spinmove in range(sweepfactor * self.nspins):
				self.move(filename)
			self.measure_operator()
		if not self.quiet:
			print("Sweeping done. Acceptance rate was = {}".format(self.acceptance()))
		estav = self.output_energy()
		return estav

	# Binning analysis
	def output_energy(self):
		nblocks = 50
		blocksize = math.floor(len(self.observable) / nblocks)
		enmean = 0
		enmeansq = 0
		enmean_unblocked = 0
		enmeansq_unblocked = 0

		for i in range(nblocks):
			eblock = 0
			for j in range(i * blocksize, (i + 1) * blocksize):
				eblock += self.observable[j].real
				delta = self.observable[j].real - enmean_unblocked
				enmean_unblocked += delta / (j + 1)
				delta2 = self.observable[j].real - enmean_unblocked
				enmeansq_unblocked += delta * delta2

			eblock /= blocksize
			delta = eblock - enmean
			enmean += delta / (i + 1)
			delta2 = eblock - enmean
			enmeansq += delta * delta2

		enmeansq /= (nblocks - 1)
		enmeansq_unblocked /= (nblocks * blocksize - 1)
		if self.opname=='energy':
			estav = enmean / self.nspins
			esterror = sqrt(enmeansq / nblocks) / self.nspins
		else:
			estav = abs(enmean) / self.nspins
			esterror = sqrt(enmeansq / nblocks) / self.nspins
		esterror = esterror.real
		# self.estav = estav
		if not self.quiet:
			print("Estimated average " + self.opname +" per spin: {0} +/- {1}".format(estav, esterror))
			print("Error estimated with binning analysis consisting of {0} bins".format(nblocks))
			print("Block size is {}".format(blocksize))
			print("Estimated autocorrelation time is {}".format(0.5 * blocksize * enmeansq / enmeansq_unblocked))

		return estav