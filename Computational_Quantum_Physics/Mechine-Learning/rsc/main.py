import numpy as np
import H_Ising1d
import Nqs
import Sampler
import Trainer
import Sigma
import sigma_a
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def model_train(nspins,alpha,gfield,batch_size=100,steps=301):
	wf = Nqs.Nqs(nspins,alpha)
	wf.W = 0.1 * np.random.random(wf.W.shape) + 0j
	wf.a = 0.1 * np.random.random(wf.a.shape) + 0j
	wf.b = 0.1 * np.random.random(wf.b.shape) + 0j
	wf.state = np.random.choice([-1,1],nspins)

	h = H_Ising1d.Ising1d(nspins,gfield)

	trainer = Trainer.Trainer(h)
	wf, elist = trainer.train(wf,batch_size,steps,init_state=wf.state,out_freq=10,file='WaveFunction_batch100/Ising1d_20_{0}_{1}_'.format(alpha,gfield))

	return elist

def model_eval(nspins,alpha,op,nsweeps,model_file,state_file=None,opname = 'energy'):
	wf = Nqs.Nqs(nspins,alpha)
	wf.load_parameters(model_file)

	samp = Sampler.Sampler(wf,op,opname=opname)
	estav = samp.run(nsweeps,state_file)
	return estav

def plot(result,alpha):
	x_steps = np.arange(0,301,10)
	g = [0.01,0.1,0.5,1.0]

	fig = plt.figure(tight_layout=True)
	gs = gridspec.GridSpec(2,2)

	ax = fig.add_subplot(gs[0,:])
	for i in range(4):
		ax.plot(x_steps,result[i,:,0],label='g={}'.format(g[i]))

	legend = ax.legend(loc='center right', shadow=False, fontsize='small')
	ax.set_ylabel('Energy')
	ax.set_xlabel('steps')

	ax = fig.add_subplot(gs[1,0])
	for j in range(4):
		ax.plot(x_steps,result[j,:,1])
	ax.set_ylabel('<'+'$\sigma_x$'+'>')
	ax.set_xlabel('steps')

	ax = fig.add_subplot(gs[1,1])
	for j in range(4):
		ax.plot(x_steps,result[j,:,2])
	ax.set_ylabel('<'+'$\sigma_z$'+'>')
	ax.set_xlabel('steps')

	fig.align_labels()
	plt.savefig('figures/{}.png'.format(alpha))

def main():
	results = np.zeros((4,4,31,3))
	opname = ['energy','sigma_x','sigma_z']
	# Train and Sample
	for i,alpha in enumerate([2,4]):
		for j,gfield in enumerate([0.01,0.1,0.5,1.0]):
			model_train(20,alpha,gfield,steps=301)
			'''
			for k,step in enumerate(np.arange(0,301,10)):
				h = H_Ising1d.Ising1d(20,gfield)
				sigma_x = Sigma.Sigmax(20,1)
				sigma_z = Sigma.Sigmaz(20,1)
				results[i,j,k,0] = model_eval(20,alpha,h,400,'WaveFunction/Ising1d_20_{0}_{1}_{2}.npz'.format(alpha,gfield,step),opname = 'energy')
				results[i,j,k,1] = model_eval(20,alpha,sigma_x,400,'WaveFunction/Ising1d_20_{0}_{1}_{2}.npz'.format(alpha,gfield,step),opname = '<sigma_x>')
				results[i,j,k,2] = model_eval(20,alpha,sigma_z,400,'WaveFunction/Ising1d_20_{0}_{1}_{2}.npz'.format(alpha,gfield,step),opname = '<sigma_z>')
		for n in range(3):
			np.savetxt('Results/sample_{0}_{1}.txt'.format(alpha,opname[n]),results[i,:,:,n].T,'%.12f')
		plot(results[i],alpha)
		'''

def main1():
	results = np.zeros((3,4,31,3))
	opname = ['energy','sigma_x','sigma_z']

	for i,alpha in enumerate([2,4]):
		for j,gfield in enumerate([1.0,0.01,0.1,0.5,1.0]):
			for k,step in enumerate(np.arange(0,301,10)):
				h = H_Ising1d.Ising1d(20,gfield)
				sigma_x = sigma_a.Sigmax(20)
				sigma_z = sigma_a.Sigmaz(20)
				results[i,j,k,0] = model_eval(20,alpha,h,400,'WaveFunction_batch300/Ising1d_20_{0}_{1}_{2}.npz'.format(alpha,gfield,step),opname = 'energy')
				results[i,j,k,1] = model_eval(20,alpha,sigma_x,400,'WaveFunction_batch300/Ising1d_20_{0}_{1}_{2}.npz'.format(alpha,gfield,step),opname = '<sigma_x>')
				results[i,j,k,2] = model_eval(20,alpha,sigma_z,400,'WaveFunction_batch300/Ising1d_20_{0}_{1}_{2}.npz'.format(alpha,gfield,step),opname = '<sigma_z>')
		for n in range(3):
			np.savetxt('Results/sample_{0}_{1}.txt'.format(alpha,opname[n]),results[i,:,:,n].T,'%.12f')	
		plot(results[i],alpha)

if __name__ == '__main__':
	main()
	'''
	opname = ['energy','sigma_x','sigma_z']
	results = np.zeros((4,26,3))
	for n in range(3):
		results[:,:,n] = (np.loadtxt('Results/sample_{0}_{1}.txt'.format(4,opname[n]))).T
	plot(results,4)
	'''