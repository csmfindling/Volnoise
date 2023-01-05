import numpy as np
from scipy.stats import gamma, norm, truncnorm, multivariate_normal, beta
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal as multi_norm
from scipy.misc import logsumexp
import sys
sys.path.append("../../utils/")
import useful_functions as uf
import pickle
from scipy.special import beta as beta_func

# td = pickle.load(open('../../../../data/python/td_volnoise_subj_10_run_0_session_0.pkl')); params = [1, 1, 1]; show_progress=True; numberOfStateSamples=200; latin_hyp_sampling =True;  epsilon_softmax=0.; noise_inertie = 0.; numberOfThetaSamples=200; coefficient = .5; beta_softmax=1.; lambda_noise=1.; eta_noise = .05; numberOfBetaSamples=20 ;alpha=0.; 
def smc2(actions, rewards, params):

	actions           = np.asarray(td['A_chosen'], dtype=np.intc)
	rewards           = np.ascontiguousarray(td['reward'])
	nb_samples        = 1000
	T, dx, N_dx       = actions.shape[0], 0.001, 1000
	x_range           = np.tile(np.arange(0,1,dx), (nb_samples, 1))
	particles         = np.ones([nb_samples, 2])
	ancestors         = np.zeros([nb_samples], dtype = np.int)
	temp, mu, lambdaa = params
	log_inc_marglkd   = 0
	anc_particles     = np.ones([nb_samples, 2])
	noisefree_update  = np.ones([nb_samples, 2])
	distances         = np.zeros([nb_samples])
	average_noise     = np.zeros([nb_samples])

	for t_idx in range(T):
		
		if t_idx > 0:
			anc_particles[:]       = particles[ancestors]
			noisefree_update[:, 0] = anc_particles[:, 0] + (actions[t_idx - 1] != rewards[t_idx - 1])
			noisefree_update[:, 1] = anc_particles[:, 1] + (actions[t_idx - 1] == rewards[t_idx - 1])
			
			
			distances[:] = np.sum(np.transpose(np.abs( \
									  		np.power(x_range.T, noisefree_update[:,0] - 1) \
													* np.power((1 - x_range).T, noisefree_update[:,1] - 1) \
															/ beta_func(noisefree_update[:,0], noisefree_update[:,1]) * dx \
											- np.power(x_range.T, anc_particles[:,0] - 1) \
													* np.power((1 - x_range).T, anc_particles[:,1] - 1) \
															/ beta_func(anc_particles[:,0], anc_particles[:,1])  * dx \
										  ) \
									) \
							  , axis = -1)


			# distances_0   = np.zeros([nb_samples]); x0       = 0
			# for idx_N_dx in range(N_dx):
			# 	distances_0 += np.abs(beta.pdf(x0, noisefree_update[:,0], noisefree_update[:,1]) * dx - beta.pdf(x0, anc_particles[:,0], anc_particles[:,1]) * dx)
			# 	x0        += dx
			# assert(np.all(np.abs(distances_0 - distances) < 1e-10))

			average_noise[:] = 1 + mu + lambdaa * distances
			particles[:]     = np.maximum(1, np.transpose(np.random.rand(particles.shape[0], particles.shape[1]).T * (1 - 1/average_noise) + 1/average_noise) * noisefree_update)

		p_0_rewarding  = beta_func(particles[:,0] + 1, particles[:,1])/beta_func(particles[:,0], particles[:,1])
		p_1_rewarding  = 1 - p_0_rewarding

		if actions[t_idx] == 0:
			log_p_act  = - np.logaddexp(0., (np.log(p_1_rewarding) - np.log(p_0_rewarding))/temp)
		else:
			log_p_act  = - np.logaddexp(0., (np.log(p_0_rewarding) - np.log(p_1_rewarding))/temp)

		log_inc_marglkd += logsumexp(log_p_act) - np.log(nb_samples)
		ancestors[:]     = uf.stratified_resampling(uf.to_normalized_weights(log_p_act))








	
