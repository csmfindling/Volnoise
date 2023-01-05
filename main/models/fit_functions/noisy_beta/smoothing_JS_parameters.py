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
from scipy.special import digamma


def KL(distribution1, distribution2):
	return np.log(beta_func(distribution1[:, 0], distribution1[:, 1])/beta_func(distribution2[:,0], distribution2[:,1])) + \
									(distribution2[:,0] - distribution1[:,0]) * digamma(distribution2[:, 0]) + \
									(distribution2[:,1] - distribution1[:,1]) * digamma(distribution2[:,1]) + \
									(distribution1[:,0] - distribution2[:,0] + distribution1[:,1] - distribution2[:,1]) * digamma(distribution2[:,0] + distribution2[:,1])

# td = pickle.load(open('../../../../data/python/td_volnoise_subj_10_run_0_session_0.pkl')); actions = td['A_chosen'][np.newaxis]; rewards=td['reward'][np.newaxis]; params = [.2, 0.6, .02]; show_progress=True; numberOfStateSamples=200; latin_hyp_sampling =True;  epsilon_softmax=0.; noise_inertie = 0.; numberOfThetaSamples=200; coefficient = .5; beta_softmax=1.; lambda_noise=1.; eta_noise = .05; numberOfBetaSamples=20 ;alpha=0.; 
def smc(actions, rewards, parameters, nb_samples = 1000):

	T                 = actions.shape[0]	
	particles         = np.ones([nb_samples, 2])
	ancestors         = np.zeros([nb_samples], dtype = np.int)
	temp              = parameters[0]
	log_inc_marglkd   = 0 
	anc_particles     = np.ones([nb_samples, 2])
	noisefree_update  = np.ones([nb_samples, 2])
	distances         = np.zeros([nb_samples])
	average_noise     = np.zeros([nb_samples])
	means_beta        = np.zeros([nb_samples])


	all_particles        = np.ones([T, nb_samples, 2])
	all_noisefree_update = np.ones([T, nb_samples, 2])
	all_ancestors        = np.ones([T, nb_samples])
	all_distances        = np.ones([T, nb_samples])


	for t_idx in range(T):
		
		if t_idx > 0:
			anc_particles[:]       = particles[ancestors]
			noisefree_update[:, 0] = np.transpose(anc_particles[:, 0].T + (actions[t_idx - 1] != rewards[t_idx - 1]))
			noisefree_update[:, 1] = np.transpose(anc_particles[:, 1].T + (actions[t_idx - 1] == rewards[t_idx - 1]))
			
			distances[:]     = (KL(noisefree_update, anc_particles) + KL(anc_particles, noisefree_update))/2.
			average_noise[:] = parameters[-1][np.newaxis].T + distances * parameters[1][np.newaxis].T

			particles[:,0] = np.maximum(noisefree_update[:,0] - np.random.exponential(average_noise), 1)
			particles[:,1] = np.maximum(noisefree_update[:,1] - np.random.exponential(average_noise), 1)

		p_0_rewarding  = particles[:,0]/(particles[:,0] + particles[:,1])
		p_1_rewarding  = 1 - p_0_rewarding

		if actions[t_idx] == 0:
			log_p_act  = - np.logaddexp(0., (np.log(p_1_rewarding) - np.log(p_0_rewarding))/temp)
		else:
			log_p_act  = - np.logaddexp(0., (np.log(p_0_rewarding) - np.log(p_1_rewarding))/temp)

		log_inc_marglkd += logsumexp(log_p_act) - np.log(nb_samples)
		ancestors[:]     = uf.stratified_resampling(uf.to_normalized_weights(log_p_act))

		# save particles
		all_particles[t_idx]        = particles
		all_noisefree_update[t_idx] = noisefree_update
		all_ancestors[t_idx]        = ancestors
		all_distances[t_idx]        = distances

	return log_inc_marglkd, all_particles, all_noisefree_update, all_ancestors, all_distances




	
