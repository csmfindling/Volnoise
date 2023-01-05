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

# parameters = pickle.load(open('../models/utils/sobol_1000_3.pkl')); td = pickle.load(open('../../../../data/python/td_volnoise_subj_10_run_0_session_0.pkl')); params = [1, 1, 1]; show_progress=True; numberOfStateSamples=200; latin_hyp_sampling =True;  epsilon_softmax=0.; noise_inertie = 0.; numberOfThetaSamples=200; coefficient = .5; beta_softmax=1.; lambda_noise=1.; eta_noise = .05; numberOfBetaSamples=20 ;alpha=0.; 
def smc(actions, rewards, parameters):

	nb_settings, T    = actions.shape[:2]
	nb_samples        = 100
	nb_param, nb_dims = parameters.shape	
	particles         = np.ones([nb_settings, nb_param, nb_samples, 2])
	ancestors         = np.zeros([nb_settings, nb_param, nb_samples], dtype = np.int)
	log_inc_marglkd   = 0
	anc_particles     = np.ones([nb_settings, nb_param, nb_samples, 2])
	noisefree_update  = np.ones([nb_settings, nb_param, nb_samples, 2])
	distances         = np.zeros([nb_settings, nb_param, nb_samples])
	average_noise     = np.zeros([nb_settings, nb_param, nb_samples])

	for t_idx in range(T):


		if t_idx > 0:
			anc_particles[:]             = np.reshape(np.array([particles[i, j, ancestors[i, j]] for i in range(nb_settings) for j in range(nb_param)]), (nb_settings, nb_param, nb_samples, 2))
			noisefree_update[:, :, :, 0] = np.transpose(anc_particles[:, :, :, 0].T + (actions[:, t_idx - 1] != rewards[:, t_idx - 1]))
			noisefree_update[:, :, :, 1] = np.transpose(anc_particles[:, :, :, 1].T + (actions[:, t_idx - 1] == rewards[:, t_idx - 1]))
			
			
			distances[:] = np.log(beta_func(noisefree_update[:,:,:,0], noisefree_update[:,:,:,1])/beta_func(anc_particles[:,:,:,0], anc_particles[:,:,:,1])) + \
									(anc_particles[:,:,:,0] - noisefree_update[:,:,:,0]) * digamma(anc_particles[:,:,:,0]) + \
									(anc_particles[:,:,:,1] - noisefree_update[:,:,:,1]) * digamma(anc_particles[:,:,:,1]) + \
									(noisefree_update[:,:,:,0] - anc_particles[:,:,:,0] + noisefree_update[:,:,:,1] - anc_particles[:,:,:,1]) * digamma(anc_particles[:,:,:,0] + anc_particles[:,:,:,1])

			average_noise[:] = 1./parameters[:,2][np.newaxis].T + distances/parameters[:,1][np.newaxis].T
			particles[:]     = np.maximum(1, np.transpose(np.random.rand(particles.shape[0], particles.shape[1], particles.shape[2], particles.shape[3]).T * (1 - 1/average_noise).T + 1/average_noise.T) * noisefree_update)

		p_0_rewarding  = beta_func(particles[:,:,:,0] + 1, particles[:,:,:,1])/beta_func(particles[:,:,:,0], particles[:,:,:,1])
		p_1_rewarding  = 1 - p_0_rewarding

		log_p_act      = np.transpose(np.transpose(- np.logaddexp(0., (np.log(p_1_rewarding) - np.log(p_0_rewarding))/parameters[:,0][np.newaxis].T).T * (1 - actions[:,t_idx]) \
											- np.logaddexp(0., (np.log(p_0_rewarding) - np.log(p_1_rewarding))/parameters[:,0][np.newaxis].T).T * actions[:,t_idx]).T * (actions[:,t_idx] != -1))

		log_inc_marglkd += (logsumexp(log_p_act, axis=2) - np.log(nb_samples)) * (actions[:,t_idx][np.newaxis].T != -1)

		weights_norm     = np.transpose(np.exp(log_p_act.T - np.max(log_p_act,axis=2).T))
		weights_norm     = np.transpose(weights_norm.T/np.sum(weights_norm, axis=2).T)

		ancestors        = uf.stratified_resampling_tensor(weights_norm)

	return log_inc_marglkd




	
