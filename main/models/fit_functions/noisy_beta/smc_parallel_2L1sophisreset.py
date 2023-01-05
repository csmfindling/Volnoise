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


def L1(distribution1, distribution2, par_confirmed, par_disconfir):
	normalized_distribution1 = distribution1 * 1. / np.expand_dims(np.sum(distribution1, axis=-1), -1)
	normalized_distribution2 = distribution2 * 1. / np.expand_dims(np.sum(distribution2, axis=-1), -1)

	proba0old = normalized_distribution1[:,:,:,0]
	proba0new = normalized_distribution2[:,:,:,0]
	
	confirmed = (proba0old <= 0.5) * (proba0new < proba0old) + (proba0old >= 0.5) * (proba0new > proba0old)
	disconfir = (proba0old > 0.5 ) * (proba0new < proba0old) + (proba0old < 0.5 ) * (proba0new > proba0old)
	
	du        = np.abs(proba0old - proba0new) * (confirmed * par_confirmed + disconfir * par_confirmed)
	
	return du


def smc(actions, rewards, parameters, nb_samples):

	nb_settings, T    = actions.shape[:2]
	nb_param, nb_dims = parameters.shape
	particles         = np.ones([nb_settings, nb_param, nb_samples, 2])
	ancestors         = np.zeros([nb_settings, nb_param, nb_samples], dtype = np.int)
	mus               = parameters[:,3]
	epsilons          = parameters[:,2]
	log_inc_marglkd   = 0 
	anc_particles     = np.ones([nb_settings, nb_param, nb_samples, 2])
	noisefree_update  = np.ones([nb_settings, nb_param, nb_samples, 2])
	distances         = np.zeros([nb_settings, nb_param, nb_samples])
	means_beta        = np.zeros([nb_settings, nb_param, nb_samples])


	for t_idx in range(T):
		# print(t_idx)
		
		if t_idx > 0:
			anc_particles[:]             = np.reshape(np.array([particles[i, j, ancestors[i, j]] for i in range(nb_settings) for j in range(nb_param)]), (nb_settings, nb_param, nb_samples, 2))
			noisefree_update[:, :, :, 0] = np.transpose(anc_particles[:, :, :, 0].T + (actions[:, t_idx - 1] != rewards[:, t_idx - 1]))
			noisefree_update[:, :, :, 1] = np.transpose(anc_particles[:, :, :, 1].T + (actions[:, t_idx - 1] == rewards[:, t_idx - 1]))
					
			distances[:]     = L1(anc_particles, noisefree_update, parameters[:,1][np.newaxis].T, parameters[:,3][np.newaxis].T)
			sampled_noise    = (np.random.rand(nb_settings, nb_param, nb_samples) < distances) * 1

			particles[:,:,:,0] = noisefree_update[:, :, :, 0] * (1 - sampled_noise) + sampled_noise * (1 + np.expand_dims(np.expand_dims(actions[:, t_idx - 1] != rewards[:, t_idx - 1], -1), -1))
			particles[:,:,:,1] = noisefree_update[:, :, :, 1] * (1 - sampled_noise) + sampled_noise * (1 + np.expand_dims(np.expand_dims(actions[:, t_idx - 1] == rewards[:, t_idx - 1], -1), -1))
			
		p_0_rewarding  = particles[:,:,:, 0]/(particles[:,:,:, 0] + particles[:,:,:, 1])
		p_1_rewarding  = 1 - p_0_rewarding

		log_p_act      = np.transpose(np.transpose(- np.logaddexp(0., (np.log(p_1_rewarding) - np.log(p_0_rewarding))/parameters[:,0][np.newaxis].T).T * (1 - actions[:,t_idx]) \
											- np.logaddexp(0., (np.log(p_0_rewarding) - np.log(p_1_rewarding))/parameters[:,0][np.newaxis].T).T * actions[:,t_idx]).T * (actions[:,t_idx] != -1))
		
		log_p_act        = np.logaddexp((log_p_act + np.log(1 - epsilons[np.newaxis].T)), np.log(epsilons[np.newaxis].T) - np.log(2))

		log_inc_marglkd += (logsumexp(log_p_act, axis=2) - np.log(nb_samples)) * (actions[:,t_idx][np.newaxis].T != -1)

		weights_norm     = np.transpose(np.exp(log_p_act.T - np.max(log_p_act,axis=2).T))
		weights_norm     = np.transpose(weights_norm.T/np.sum(weights_norm, axis=2).T)
		ancestors        = uf.stratified_resampling_tensor(weights_norm)

	return log_inc_marglkd




	
