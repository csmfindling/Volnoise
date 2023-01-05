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


def L1(count1, count2, par_confirmed, par_disconfir):
	du        = np.abs( ( count2 - count1 ) / ( np.abs(count1) + 1 ) )
	confirmed = (count1 >= 0) * (count2 > count1) + (count1 <= 0) * (count2 < count1)
	disconfir = (count1 > 0 ) * (count2 < count1) + (count1 < 0 ) * (count2 > count1)	
	
	return du * (confirmed * 2. * par_confirmed + disconfir * 10.* par_disconfir)

def smc(actions, rewards, parameters, nb_samples):

	nb_settings, T    = actions.shape[:2]
	nb_param, nb_dims = parameters.shape
	particles         = np.zeros([nb_settings, nb_param, nb_samples])
	ancestors         = np.zeros([nb_settings, nb_param, nb_samples], dtype = np.int)
	mus               = parameters[:,3]
	epsilons          = parameters[:,2]
	log_inc_marglkd   = 0 
	anc_particles     = np.zeros([nb_settings, nb_param, nb_samples])
	noisefree_update  = np.zeros([nb_settings, nb_param, nb_samples])
	distances         = np.zeros([nb_settings, nb_param, nb_samples])
	list_llm          = np.zeros([T])


	for t_idx in range(T):
		# print(t_idx)
		
		if t_idx > 0:
			anc_particles[:]    = np.reshape(np.array([particles[i, j, ancestors[i, j]] for i in range(nb_settings) for j in range(nb_param)]), (nb_settings, nb_param, nb_samples))
			noisefree_update[:] = np.transpose(anc_particles.T + 1 * (actions[:, t_idx - 1] != rewards[:, t_idx - 1]) - 1 * (actions[:, t_idx - 1] == rewards[:, t_idx - 1]))
					
			distances[:]     = L1(anc_particles, noisefree_update, parameters[:,1][np.newaxis].T, parameters[:,3][np.newaxis].T)
			#print(distances.mean())
			sampled_noise    = np.random.beta(parameters[:,2][np.newaxis].T * 5., distances) 
			particles        = noisefree_update * sampled_noise

		log_p_act      = np.transpose(np.transpose(- np.logaddexp(0., -particles/parameters[:,0][np.newaxis].T).T * (1 - actions[:,t_idx]) \
											- np.logaddexp(0., particles/parameters[:,0][np.newaxis].T).T * actions[:,t_idx]).T * (actions[:,t_idx] != -1))
		
		#log_p_act        = np.logaddexp((log_p_act + np.log(1 - epsilons[np.newaxis].T)), np.log(epsilons[np.newaxis].T) - np.log(2))

		log_inc_marglkd += (logsumexp(log_p_act, axis=2) - np.log(nb_samples)) * (actions[:,t_idx][np.newaxis].T != -1)

		weights_norm     = np.transpose(np.exp(log_p_act.T - np.max(log_p_act,axis=2).T))
		weights_norm     = np.transpose(weights_norm.T/np.sum(weights_norm, axis=2).T)
		ancestors        = uf.stratified_resampling_tensor(weights_norm)

	return log_inc_marglkd




