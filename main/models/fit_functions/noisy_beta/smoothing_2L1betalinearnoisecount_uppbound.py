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
    
    return du * (confirmed * 2. * par_confirmed + disconfir * 10. * par_disconfir)

# td = pickle.load(open('../../../../data/python/td_volnoise_subj_10_run_0_session_0.pkl')); actions = td['A_chosen'][np.newaxis]; rewards=td['reward'][np.newaxis]; params = [.2, 0.6, .02]; show_progress=True; numberOfStateSamples=200; latin_hyp_sampling =True;  epsilon_softmax=0.; noise_inertie = 0.; numberOfThetaSamples=200; coefficient = .5; beta_softmax=1.; lambda_noise=1.; eta_noise = .05; numberOfBetaSamples=20 ;alpha=0.; 
def smc(actions, rewards, parameters, nb_samples = 1000):

    T                 = actions.shape[0]    
    particles         = np.ones([nb_samples])
    ancestors         = np.zeros([nb_samples], dtype = np.int)
    temp              = parameters[0]
    log_inc_marglkd   = 0 
    anc_particles     = np.ones(nb_samples)
    noisefree_update  = np.ones([nb_samples])
    distances         = np.zeros([nb_samples])
    average_noise     = np.zeros([nb_samples])
    means_beta        = np.zeros([nb_samples])


    all_particles        = np.ones([T, nb_samples])
    all_noisefree_update = np.ones([T, nb_samples])
    all_ancestors        = np.ones([T, nb_samples])
    all_distances        = np.ones([T, nb_samples])


    for t_idx in range(T):
        
        if t_idx > 0:
            anc_particles[:]       = particles[ancestors]            
            noisefree_update[:]    = np.transpose(anc_particles.T + 1 * (actions[t_idx - 1] != rewards[t_idx - 1]) - 1 * (actions[t_idx - 1] == rewards[t_idx - 1]))

            distances[:]     = L1(anc_particles, noisefree_update, parameters[1][np.newaxis].T, parameters[3][np.newaxis].T) + parameters[2][np.newaxis].T * np.abs(noisefree_update)
            #print(distances.mean())
            sampled_noise    = np.random.beta(1., np.minimum(distances, 1)) 
            particles        = noisefree_update * sampled_noise                                    

        if actions[t_idx] == 0:
            log_p_act  = - np.logaddexp(0., -particles/temp)
        else:
            log_p_act  = - np.logaddexp(0., particles/temp)

        log_inc_marglkd += logsumexp(log_p_act) - np.log(nb_samples)
        ancestors[:]     = uf.stratified_resampling(uf.to_normalized_weights(log_p_act))

        # save particles
        all_particles[t_idx]        = particles
        all_noisefree_update[t_idx] = noisefree_update
        all_ancestors[t_idx]        = ancestors
        all_distances[t_idx]        = distances

    return log_inc_marglkd, all_particles, all_noisefree_update, all_ancestors, all_distances




    
