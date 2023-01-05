# import numpy as np
# from scipy.stats import gamma, norm, truncnorm, multivariate_normal, beta
# import matplotlib.pyplot as plt
# from numpy.random import multivariate_normal as multi_norm
# from scipy.misc import logsumexp
# import sys
# sys.path.append("../../utils/")
# import useful_functions as uf
# import pickle
# from scipy.special import beta as beta_func

# # td = pickle.load(open('../../../../data/python/td_volnoise_subj_10_run_0_session_0.pkl')); params = [1, 1, 1]; show_progress=True; numberOfStateSamples=200; latin_hyp_sampling =True;  epsilon_softmax=0.; noise_inertie = 0.; numberOfThetaSamples=200; coefficient = .5; beta_softmax=1.; lambda_noise=1.; eta_noise = .05; numberOfBetaSamples=20 ;alpha=0.; 
# def smc2(actions, rewards, idx_block):

#     actions            = np.asarray(td['A_chosen'], dtype=np.intc)
#     rewards            = np.ascontiguousarray(td['reward'])
#     nb_z, nb_theta     = 100, 100
#     T, dx, N_dx        = actions.shape[0], 0.001, 1000
#     x_range            = np.tile(np.arange(0,1,dx), (nb_z * nb_theta, 1))
#     parameters         = np.random.rand(nb_theta, 3)
#     particles          = np.ones([nb_theta * nb_z, 2])
#     ancestors          = np.zeros([nb_theta * nb_z], dtype = np.int)
#     anc_particles      = np.ones([nb_theta * nb_z, 2])
#     noisefree_update   = np.ones([nb_theta * nb_z, 2])
#     distances          = np.zeros([nb_theta * nb_z])
#     average_noise      = np.zeros([nb_theta * nb_z])
#     weights_z          = np.zeros([nb_theta * nb_z])
#     sum_weights_z      = np.zeros([nb_theta])
#     log_marglkd        = 0
#     log_weights_theta  = np.zeros([nb_theta])
#     log_lkd_theta      = np.zeros([nb_theta])
#     weights_theta_norm = np.zeros([nb_theta])
#     log_marglkd        = 0
#     ess_list           = []

#     # move particles
#     log_lkd_theta_new  = np.zeros([nb_theta])
#     particles_new      = np.zeros([nb_theta, nb_z])
#     log_p_act_new      = np.zeros([nb_theta, nb_z])

#     for t_idx in range(T):
        
#         if t_idx > 0:
#             max_log_p_act               = np.max(log_p_act, axis=1)
#             p_act                       = np.transpose(np.exp(log_p_act.T - max_log_p_act))
#             p_act                       = np.transpose(p_act.T/np.sum(p_act, axis=1))
#             ancestors[:]                = np.ravel(uf.stratified_resampling_matrix(p_act))

#             anc_particles[:]       = particles[ancestors]
#             noisefree_update[:, 0] = anc_particles[:, 0] + (actions[t_idx - 1] != rewards[t_idx - 1])
#             noisefree_update[:, 1] = anc_particles[:, 1] + (actions[t_idx - 1] == rewards[t_idx - 1])
            
            
#             distances[:] = np.sum(np.transpose(np.abs( \
#                                             np.power(x_range.T, noisefree_update[:,0] - 1) \
#                                                     * np.power((1 - x_range).T, noisefree_update[:,1] - 1) \
#                                                             / beta_func(noisefree_update[:,0], noisefree_update[:,1]) * dx \
#                                             - np.power(x_range.T, anc_particles[:,0] - 1) \
#                                                     * np.power((1 - x_range).T, anc_particles[:,1] - 1) \
#                                                             / beta_func(anc_particles[:,0], anc_particles[:,1])  * dx \
#                                           ) \
#                                     ) \
#                               , axis = -1)

#             average_noise[:] = np.ravel(np.transpose(1 + parameters[:,2] + parameters[:,1] * np.reshape(distances, (nb_theta, nb_z)).T))
#             particles[:]     = np.maximum(1, np.transpose(np.random.rand(particles.shape[0], particles.shape[1]).T * (1 - 1/average_noise) + 1/average_noise) * noisefree_update)

#         if idx_block[t_idx] == 1:
#             particles[:] = np.ones([nb_theta * nb_z, 2])

#         p_0_rewarding  = np.reshape(beta_func(particles[:,0] + 1, particles[:,1])/beta_func(particles[:,0], particles[:,1]), (nb_theta, nb_z))
#         p_1_rewarding  = 1 - p_0_rewarding

#         if actions[t_idx] == 0:
#             log_p_act  = - np.reshape(np.transpose(np.logaddexp(0., (np.log(p_1_rewarding) - np.log(p_0_rewarding)).T/parameters[:,0])), (nb_theta, nb_z))
#         else:
#             log_p_act  = - np.reshape(np.transpose(np.logaddexp(0., (np.log(p_0_rewarding) - np.log(p_1_rewarding)).T/parameters[:,0])), (nb_theta, nb_z))

#         log_inc_marglkd       = logsumexp(log_p_act, axis=1) - np.log(nb_z)
#         log_marglkd          += logsumexp(log_weights_theta + log_inc_marglkd) - logsumexp(log_weights_theta)
#         log_weights_theta    += log_inc_marglkd
#         log_lkd_theta        += log_inc_marglkd
#         weights_theta_norm[:] = uf.to_normalized_weights(log_weights_theta)

#         ess_list.append(1./sum(weights_theta_norm**2))
#         if 1./sum(weights_theta_norm**2) < .0001 * nb_theta:
#             means_    = np.tile(np.sum(weights_theta_norm * parameters.T, axis=1), (nb_theta, 1))
#             std_      = np.tile(np.sqrt(np.diag(np.dot((parameters - means_).T * weights_theta_norm, (parameters - means_)))), (nb_theta, 1))

#             a_prop, b_prop    = (0 - means_) / std_, (1 - means_) / std_
#             proposed_thetas   = truncnorm.rvs(a_prop, b_prop, loc=means_, scale=std_)

#             particles_move, log_p_act_move, log_lkd_theta_move = back_smc(actions, rewards, proposed_thetas, t_idx + 1, nb_theta, nb_z, x_range, dx)

#             anc_theta_samples = uf.stratified_resampling(weights_theta_norm)
#             alphas_ratio      = log_lkd_theta_move - log_lkd_theta[anc_theta_samples] \
#                                     + np.sum(norm.logpdf(proposed_thetas, means_, std_), axis=1) - np.sum(norm.logpdf(parameters[anc_theta_samples], means_, std_), axis=1)

#             indexes_acc       = np.log(np.random.rand(nb_theta)) < np.minimum(alphas_ratio, 0)

#             # udpate particles
#             log_lkd_theta_new[indexes_acc]  = log_lkd_theta_move[indexes_acc]
#             log_lkd_theta_new[~indexes_acc] = log_lkd_theta[anc_theta_samples][~indexes_acc]

#             particles_new[indexes_acc]      = np.reshape(particles_move, (nb_theta, nb_z, 2))[indexes_acc]
#             particles_new[~indexes_acc]     = np.reshape(particles, (nb_theta, nb_z, 2))[anc_theta_samples][~indexes_acc]

#             log_p_act_new[indexes_acc]      = log_p_act_move[indexes_acc]
#             log_p_act_new[~indexes_acc]     = log_p_act[anc_theta_samples][~indexes_acc]

#             log_lkd_theta[:]                = log_lkd_theta_new
#             particles[:]                    = np.reshape(particles_new, (nb_theta * nb_z, 2))
#             log_p_act[:]                    = log_p_act_new
#             log_weights_theta[:]            = 0.

#             print('acceptance ratio is {0}'.format(np.mean(indexes_acc)))

#     return particles, log_marglkd 



# def back_smc(actions, rewards, proposed_thetas, T, nb_theta, nb_z, x_range, dx):

#     ancestors_move        = np.zeros([nb_theta * nb_z], dtype = np.int)
#     anc_particles_move    = np.zeros([nb_theta * nb_z, 2])
#     noisefree_update_move = np.zeros([nb_theta * nb_z, 2])
#     particles_move        = np.zeros([nb_theta * nb_z, 2]) + 1
#     distances_move        = np.zeros([nb_theta * nb_z]) 
#     log_inc_marglkd_move  = np.zeros([nb_theta])
#     log_marglkd_move      = 0
#     log_lkd_theta_move    = np.zeros([nb_theta])
#     average_noise_move    = np.zeros([nb_theta * nb_z]) 
#     distances_move        = np.zeros([nb_theta * nb_z]) 

#     for t_idx in range(T):
        
#         if t_idx > 0:

#             max_log_p_act_move          = np.max(log_p_act_move, axis=1)
#             p_act_move                  = np.transpose(np.exp(log_p_act_move.T - max_log_p_act_move))
#             p_act_move                  = np.transpose(p_act_move.T/np.sum(p_act_move, axis=1))
#             ancestors_move[:]           = np.ravel(uf.stratified_resampling_matrix(p_act_move))
#             anc_particles_move[:]       = particles_move[ancestors_move]
#             noisefree_update_move[:, 0] = anc_particles_move[:, 0] + (actions[t_idx - 1] != rewards[t_idx - 1])
#             noisefree_update_move[:, 1] = anc_particles_move[:, 1] + (actions[t_idx - 1] == rewards[t_idx - 1])
            
            
#             distances_move[:] = np.sum(np.transpose(np.abs( \
#                                             np.power(x_range.T, noisefree_update_move[:,0] - 1) \
#                                                     * np.power((1 - x_range).T, noisefree_update_move[:,1] - 1) \
#                                                             / beta_func(noisefree_update_move[:,0], noisefree_update_move[:,1]) * dx \
#                                             - np.power(x_range.T, anc_particles_move[:,0] - 1) \
#                                                     * np.power((1 - x_range).T, anc_particles_move[:,1] - 1) \
#                                                             / beta_func(anc_particles_move[:,0], anc_particles_move[:,1])  * dx \
#                                           ) \
#                                     ) \
#                               , axis = -1)

#             average_noise_move[:] = np.ravel(np.transpose(1 + proposed_thetas[:,2] + proposed_thetas[:,1] * np.reshape(distances_move, (nb_theta, nb_z)).T))
#             particles_move[:]     = np.maximum(1, np.transpose(np.random.rand(particles_move.shape[0], particles_move.shape[1]).T * (1 - 1/average_noise_move) + 1/average_noise_move) * noisefree_update_move)

#         # if idx_block[t_idx] == 1:
#         #     particles_move[:] = np.ones([nb_theta * nb_z, 2])

#         p_0_rewarding_move  = np.reshape(beta_func(particles_move[:,0] + 1, particles_move[:,1])/beta_func(particles_move[:,0], particles_move[:,1]), (nb_theta, nb_z))
#         p_1_rewarding_move  = 1 - p_0_rewarding_move

#         if actions[t_idx] == 0:
#             log_p_act_move  = - np.reshape(np.transpose(np.logaddexp(0., (np.log(p_1_rewarding_move) - np.log(p_0_rewarding_move)).T/proposed_thetas[:,0])), (nb_theta, nb_z))
#         else:
#             log_p_act_move  = - np.reshape(np.transpose(np.logaddexp(0., (np.log(p_0_rewarding_move) - np.log(p_1_rewarding_move)).T/proposed_thetas[:,0])), (nb_theta, nb_z))

#         log_lkd_theta_move   += logsumexp(log_p_act_move, axis=1) - np.log(nb_z)

#     return particles_move, log_p_act_move, log_lkd_theta_move





    
