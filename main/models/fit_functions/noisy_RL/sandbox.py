# from smc_object import smc_object
# #p = '../../data/exp1/noise_inference_subj2.pkl'
# p = '../../data/exp2/python/subj3_exp2.pkl'
# a = smc_object(path = p, exp_idx = 2, param = 4)
# a.do_inference(noise=1, show_progress=False)

import sys
sys.path.append("../")
from smc_object import smc_object
#p = '../../data/exp1/noise_inference_subj2.pkl'
p = '../../../data/exp3/python/subj1_complete.pkl'
a = smc_object(path = p, oneq_value = 1)
#a.do_inference(noise=0, show_progress=False)


for i in range(samples.shape[0]):
	print '\n'
	print logThetalkd[i]
	sample_p = samples[i]
	# print smc_c.smc_guided_2butt_c(state_candidates, state_candidates_a, weights_candidates, samples[i], ancestors_indexes_p, \
	# 																	idx_blocks, actions, rewards, choices, t_idx + 1)
	
	print smc_c.smc_2q_c(state_candidates, noises_descen_p, state_candidates_a, noises_ancest_p, weights_candidates, sample_p, ancestors_indexes_p, \
																		actions, rewards, t_idx + 1, apply_cst_noise, apply_inertie, apply_weber, 1, temperature)
	print sample_p
	# print smc_c.smc_2q_c(state_candidates, state_candidates_a, weights_candidates,  samples[i], ancestors_indexes_p, \
	# 																	idx_blocks, actions, rewards, choices, t_idx + 1, apply_rep, apply_weber, 1)


for i in range(samples.shape[0]):
	print '\n'
	print p_loglkd[i]
	[loglkd_prop, Q_prop, prev_action_prop] = get_loglikelihood(samples[i], rewards, actions, choices, idx_blocks, t_idx + 1) 
	print loglkd_prop


for i in range(samples.shape[0]):
	print '\n'
	print logThetalkd[i]
	print smc_c.smc_boostrap_4butt_c(state_candidates, state_candidates_a, weights_candidates, samples[i], ancestors_indexes_p, \
																		idx_blocks, actions, rewards, choices, t_idx + 1)



indexes = [str(logThetalkd[i]).startswith('-165.72383') for i in range(len(logThetalkd))]
i = np.where(indexes)[0][0]


sample = np.array([0.37516301, 1.65858557, 0.1583635])
print smc_c.smc_2q_c(state_candidates, state_candidates_a, weights_candidates,  sample, ancestors_indexes_p, \
																		idx_blocks, actions, rewards, choices, t_idx + 1, apply_rep, apply_weber, 1)