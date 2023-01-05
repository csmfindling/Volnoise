import numpy as np
from scipy.stats import gamma, norm
import matplotlib.pyplot as plt
import sys
sys.path.append("../useful_functions/")
import useful_functions as uf
import warnings

# td=td_list[0]; sample=theta; forced_actions = None;  beta_softmax=-1; apply_guided=0; apply_weber=1;apply_cst_noise=1; apply_inertie=1;nb_traj=1
def simulate_noisy_rl(td, rewards, sample, apply_cst_noise, apply_inertie, apply_weber, nb_traj=1):
	assert(nb_traj==1)
	T                  = rewards.shape[-1]
	noisy_trajectories = np.zeros([T, 2])
	actions_simul      = np.zeros(T) - 1
	prev_act		   = - 1
	alpha_c            = sample[0]
	alpha_u            = sample[0]
	beta_softmax       = sample[1]
	epsilon            = sample[2]
	rew_sim            = np.zeros(T)
	addednoise_levels  = np.zeros([T, 2])
	if apply_cst_noise and apply_inertie:
		eta     = sample[-2]
		inertie = sample[-1]
	elif apply_cst_noise:
		eta     = sample[-1]
		inertie = 0.
	elif apply_inertie:
		eta     = 0.
		inertie = sample[-1]
	else:
		eta     = 0.
		inertie = 0.
	for t_idx in range(T):
		if t_idx > 0.:
			prev_rew = rewards[:,t_idx - 1]
			addednoise0_anc = addednoise0;
			addednoise1_anc = addednoise1;
			addednoise_levels[t_idx - 1, 0] = addednoise0;
			addednoise_levels[t_idx - 1, 1] = addednoise1;
				
			if actions_simul[t_idx - 1] == 0:
				mu0 = (1 - alpha_c) * noisy_trajectories[t_idx - 1, 0] + alpha_c * prev_rew[0]
				mu1 = (1 - alpha_u) * noisy_trajectories[t_idx - 1, 1] + alpha_u * prev_rew[1]
			else:
				mu0 = (1 - alpha_u) * noisy_trajectories[t_idx - 1, 0] + alpha_u * prev_rew[0]
				mu1 = (1 - alpha_c) * noisy_trajectories[t_idx - 1, 1] + alpha_c * prev_rew[1]

			if apply_weber == 1:
				noise_level0 = np.log(1 + eta + np.abs(prev_rew[0] - noisy_trajectories[t_idx - 1, 0]) * epsilon)
				noise_level1 = np.log(1 + eta + np.abs(prev_rew[1] - noisy_trajectories[t_idx - 1, 1]) * epsilon)
			else:
				noise_level0 = epsilon
				noise_level1 = epsilon

			
			addednoise0 = inertie * addednoise0_anc + (1 - inertie) * noise_level0 * np.random.normal()
			addednoise1 = inertie * addednoise1_anc + (1 - inertie) * noise_level1 * np.random.normal()
			#print addednoise1;
			#print addednoise0;
			noisy_trajectories[t_idx, 0] = mu0 + addednoise0;
			noisy_trajectories[t_idx, 1] = mu1 + addednoise1;

			# probability to choose 1
			proba_1 = 1./(1. + np.exp(beta_softmax * (noisy_trajectories[t_idx, 0] - noisy_trajectories[t_idx, 1])))

			# simulate action
			if np.random.rand() < proba_1:
				actions_simul[t_idx] = 1
			else:
				actions_simul[t_idx] = 0

			rew_sim[t_idx] = rewards[actions_simul[t_idx].astype(int), t_idx]
			# prev action
			prev_act = actions_simul[t_idx]

		else:
			noisy_trajectories[t_idx]   = 0.5
			prev_act                    = -1
			addednoise0                 = .0;
			addednoise1                 = .0;
			if np.random.rand() < .5:
				actions_simul[t_idx] = 1
			else:
				actions_simul[t_idx] = 0

			rew_sim[t_idx] = rewards[actions_simul[t_idx].astype(int), t_idx]
			# previous action
			prev_act = actions_simul[t_idx]
		
	return td, addednoise_levels, noisy_trajectories, actions_simul, rew_sim

def simulate_noiseless_rl(rewards, idx_blocks, choices, forced_actions, sample, apply_rep_bias, apply_weber_decision_noise, nb_traj = 1):
	assert(apply_weber_decision_noise == 0), 'Simulation not developped for apply_weber_decision_noise = 1'
	warnings.warn('Repetition bias is not annuled when new block')

	T                      = len(choices)
	noiseless_trajectories = np.zeros([nb_traj, T, 2])
	actions_simul          = np.zeros([nb_traj, T])
	prev_act	    	   = np.zeros(nb_traj) - 1
	alpha_c                = sample[0]
	alpha_u                = sample[1]
	beta_softmax           = sample[2]
	if apply_rep_bias:
		rep_bias = sample[3]

	for t_idx in range(T):
		if idx_blocks[t_idx] == 0:
			prev_rew = rewards[:,t_idx - 1]
			for traj_idx in range(nb_traj):
				if choices[t_idx - 1] == 1:
					assert(prev_act[traj_idx] == actions_simul[traj_idx, t_idx - 1])
				
				if actions_simul[traj_idx, t_idx - 1] == 0:
					noiseless_trajectories[traj_idx, t_idx, 0] = (1 - alpha_c) * noiseless_trajectories[traj_idx, t_idx - 1, 0] + alpha_c * prev_rew[0]
					noiseless_trajectories[traj_idx, t_idx, 1] = (1 - alpha_u) * noiseless_trajectories[traj_idx, t_idx - 1, 1] + alpha_u * prev_rew[1]
				else:
					noiseless_trajectories[traj_idx, t_idx, 0] = (1 - alpha_u) * noiseless_trajectories[traj_idx, t_idx - 1, 0] + alpha_u * prev_rew[0]
					noiseless_trajectories[traj_idx, t_idx, 1] = (1 - alpha_c) * noiseless_trajectories[traj_idx, t_idx - 1, 1] + alpha_c * prev_rew[1]

		else:
			prev_act[:]                      = -1
			noiseless_trajectories[:, t_idx] = 0.5

		if choices[t_idx] == 1:
			for traj_idx in range(nb_traj):
				if (apply_rep_bias==0) or (prev_act[traj_idx]==-1):
					proba_1 = 1./(1. + np.exp(beta_softmax * (noiseless_trajectories[traj_idx, t_idx, 0] - noiseless_trajectories[traj_idx, t_idx, 1])))
				else:
					proba_1 = 1./(1. + np.exp(beta_softmax * (noiseless_trajectories[traj_idx, t_idx, 0] - noiseless_trajectories[traj_idx, t_idx, 1]) - np.sign(prev_act[traj_idx] - .5) * rep_bias))
				if np.random.rand() < proba_1:
					actions_simul[traj_idx, t_idx] = 1
				else:
					actions_simul[traj_idx, t_idx] = 0
				prev_act[traj_idx] = actions_simul[traj_idx, t_idx]
		else:
			actions_simul[:, t_idx] = forced_actions[t_idx]
	return noiseless_trajectories, actions_simul














