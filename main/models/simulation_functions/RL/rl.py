import numpy as np
from scipy.stats import gamma, norm
import matplotlib.pyplot as plt
import sys
sys.path.append("../useful_functions/")
import useful_functions as uf
import warnings

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.expand_dims(np.max(x, axis=2), axis=-1))
    return e_x / np.expand_dims(e_x.sum(axis=-1), axis=-1)

# td=td_list[0]; sample=theta; forced_actions = None;  beta_softmax=-1; apply_guided=0; apply_weber=1;apply_cst_noise=1; apply_inertie=1;nb_traj=1
def simulate_noiseless_rl(td, learningrates, softmaxs, mapping):
    T                  = len(td['Z'])
    nb_learningrates   = len(learningrates)
    nb_softmaxs        = len(softmaxs)
    numberOfStim       = td['state_num']
    numberOfActions    = td['action_num']
    trajectories       = np.zeros([nb_softmaxs, nb_learningrates, numberOfStim, numberOfActions])
    actions_simul      = np.zeros([nb_softmaxs, nb_learningrates, T], dtype=np.int) - 1
    performance        = np.zeros([nb_softmaxs, nb_learningrates, T])
    for t_idx in range(T):
        if t_idx > 0.:           

            # probability to choose 1
            x       = np.multiply(softmaxs[:,np.newaxis, np.newaxis],  trajectories[:,:,td['S'][t_idx]])
            proba_1 = softmax(x)
            actions = np.sum(1 - (np.random.rand(nb_softmaxs, nb_learningrates)[:,:,np.newaxis] < np.cumsum(proba_1, axis=-1)), axis=-1)
            actions_simul[: ,: ,t_idx] = actions

        else:
            trajectories[:, :]          = 1./numberOfActions
            actions_simul[: ,: ,t_idx]  = np.random.randint(numberOfActions, size=(nb_softmaxs, nb_learningrates))

        if td['rewardObserved'][t_idx]:
            reward = (not td['trap'][t_idx]) * (actions_simul[: ,: ,t_idx] == mapping[td['S'][t_idx].astype(int), td['Z'][t_idx].astype(int)]) + \
                      td['trap'][t_idx] * (actions_simul[: ,: ,t_idx] != mapping[td['S'][t_idx].astype(int), td['Z'][t_idx].astype(int)])
            trajectories[np.tile(np.arange(nb_softmaxs), (nb_learningrates, 1)), np.tile(np.arange(nb_learningrates), (nb_softmaxs, 1)).T,td['S'][t_idx], actions_simul[:,:,t_idx].T] = \
                    np.transpose(learningrates * reward) + np.expand_dims(1 - learningrates, -1) * trajectories[np.tile(np.arange(nb_softmaxs), (nb_learningrates, 1)), np.tile(np.arange(nb_learningrates), (nb_softmaxs, 1)).T,td['S'][t_idx], actions_simul[:,:,t_idx].T]


        performance[:,:,t_idx:] += np.expand_dims((actions_simul[:,:,t_idx] == mapping[td['S'][t_idx].astype(int), td['Z'][t_idx].astype(int)]), -1)
        
    return performance






















