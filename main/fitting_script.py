import pickle
from scipy.stats import truncnorm, multivariate_normal
path_to_models   = 'models/'
import sys
sys.path.append(path_to_models + 'utils/')
sys.path.append(path_to_models + 'lib_c/')
sys.path.append(path_to_models)
from fit_functions.noisy_beta import smc_parallel_JS_variance
import glob
import numpy as np
from scipy.stats import norm, beta

def bestBetaParams(sample):
  mu    = np.mean(sample)
  var   = np.var(sample) 
  alpha = ((1. - mu) / var - 1. / mu) * (mu**2)
  beta  = alpha * (1 / mu - 1.)
  return alpha, beta

def to_normalized_weights(logWeights):
    b = np.max(logWeights)
    weights = [np.exp(logw - b) for logw in logWeights]
    return weights/sum(weights)

if __name__=='__main__':

    path_to_results = 'results/marglkd/'
    nb_runs         = 3
    nb_sessions     = 2
    refine          = 0

    try:
        index = int(sys.argv[1]) - 1
    except:
        index = 22

    files           = glob.glob('data/python/td_volnoise_subj_*')
    nb_subj         = 25
    subject_indexes = np.array([np.all(['data/python/td_volnoise_subj_{2}_run_{0}_session_{1}.pkl'.format(i,j,k) in files for i in range(3) for j in range(2)]) * k for k in range(1,nb_subj + 1)])
    subject_indexes = subject_indexes[subject_indexes != 0]
    subject_indexes = np.append(12, subject_indexes) # add subject no12 which did 5 out of 6 blocks
    nb_subj         = len(subject_indexes)

    subj_idx             = subject_indexes[index]


    print('Processing subject {0}, model beta variance'.format(subj_idx))
    td_list = []
    for idx_session in range(nb_sessions):
        if subj_idx == 12 and idx_session == (nb_sessions - 1): # take care of nb12 (again)
            nb_runs = 2
        for idx_run in range(nb_runs):
            td_list.append(pickle.load(open('data/python/td_volnoise_subj_{0}_run_{1}_session_{2}.pkl'.format(subj_idx, idx_run, idx_session))))

    actions_ = np.array([td_list[i]['A_chosen'] for i in range(len(td_list))])
    rewards_ = np.array([td_list[i]['reward'] for i in range(len(td_list))])
    actions  = np.zeros([len(td_list), 180]) - 1
    rewards  = np.zeros([len(td_list), 180]) - 1
    for idx_td_list in range(len(td_list)):
        actions[idx_td_list, :len(actions_[idx_td_list])] = actions_[idx_td_list]
        rewards[idx_td_list, :len(rewards_[idx_td_list])] = rewards_[idx_td_list]


    ###### obtain posterior
    parameters          = np.concatenate((pickle.load(open('models/utils/sobol_1000_2.pkl')), np.zeros(1000)[:,np.newaxis], np.zeros(1000)[:,np.newaxis]), axis=-1)
    parameters[:,2]     = parameters[:,2]/2.
    
    log_inc_marglkd     = smc_parallel_JS_variance.smc(actions, rewards, parameters)

    pickle.dump([parameters, log_inc_marglkd], open(path_to_results + 'varianceJSbeta_subj{0}.pkl'.format(subj_idx), 'wb'))








