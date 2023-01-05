import pickle
from scipy.stats import truncnorm, multivariate_normal
path_to_models   = '../models/'
import sys
sys.path.append(path_to_models + 'utils/')
sys.path.append(path_to_models + 'lib_c/')
sys.path.append(path_to_models)
from fit_functions.noisy_beta import smc_parallel2_KL
import glob

if __name__=='__main__':

    path_to_results = '../../results/betamodel_marg_lkd/'
    nb_runs         = 3
    nb_sessions     = 2

    try:
        index = int(sys.argv[1])
    except:
        index = 22

    files           = glob.glob('../../data/python/td_volnoise_subj_*')
    nb_subj         = 25
    subject_indexes = np.array([np.all(['../../data/python/td_volnoise_subj_{2}_run_{0}_session_{1}.pkl'.format(i,j,k) in files for i in range(3) for j in range(2)]) * k for k in range(1,nb_subj + 1)])
    subject_indexes = subject_indexes[subject_indexes != 0]
    subject_indexes = np.append(12, subject_indexes) # add painful subject no12
    nb_subj         = len(subject_indexes)

    subj_idx             = subject_indexes[index]


    print('Processing subject {0}, model beta simple'.format(subj_idx))
    td_list = []
    for idx_session in range(nb_sessions):
        if subj_idx == 12 and idx_session == (nb_sessions - 1): # take care of nb12 (again)
            nb_runs = 2
        for idx_run in range(nb_runs):
            td_list.append(pickle.load(open('../../data/python/td_volnoise_subj_{0}_run_{1}_session_{2}.pkl'.format(subj_idx, idx_run, idx_session))))

    actions_ = np.array([td_list[i]['A_chosen'] for i in range(len(td_list))])
    rewards_ = np.array([td_list[i]['reward'] for i in range(len(td_list))])
    actions  = np.zeros([len(td_list), 180]) - 1
    rewards  = np.zeros([len(td_list), 180]) - 1
    for idx_td_list in range(len(td_list)):
        actions[idx_td_list, :len(actions_[idx_td_list])] = actions_[idx_td_list]
        rewards[idx_td_list, :len(rewards_[idx_td_list])] = rewards_[idx_td_list]

    ##### obtain posterior
    nb_iterations    = 100
    nb_param         = 3

    parameters        = pickle.load(open('../models/utils/sobol_10_3.pkl'))

    log_inc_marglkd  = smc_parallel2_KL.smc(actions, rewards, parameters)

    pickle.dump([parameters, log_inc_marglkd], open(path_to_results + 'raw_margLkd_estimation_subj{0}.pkl'.format(subj_idx), 'wb'))

    # ##### obtain marglkd

    # nb_iterations   = 1000
    # nb_param        = 3

    # mean_   = np.mean(parameters[:500], axis=0)
    # covar_  = np.dot((parameters[:500] - mean_).T, (parameters[:500] - mean_))/(len(parameters[:500]) - 1)

    # marg_lkd_list = np.zeros(nb_iterations)
    # correction    = np.zeros(nb_iterations)
    # all_samples   = np.zeros([nb_iterations, 3])

    # for idx_it in range(nb_iterations):
    #     sample              = np.random.multivariate_normal(mean_, covar_)
    #     lkd[idx_it]         = np.sum(smc_parallel_KL.smc(actions, rewards, sample))
    #     correction[idx_it]  = multivariate_normal.logpdf(sample, mean_, covar_)
    #     all_samples[idx_it] = sample

    # pickle.dump([all_samples, lkd, correction], open(path_to_results + 'marglkd_subj{0}.pkl'.format(subj_idx)))

    # #### obtain trajectories

    # # mean_ = np.mean(parameters[:500], axis=0)













