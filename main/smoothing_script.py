import pickle
from scipy.stats import truncnorm, multivariate_normal
path_to_models   = 'models/'
import sys
sys.path.append(path_to_models + 'utils/')
sys.path.append(path_to_models + 'lib_c/')
sys.path.append(path_to_models)
from fit_functions.noisy_beta import smoothing_JS_variance
import glob
import numpy as np


def smoothing_ances_traj(filtering_ances_traj):
    T, nb_samples = filtering_ances_traj.shape
    index         = np.zeros(filtering_ances_traj.shape)
    index[-1]     = filtering_ances_traj[-1]
    for i in np.arange(T-1)[::-1]:
        index[i] = filtering_ances_traj[i, np.array(index[i + 1], dtype=np.int)]
    return index


class ImportanceSampling():

    def __init__(self, td):
        self.td      = td
        self.actions = td['A_chosen']
        self.rewards = td['reward']

    def evaluateSample(self, Xin, nb_samples=2000):
        res                 = smoothing_JS_variance.smc(self.actions, self.rewards, Xin, nb_samples=nb_samples)
        smoothing_anc       = np.array(smoothing_ances_traj(res[-2]), dtype=np.int)
        idx_traj            = np.random.randint(smoothing_anc.shape[-1])
        sel_smoothing_anc   = np.array(smoothing_anc[:,idx_traj], dtype=np.int)                
        smoothing_part      = res[-4][np.arange(len(res[-4]), dtype=np.int), sel_smoothing_anc]
        smoothing_noisefree = res[-3][np.arange(len(res[-3]), dtype=np.int), sel_smoothing_anc]
        smoothing_dist      = res[-1][np.arange(len(res[-1]), dtype=np.int), sel_smoothing_anc]
        smoothing_res       = [smoothing_part, smoothing_noisefree, smoothing_dist]
        return res[0], smoothing_res
        

if __name__=='__main__':
    # variable to be set    
    path_to_results = 'results/smoothing/'
    nb_iterations   = 1000
    maps            = pickle.load(open('varianceJSbeta.pkl'))
    nb_runs         = 3
    nb_sessions     = 2

    try:
        index = int(sys.argv[1]) - 1
    except:
        index = 78

    files           = glob.glob('data/python/td_volnoise_subj_*')
    nb_subj         = 25
    subject_indexes = np.array([np.all(['data/python/td_volnoise_subj_{2}_run_{0}_session_{1}.pkl'.format(i,j,k) in files for i in range(3) for j in range(2)]) * k for k in range(1,nb_subj + 1)])
    subject_indexes = subject_indexes[subject_indexes != 0]
    nb_subj         = len(subject_indexes)    

    assert(index <  nb_subj * nb_sessions * nb_runs)
    i_subj      = index/(nb_sessions * nb_runs)
    i_          = index - i_subj * (nb_sessions * nb_runs)
    i_session   = i_/nb_runs
    i_run       = i_ - i_session * nb_runs

    subj_number    = subject_indexes[i_subj]

    print('Processing beta model smoothing trajectories for subject {0}, session {1}, run {2}'.format(subj_number, i_session, i_run))

    td              = pickle.load(open('data/python/td_volnoise_subj_{0}_run_{1}_session_{2}.pkl'.format(subj_number, i_run, i_session)))
    trajectories    = []
    logproba        = []
    get_marg_lkd    = ImportanceSampling(td)
    print(maps[subj_number])
    logmargl, res   = get_marg_lkd.evaluateSample(maps[subj_number], nb_samples=2000)
    print(logmargl)
    trajectories.append(res)
    logproba.append(logmargl)
    accep_ratio     = 0

    for idx_it in range(1, nb_iterations):
        logmarg_prop, traj_prop  = get_marg_lkd.evaluateSample(maps[subj_number], nb_samples=2000)
        log_alpha                = min(0, logmarg_prop - logmargl)
        if np.log(np.random.rand()) < log_alpha:
            trajectories.append(traj_prop)
            logproba.append(logmarg_prop)
            logmargl             = logmarg_prop
            accep_ratio         += 1
            if idx_it % 10 == 0:
                print('iteration number {0}'.format(idx_it))
                print('accep_ratio is {0}'.format(accep_ratio * 1./nb_iterations))
                print('\n')
        else:
            trajectories.append(trajectories[-1])
            logproba.append(logproba[-1])

    path = path_to_results + 'betamodel_traj_varianceJS_subj_{0}_run_{1}_session_{2}_nbit_{3}.pkl'.format(subj_number, i_run, i_session, nb_iterations)
    print('saved at ' + path)
    f = open(path, 'wb')
    pickle.dump([trajectories, logproba, accep_ratio * 1./nb_iterations], f)
    f.close()













