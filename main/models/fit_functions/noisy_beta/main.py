import pickle
from smc import smc
from scipy.stats import truncnorm

if __name__=='__main__':

    path_to_results = '../../results/marg_lkd/'
    nb_runs         = 3
    nb_sessions     = 2

    try:
        index = int(sys.argv[1])
    except:
        index = 97

    files           = glob.glob('../../data/python/td_volnoise_subj_*')
    nb_subj         = 25
    subject_indexes = np.array([np.all(['../../data/python/td_volnoise_subj_{2}_run_{0}_session_{1}.pkl'.format(i,j,k) in files for i in range(3) for j in range(2)]) * k for k in range(1,nb_subj + 1)])
    subject_indexes = subject_indexes[subject_indexes != 0]
    subject_indexes = np.append(12, subject_indexes) # add painful subject no12
    nb_subj         = len(subject_indexes)

    print('Processing subject {0}, model {1}'.format(subj_idx, cat_names[cat_optim]))
    td_list = []
    for idx_session in range(nb_sessions):
        if subj_idx == 12 and idx_session == (nb_sessions - 1): # take care of nb12 (again)
            nb_runs = 2
        for idx_run in range(nb_runs):
            td_list.append(pickle.load(open('../../data/python/td_volnoise_subj_{0}_run_{1}_session_{2}.pkl'.format(subj_idx, idx_run, idx_session))))



    td = pickle.load(open('../../../../data/python/td_volnoise_subj_10_run_0_session_0.pkl'));

    actions, rewards            = td['A_chosen'], td['reward']

    nb_iterations   = 100
    nb_param        = 3
    particles       = np.zeros([nb_iterations, nb_param])
    particles[0]    = np.random.rand(nb_param)
    log_inc_marglkd = smc(actions, rewards, particles[0])
    std_            = 0.2
    count           = 0.
    for idx_it in range(1, nb_iterations):
        if idx_it % 100 == 0:
            print 'iteration {0}'.format(idx_it)
            print 'acceptance ratio {0}'.format(count/idx_it)
            print '\n'

        a_prop, b_prop       = (0 - particles[idx_it - 1]) / std_, (1 - particles[idx_it - 1]) / std_
        candidate            = truncnorm.rvs(a_prop, b_prop, particles[idx_it - 1], std_)
        log_inc_marglkd_cand = smc(actions, rewards, candidate)
        a_anc, b_anc         = (0 - candidate) / std_, (1 - candidate) / std_
        logalpha             = log_inc_marglkd_cand - log_inc_marglkd \
                                        + np.sum(truncnorm.logpdf(candidate, a_prop, b_prop, particles[idx_it - 1], std_) \
                                                            - truncnorm.logpdf(particles[idx_it - 1], a_anc, b_anc, candidate, std_))

        if np.log(np.random.rand()) < logalpha:
            particles[idx_it] = candidate
            log_inc_marglkd   = log_inc_marglkd_cand
            count            += 1
        else:
            particles[idx_it] = particles[idx_it - 1]