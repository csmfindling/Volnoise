##### Useful functions used in the script #####

# Libraries
import numpy as np
import math
import operator
import functools
from scipy.stats import beta as betalib
from scipy.special import gammaln
from scipy.stats import norm
from scipy.stats import moment
from scipy.stats import invgamma
import warnings


# Return dictionnary of addresses for each variable; takes an input a list of variables
def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]

def return_all_addresses(*args):
    warnings.warn("Careful in the order of the variables for the return_addresses function")
    assert(len(args) == 19)
    dictionary = {}
    name_variables = ['logThetaLks', 'logThetaWeights', 'betaSamples', 'etaSamples', 'currentTaskSetSamples', 'currentLatentSamples' , 'currentTemperatureSamples' , \
                        'ancestorLatentSamples', 'ancestorTemperatureSamples', 'logLatentWeights', 'latentWeights' , 'candidateLatentSamples', 'candidateTemperatureSamples', 'candidateLogWeights', 'candidateWeights', \
                        'mapping', 'stimuli' , 'rewards' , 'actions']
    for i in range(len(args)):
        dictionary[name_variables[i]] = hex(id(args[i]))
    return dictionary

def return_address_var(arg):
    return hex(id(arg))

def stratified_resampling(logThetaLks, logThetaWeights, betaSamples, etaSamples, currentTaskSetSamples, currentLatentSamples , currentTemperatureSamples , ancestorLatentSamples, \
                                            ancestorTemperatureSamples, latentWeights , candidateLatentSamples, candidateTemperatureSamples, candidateWeights, mapping, \
                                            stimuli , rewards , actions):
    N = len(w)
    v = np.cumsum(w) * N
    s = np.random.uniform()
    o = np.zeros(N, dtype=np.int)
    m = 0
    for i in range(N):
        while v[m] < s : m = m + 1
        o[i] = m; s = s + 1
    return o


# Stratified resample:
def stratified_resampling(w):
    N = len(w)
    v = np.cumsum(w) * N
    s = np.random.uniform()
    o = np.zeros(N, dtype=np.int)
    m = 0
    for i in range(N):
        while v[m] < s : m = m + 1
        o[i] = m; s = s + 1
    return o

# Stratified resample:
def stratified_resampling_matrix(w_matrix):
    N = len(w_matrix[0])
    v = np.cumsum(w_matrix, axis=1) * N
    s = np.random.uniform(size=len(w_matrix))
    o = np.zeros([len(w_matrix), N], dtype=np.int)
    m = np.zeros(len(w_matrix), dtype=np.int)
    for idx in range(N):
        v_tmp = np.array([v[i, m[i]] for i in range(len(v))])
        while np.any(v_tmp < s) : 
            x     = np.where(v_tmp < s)[0]
            m[x]  = m[x] + 1
            v_tmp = np.array([v[i, m[i]] for i in range(len(v))])
        o[:,idx] = m; s = s + 1
    return o

# tensor
def stratified_resampling_tensor(w_tensor):
    N_setting, n_param, N_z = w_tensor.shape
    v = np.cumsum(w_tensor, axis=2) * N_z
    s = np.random.uniform(size=(N_setting, n_param))
    o = np.zeros([N_setting, n_param, N_z], dtype=np.int)
    m = np.zeros([N_setting, n_param], dtype=np.int)
    for idx in range(N_z):
        v_tmp   = np.reshape(np.array([v[i, j, m[i,j]] for i in range(N_setting) for j in range(n_param)]), (N_setting, n_param))
        while np.any(v_tmp < s) : 
            x,y     = np.where(v_tmp < s)
            m[x,y]  = m[x,y] + 1
            v_tmp   = np.reshape(np.array([v[i, j, m[i,j]] for i in range(N_setting) for j in range(n_param)]), (N_setting, n_param)) #v_tmp   = np.reshape(np.array([v[i, j, m[i,j]] for i in range(N_setting) for j in range(n_param)]), (N_setting, n_param))
        o[:,:,idx] = m; s = s + 1
    return o


# Multinomial un-normalized pick function; the un-normalized probability distribution is p.
def random_pick(p):
	return np.random.choice(len(p), p = np.divide(p,np.sum(p)))

def random_pick_list(p,n):
    return np.random.choice(len(p), size = n, p = np.divide(p,np.sum(p)))

def truncated_normal(mu, std, mini, numberOfSamples):
    res = np.zeros(numberOfSamples)
    for i in range(numberOfSamples):
        sample = np.random.normal(mu,std)
        while (sample < mini):
            sample = np.random.normal(mu,std)
        res[i] = sample
    return res

def sample_inv_gamma(a, b, size=1):
    sample = invgamma.rvs(a=a, size=size)
    return b*sample

def ppf_inv_gamma(x, a, b):
    return b * invgamma.ppf(x, a)
    # b/gammainccinv(a, r[10])

def random_nRW(m, var, mini, maxi):
    std   = np.sqrt(var) 
    s     = np.random.normal(m, std)
    s     = s *(s > mini)*(s < maxi) + (s > maxi)
    return s

def random_nRW_vec(m, var, mini, maxi, nbOfSamples):
    std   = np.sqrt(var)
    s     = np.random.normal(m,std, size = nbOfSamples)
    s     = s *(s > mini)*(s < maxi) + (s > maxi)
    return s

def estimate_param_truncated_normal(data, mini):
    moment_1 = np.mean(data)
    moment_2 = np.mean(data**2)
    moment_3 = np.mean(data**3)
    mu       = mini + (2 * moment_1 * moment_2 - moment_3)/(2 * moment_1**2 - moment_2)
    sigma2   = (moment_1 * moment_3 - moment_2**2)/(2 * moment_1**2 - moment_2)
    return mu, np.sqrt(sigma2)

# def numb_switches_to_task(k, state_seq):
# 	switch_seq = np.concatenate([np.array([True]),~np.equal(state_seq[1:],state_seq[:-1])]);
# 	return np.sum(np.multiply((state_seq==k),(switch_seq==1)))

#def numb_switches_to_task_vect(array_k, state_seq):
#	switch_to_task_vect_func = np.vectorize(numb_switches_to_task, excluded=['state_seq'])
#	return(switch_to_task_vect_func(k=array_k,state_seq=state_seq))

# def metropolis_hasting_gamma(target_function, gamma_parameters, previous_value, nb_it):
#     value = previous_value;
#     alpha_mean = 0;
#     for i in range(nb_it):
#         candidate = np.random.dirichlet(gamma_parameters);
#         alpha     = np.minimum(0, np.log(target_function(candidate)) \
#                                + np.log(dirichlet_pdf(value, gamma_parameters)) \
#                                - np.log(target_function(value)) \
#                                - np.log(dirichlet_pdf(candidate, gamma_parameters)))
#         assert(not math.isnan(alpha))
#         alpha_mean = np.divide(alpha_mean*i,(i+1)) + np.divide(np.exp(alpha),(i+1))
#         if np.random.rand(1) < np.exp(alpha): value = candidate;
#     return [value, alpha_mean]

# def metropolis_hasting_tau(target_function, tau_parameters, previous_value, nb_it):
# 	value = previous_value;
# 	alpha_mean = 0;
# 	for i in range(nb_it):
# 		candidate = np.random.beta(tau_parameters[0], tau_parameters[1]);
# 		alpha     = np.minimum(0, np.log(target_function(candidate)) \
# 									+ np.log(betalib.pdf(value, tau_parameters[0], tau_parameters[1]))\
# 									- np.log(target_function(value))\
# 									- np.log(betalib.pdf(candidate, tau_parameters[0], tau_parameters[1])))
# 		assert(not math.isnan(alpha))
# 		alpha_mean = np.divide(alpha_mean*i,(i+1)) + np.divide(np.exp(alpha),(i+1))
# 		if np.random.rand(1) < np.exp(alpha): value = candidate;
# 	return [value, alpha_mean]

def dirichlet_pdf(x, alpha):
  return (math.gamma(sum(alpha)) / 
          functools.reduce(operator.mul, [math.gamma(a) for a in alpha]) *
          functools.reduce(operator.mul, [x[i]**(alpha[i]-1.0) for i in range(len(alpha))]))

def log_dirichlet_pdf(x,alpha):
    return (gammaln(sum(alpha)) - functools.reduce(operator.add, [gammaln(a) for a in alpha]) + functools.reduce(operator.add, [(alpha[i] - 1)*np.log(x[i]) for i in range(len(alpha))]))

def log_beta_pdf(x, a, b):
    return gammaln(a + b) - gammaln(a) - gammaln(b) + (a - 1)*np.log(x) + (b - 1)*np.log(1 - x)

def log_truncated_normal_pdf(x, mu, std, mini):
    assert(mini == 0)
    return np.log(norm.pdf((x-mu)/std)) - np.log(1 - norm.cdf(-mu/std)) - np.log(std)

def log_invgamma_pdf(x, a, b):
    return a * np.log(b) - gammaln(a) - (a + 1)*np.log(x) - b/x

def log_sum(logvector):
    b = np.max(logvector)
    return b + np.log(functools.reduce(operator.add, [np.exp(logw - b) for logw in logvector]))

def to_normalized_weights(logWeights):
    b = np.max(logWeights)
    weights = [np.exp(logw - b) for logw in logWeights]
    return weights/sum(weights)

# def MLE_beta_fit(data):
#     mean = np.mean(data);
#     std  = np.std(data);
#     a    = np.multiply(np.divide(1-mean, std**2) - np.divide(1,mean), mean**2)
#     b    = np.multiply(a, np.divide(1,mean) - 1)
#     assert(not math.isnan(a)); assert(not math.isnan(b));
#     return np.array([a,b])

def autocorrelation(x):
    x_norm = np.divide(x - np.mean(x),np.std(x))
    result = np.correlate(np.divide(x_norm, len(x)), x_norm, mode='full')
    return result[result.size/2:]

def plot_results(td, Z_prob, last_Z_prob, tau_params, beta_params, gamma_params, m_h_tau, m_h_gamma, A_corr_count, tau_autocorrelation):
    import matplotlib.pyplot as plt

    [trial_num, K] = gamma_params.shape; Z_true = td['Z'];
    sample_num     = len(tau_autocorrelation);
    plt.figure(figsize=(12, 9));

    # Plot beta
    plt.subplot(2,2,1);
    beta_mean = np.divide(beta_params[:,0], np.sum(beta_params,axis=1));
    beta_std  = np.sqrt(np.divide(np.multiply(beta_mean, beta_params[:,1]), np.multiply(np.sum(beta_params, axis=1), np.sum(beta_params, axis=1)+1)))
    plt.plot(beta_mean, 'r-');plt.hold(True); plt.fill_between(np.arange(trial_num),beta_mean-beta_std, beta_mean+beta_std,facecolor=[1,.5,.5], color=[1,.5,.5]); 
    # Mark switch and trap trials
    plt.axis([0,trial_num-1, 0, 1 ]);
    switch_trials = np.where(td['B'])[0];
    #for switch in switch_trials:
    #    plt.plot([switch, switch], plt.gca().get_ylim(), 'g', linewidth=1);
    #trap_trials = np.where(td['trap'])[0]
    #for trap in trap_trials:
    #    plt.plot([trap, trap], plt.gca().get_ylim(), 'm', linewidth=1);
    plt.plot([0, trial_num], [td['beta'], td['beta']], 'r--', linewidth=2);
    plt.hold(False);
    plt.ylabel('Estimated beta parameters'); 

    # Plot tau
    plt.subplot(2,2,3);
    tau_mean = np.divide(tau_params[:,0], np.sum(tau_params, axis=1));
    tau_std  = np.sqrt(np.divide(np.multiply(tau_mean, tau_params[:,1]), np.multiply(np.sum(tau_params, axis=1), np.sum(tau_params, axis=1)+1)));
    plt.plot(tau_mean, 'b-');plt.hold(True); plt.fill_between(np.arange(trial_num), tau_mean - tau_std, tau_mean+tau_std, facecolor=[.5,.5,1],color = [.5,.5,1]); 
    # Mark switch and trap trials
    plt.axis([0, trial_num-1, 0, 1]);
    #for switch in switch_trials:
    #    plt.plot([switch, switch], plt.gca().get_ylim(), 'g', linewidth=1);
    #for trap in trap_trials:
    #    plt.plot([trap, trap], plt.gca().get_ylim(), 'm', linewidth=1);
    plt.plot([0, trial_num], [td['tau'], td['tau']], 'b--', linewidth=2);
    plt.hold(False);
    plt.ylabel('Estimated tau paramaters'); 

    # Plot gamma paramaters
    plt.subplot(2,2,4);
    plt.imshow(gamma_params.T); plt.hold(True);
    #for switch in switch_trials:
    #    plt.plot([switch, switch], plt.gca().get_ylim(), 'g', linewidth=1);
    plt.plot(Z_true, 'k--', linewidth=1);
    plt.axis([0,trial_num-1, 0, K-1]);
    plt.xlabel('trials');
    plt.hold(False);
    plt.ylabel('Estimated gamma parameters');

    # Plot state probability
    plt.subplot(2,2,2);
    plt.imshow(Z_prob.T); plt.hold(True);
    plt.plot(Z_true, 'w--');
    plt.axis([0, trial_num-1, 0, K-1]); # For speed
    plt.xlabel('trials');
    plt.hold(False);
    plt.ylabel('p(TS|past) at decision time');

    plt.draw();

    # Plot performances
    plt.figure(figsize=(12,9));

    #plot final performance
    plt.subplot(2,2,1)
    plt.plot(np.divide(A_corr_count, np.arange(trial_num)+1), 'k-', linewidth=2); plt.hold(True);
    plt.axis([0,trial_num-1,0,1]);
    plt.hold(False)
    plt.xlabel('trials');
    plt.ylabel('proportion correct answers');

    #plot final Z estimated
    plt.subplot(2,2,3);
    plt.imshow(last_Z_prob.T); plt.hold(True);
    plt.plot(Z_true, 'w--');
    plt.axis([0, trial_num-1, 0, K-1]); # For speed
    plt.hold(False);
    plt.xlabel('trials');
    plt.ylabel('p(TS|past) at current time');

    #plot gamma metropolis-hasting acceptance rate
    plt.subplot(2,2,2);
    plt.plot(m_h_gamma, 'g-');plt.hold(True);
    plt.plot(m_h_tau, 'b-');
    plt.axis([0,trial_num-1, 0,1]); 
    plt.hold(False);
    plt.xlabel('trials');
    plt.ylabel('gamma(green)/tau(blue) acceptance rates');

    #plot gibbs autocorrelation function
    plt.subplot(2,2,4);
    plt.plot(tau_autocorrelation, 'k-'); plt.hold(True);
    plt.axis([0,sample_num-1, 0,1]); 
    plt.hold(False);
    plt.xlabel('trials');
    plt.ylabel('Gibbs sampler autocorrelation');


    plt.draw();

    return 'plot ok'