import pickle
path = '../../../data/open_data_case_0.pkl'
td = pickle.load(open(path,'rb'))
# show_progress=True; numberOfStateSamples=500; numberOfThetaSamples=1000; coefficient = .5;


import smc_c
import numpy as np
import get_mapping
import useful_functions
import sys
from scipy.stats import beta as betalib
from scipy.stats import norm as normlib
import matplotlib.pyplot as plt
import time
import numpy
import math


from scipy.special import beta

import bootstrap
a = bootstrap.SMC2(td, beta_softmax=1., lambda_noise=.4, eta_noise=.1, numberOfStateSamples=200, numberOfThetaSamples=200, numberOfBetaSamples=20, coefficient = .5)


from scipy.stats import beta
import matplotlib.pyplot as plt
a = .001
b = .1
x = np.linspace(beta.ppf(0.01, a, b), beta.ppf(0.99, a, b), 100)
fig, ax = plt.subplots(1, 1)
ax.plot(x, beta.pdf(x, a, b),'r-', lw=5, alpha=0.6, label='beta pdf')

def most_common(lst):
    return max(set(list(lst)), key=list(lst).count)

ts   = np.argmax(tsProbability[:T], axis=1)
switch = td['Z'][1:T]!=td['Z'][:(T-1)]
t = 1
index = 0
lis = []
for t in range(T):
	if switch[t-1]:
		last_ts = most_common(ts[index:t])
		i = 0
		while ts[t+i] == last_ts:
			i = i+1
		lis.append(i) 
		index   = t 
	else:
		t += 1


n = 100
for n in range(100):
	gammaCandidate = gammaSamples[n]
	betaCandidate = betaSamples[n]
	d= smc_c.guided_smc_c(candidateTaskSetSamples, T, currentTemperatures, betaCandidate, \
                                                                  gammaCandidate, mapping, stimuli[:T], rewards[:T], actions[:T])
	print d
	print logThetaLks[n]
	print '\n'


print c_smc_functions.guided_smc_c(T, betaCandidate, etaCandidate, gammaCandidate);

	