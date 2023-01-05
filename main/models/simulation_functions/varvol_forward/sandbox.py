index = 0;

logThetaLks11 = np.zeros(numberOfThetaSamples)
for t in range(0, T):
    smc_c.bootstrapUpdateStep_c(currentStateSamples, currentTauSamples, gammaSamples, betaSamples, nuSamples, tauDefault, T, \
                                            np.ascontiguousarray(ancestorStateSamples, dtype=np.intc), ancestorTauSamples, ancestorsWeights, np.ascontiguousarray(mapping), stimuli[T-1])
    # Update theta weights
    tasksets                                 = np.equal(mapping[stimuli[t], :], actions[t])
    lastPossibleTasksets                     = np.logical_xor(tasksets, ~rewards[t])
    ancestorStateSamples                           = np.array(currentStateSamples)
    unnormalisedAncestorsWeights               = (lastPossibleTasksets[ancestorStateSamples].T * betaSamples).T + (~lastPossibleTasksets[ancestorStateSamples].T * (1 - betaSamples)).T
   # for idx in range(numberOfStateSamples):
   # 	unnormalisedAncestorsWeights[0,idx]             = lastPossibleTasksets[ancestorSamples0[0,idx]] * betaSamples0[0] + (1 - lastPossibleTasksets[ancestorSamples0[0,idx]])*(1 - betaSamples0[0])
    logThetaLks11                             += np.log(np.sum(unnormalisedAncestorsWeights, axis=1)/numberOfStateSamples)
    ancestorsWeights                         = (unnormalisedAncestorsWeights.T/np.sum(unnormalisedAncestorsWeights, axis=1)).T

print(logThetaLks1)

for i in range(100):
	nuCandidate = nuSamples[i]
	betaCandidate = betaSamples[i]
	gammaCandidate = gammaSamples[i]
	logLksCandidate = smc_c.guidedSmc_c(np.ascontiguousarray(stateSamplesCandidates), tauSamplesCandidates, weightsSamplesCandidates, gammaCandidate, betaCandidate, nuCandidate, tauDefault, np.ascontiguousarray(mapping), \
                                                            np.ascontiguousarray(stimuli[:T], dtype=np.intc), np.ascontiguousarray(rewards[:T], dtype=np.intc), np.ascontiguousarray(actions[:T], dtype=np.intc), numberOfStateSamples)
	print(logLksCandidate)
	print(logThetaLks[i])
	print(logThetaLks11[i])
	index += logLksCandidate > logThetaLks[i]
	print('\n')

nuCandidate    = nuSamples[0]
betaCandidate  = betaSamples[0]
gammaCandidate = gammaSamples[0]

currentStateSamples0   = np.array([currentStateSamples[0]])
currentTauSamples0   = np.array([currentTauSamples[0]])
ancestorsWeights0 = np.array([ancestorsWeights[0]])
gammaSamples0     = np.array([gammaSamples[0]])
betaSamples0      = np.array([betaSamples[0]])
nuSamples0       = np.array([nuSamples[0]])
logThetaLks0      = np.zeros(1);
logThetaLks1      = np.zeros(1);
logThetaWeights0  = np.zeros(1);
ancestorStateSamples0  = np.array([ancestorStateSamples[0]])
ancestorTauSamples0 = np.array([ancestorTauSamples[0]])

for t in range(1,T+1):
    smc_c.guidedUpdateStep_c(logThetaLks0, logThetaWeights0, np.ascontiguousarray(currentStateSamples0), currentTauSamples0, gammaSamples0, betaSamples0, nuSamples0, tauDefault, t, np.ascontiguousarray(ancestorStateSamples0),\
                                        ancestorTauSamples0, ancestorsWeights0, np.ascontiguousarray(mapping), stimuli[t-2], stimuli[t-1], rewards[t-1], actions[t-1])
    ancestorTauSamples0   = np.array(currentTauSamples0)
    ancestorStateSamples0 = np.array(currentStateSamples0)

print logThetaLks0

logLksCandidate = smc_c.guidedSmc_c(np.ascontiguousarray(stateSamplesCandidates), tauSamplesCandidates, weightsSamplesCandidates, gammaCandidate, betaCandidate, nuCandidate, tauDefault, np.ascontiguousarray(mapping), \
                                                            np.ascontiguousarray(stimuli[:T], dtype=np.intc), np.ascontiguousarray(rewards[:T], dtype=np.intc), np.ascontiguousarray(actions[:T], dtype=np.intc), numberOfStateSamples)

print logLksCandidate

for t in range(0, T):
    smc_c.bootstrapUpdateStep_c(np.ascontiguousarray(currentSamples0), gammaSamples0, betaSamples0, tauSamples0, t, np.ascontiguousarray(ancestorSamples0), np.ascontiguousarray(ancestorsWeights0), np.ascontiguousarray(mapping), stimuli[T-1])
    # Update theta weights
    tasksets                                 = np.equal(mapping[stimuli[t], :], actions[t])
    lastPossibleTasksets                     = np.logical_xor(tasksets, ~rewards[t])
    ancestorSamples0                         = np.array(currentSamples0)
    unnormalisedAncestorsWeights             = (lastPossibleTasksets[ancestorSamples0].T * betaSamples0).T + (~lastPossibleTasksets[ancestorSamples0].T * (1 - betaSamples0)).T
   # for idx in range(numberOfStateSamples):
   # 	unnormalisedAncestorsWeights[0,idx]             = lastPossibleTasksets[ancestorSamples0[0,idx]] * betaSamples0[0] + (1 - lastPossibleTasksets[ancestorSamples0[0,idx]])*(1 - betaSamples0[0])
    logThetaLks1                            += np.log(np.sum(unnormalisedAncestorsWeights, axis=1)/numberOfStateSamples)
    ancestorsWeights0                         = (unnormalisedAncestorsWeights.T/np.sum(unnormalisedAncestorsWeights, axis=1)).T

print(logThetaLks1)