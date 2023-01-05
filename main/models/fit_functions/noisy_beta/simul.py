import pickle
from smc_simul import smc

td = pickle.load(open('../../../../../../CDD/CDD_ScienceCog_Ulm/volatility/theory/data/tasks/td28_numStimuli1_numActions2.p'))

smc(td, [1,0.72])

p = smc(td, [0.5,10000])