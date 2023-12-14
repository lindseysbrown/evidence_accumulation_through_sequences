# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 10:44:19 2022

@author: lindseyb
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Arial']})

trialdata = pd.read_pickle('trialdata.pkl')

#evidence helper function
def get_cum_evidence(lefts, rights, pos):
    cum_ev = np.zeros(len(pos))
    for i, p in enumerate(pos):
        cum_ev[i] = np.sum(rights<p)-np.sum(lefts<p)
    return cum_ev

def get_average(data, evidence, n_neurons, evs, poses):
    firingtotals = np.zeros((n_neurons, len(poses), len(evs)))
    counts = np.zeros((n_neurons, len(poses), len(evs)))
    for e, ev in enumerate(evs):
        its, ips = np.where(evidence==ev)
        for (t, p) in zip(its, ips):
            firingtotals[:, p, e]=firingtotals[:, p, e]+data[t, :, p].T
            counts[:, p, e] = counts[:, p, e]+1
    countsfordivision = counts.copy()
    countsfordivision[np.where(counts==0)]=1
    firingavg = np.divide(firingtotals,countsfordivision)
    return firingavg, counts

alldata = np.load('MIModel.npy')

[n_trials, n_neurons, n_pos] = np.shape(alldata)

lefts = trialdata['leftcues']
rights = trialdata['rightcues']

pos = np.linspace(-30, 300, 3301)
evlevels = np.arange(-15, 16)

evidence = np.zeros((n_trials, n_pos))
for i in range(n_trials):
    evidence[i] = get_cum_evidence(lefts[i], rights[i], pos)

firingavg, counts = get_average(alldata, evidence, n_neurons, evlevels, pos)
undersampled = np.where(counts[0, :, :]<2)

plt.figure()

#get tuning curves for example left and right neurons
nLs = [10, 11, 12, 13, 14]
for i, nL in enumerate(nLs):
    FAL = firingavg[nL, :, :]
    FAL[undersampled] = np.nan
    activepL = np.argmax(np.nanmean(FAL, axis=1))
    plt.plot(evlevels, FAL[activepL, :], color = 'blue', alpha=(5-i)/5)

nRs = [27, 28, 29, 30, 31]
for i, nR in enumerate(nRs):
    FAR = firingavg[nR, :, :]
    FAR[undersampled] = np.nan
    activepR = np.argmax(np.nanmean(FAR, axis=1))
    plt.plot(evlevels, FAR[activepR, :], color = 'red', alpha=(i+1)/5)

plt.xlim([-12, 12])
plt.xlabel('evidence')

plt.savefig('LRSwitch-MIMultipleTuningCurvesChains.pdf', transparent = True)
