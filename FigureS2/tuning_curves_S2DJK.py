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

trialdata = pd.read_pickle('trialdata.pkl') #load trial data corresponding to simulation
alldata = np.load('uncoupledchainsModel.npy') #simulated neural activity data, use file corresponding to output of desired model, including saturating uncoupled chains and non-saturating mutually inhibiting chains

#evidence helper function
def get_cum_evidence(lefts, rights, pos):
    '''
    calculate cumulative evidence at each position in the maze
    === inputs ===
    lefts: list of positions of left towers
    rights: list of positions of right towers
    pos: list of maze positions
    '''
    cum_ev = np.zeros(len(pos))
    for i, p in enumerate(pos):
        cum_ev[i] = np.sum(rights<p)-np.sum(lefts<p)
    return cum_ev

def get_average(data, evidence, n_neurons, evs, poses):
    '''
    calculate average neural firing at each position and evidence bin
    === inputs ===
    data: neural firing data for each trial at each position
    evidence: cumulative evidence for each trial at each position
    n_neurons: number of neurons simulated
    evs: list of evidence bins
    poses: list of position bins
    '''
    firingtotals = np.zeros((n_neurons, len(poses), len(evs)))
    counts = np.zeros((n_neurons, len(poses), len(evs)))
    for e, ev in enumerate(evs):
        its, ips = np.where(evidence==ev)
        for (t, p) in zip(its, ips):
            firingtotals[:, p, e]=firingtotals[:, p, e]+data[t, :, p].T #for each observation, add to the observations in that position x evidence bin
            counts[:, p, e] = counts[:, p, e]+1 #track the number of observations of each position x evidnce bin
    countsfordivision = counts.copy()
    countsfordivision[np.where(counts==0)]=1
    firingavg = np.divide(firingtotals,countsfordivision) #get average firing
    return firingavg, counts

[n_trials, n_neurons, n_pos] = np.shape(alldata)

lefts = trialdata['leftcues']
rights = trialdata['rightcues']

pos = np.linspace(-30, 300, 3301) #position bins
evlevels = np.arange(-15, 16) #evidence bins

#calculate cumulative evidence at each position on each trial
evidence = np.zeros((n_trials, n_pos))
for i in range(n_trials):
    evidence[i] = get_cum_evidence(lefts[i], rights[i], pos)

#get average firing rate
firingavg, counts = get_average(alldata, evidence, n_neurons, evlevels, pos)
undersampled = np.where(counts[0, :, :]<2)

plt.figure()

#get tuning curves for example left and right neurons
nLs = [10, 11, 12, 13, 14] #indices of selected left neurons
for i, nL in enumerate(nLs):
    FAL = firingavg[nL, :, :] #firing average of neuron at all position and evidence bins
    FAL[undersampled] = np.nan
    activepL = np.argmax(np.nanmean(FAL, axis=1)) #determine maximal firing position
    plt.plot(evlevels, FAL[activepL, :], color = 'blue', alpha=(5-i)/5) #plot average firing at maximal position

nRs = [27, 28, 29, 30, 31] #indices of selected right neurons
for i, nR in enumerate(nRs): #repeat to get tuning curves for right neurons
    FAR = firingavg[nR, :, :]
    FAR[undersampled] = np.nan
    activepR = np.argmax(np.nanmean(FAR, axis=1))
    plt.plot(evlevels, FAR[activepR, :], color = 'red', alpha=(i+1)/5)

plt.xlim([-12, 12])
plt.xlabel('evidence')
plt.show()
