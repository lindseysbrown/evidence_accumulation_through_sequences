# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 11:53:49 2022

@author: lindseyb
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from skimage.measure import block_reduce
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Arial']})

#set colormap
current_cmap = matplotlib.cm.get_cmap('YlOrRd')
current_cmap.set_bad(color='black', alpha=.3) 

#load trial evidence data
trialdata = pd.read_pickle('ExampleData/trialdata.pkl')

alldata = .5*np.load('MIreduced.npy') #results of simulation of mutually inhibiting chains model, mapped to be in position bins of 5cm
leftchoicecells = np.zeros((1000, 17*8, 66))
rightchoicecells = np.zeros((1000, 17*8, 66))

pos = np.arange(-30, 300, 5)
neurons = 17
Plength = 20
    

for i, p in enumerate(pos):
    activecell = int(np.floor((p+30)/Plength)) #read from the cell that is active at that position
    base = np.sum(np.arange(activecell)) #determine number of choice readout cells
    for a in range(activecell):
        leftchoicecells[:, base+a, i] = 5*(np.sign(alldata[:, activecell, i]-alldata[:, activecell+neurons, i]+np.random.normal(size = (1000,), scale=2))+1) #readout choice response with noise, left preferring
        rightchoicecells[:, base+a, i] = 5*(np.sign(alldata[:, activecell+neurons, i]-alldata[:, activecell, i]+np.random.normal(size = (1000,), scale=2))+1) #readout choice response with noise, right preferring
        
        

#function to determine cumulative evidence at each position    
def get_cum_evidence(lefts, rights, pos):
    cum_ev = np.zeros(len(pos))
    for i, p in enumerate(pos):
        cum_ev[i] = np.sum(lefts<p)-np.sum(rights<p)
    return cum_ev

pos = np.arange(-30, 300, 5)
n_trials = 1000
n_pos = len(pos)
evidence = np.zeros((n_trials, n_pos))
for i in range(n_trials):
    evidence[i] = get_cum_evidence(trialdata['leftcues'][i], trialdata['rightcues'][i], pos)


#make plot of population average firing rate vs. position and evidence
def get_tuningcurve(data, evidence, tuningdict):
    '''
    build dictionary of neural firing rates in each position x evidence bin

    === inputs ===
    data: neural firing data at each position on each trial
    evidence: cumulative evidence at each position on each trial
    tuningdict: current dictionary of neural firing rates with keys tuples of evidence and position indices and values an array of observed firing rates

    === outputs ===
    tuningdict: updated dictionary of neural firing rates

    '''
    evs = np.arange(-15, 16)
    obs = np.zeros((len(evs), 66))
    for p in range(66):
        for i, e in enumerate(evs):
            validis = np.where(evidence[:, p]==e)[0]
            if len(validis)>2: #1: #2 or more trials
                if (i, p) in tuningdict:
                    tuningdict[(i, p)] = np.concatenate((tuningdict[(i, p)], np.nanmean(data[validis, :, p], axis=0)))
                else:
                    tuningdict[(i, p)] = np.mean(data[validis, :, p], axis=0)
    return tuningdict

def jointdict_to_tuning(tuningdictright, tuningdictleft):
    '''
    take tuning dictionaries and convert to population averages

    ===inputs===
    tuningdictright: dictionary with keys tuples of evidence and position and values lists of mean firing rate in that bin of each right preferring neuron
    tuningdictleft: dictionary with keys tuples of evidence and position and values lists of mean firing rate in that bin of each left preferring neuron

    ===outputs===
    array of average activity in preferred evidence by position bins
    '''
    evs = np.arange(-15, 16)
    obs = np.zeros((len(evs), 66))
    for p in range(66):
        for i, e in enumerate(evs):
            ri = np.where(evs==-1*e)[0][0]
            if (i, p) in tuningdictleft:
                obsL = tuningdictleft[(i,p)]
            else:
                obsL = []
            if (ri, p) in tuningdictright:
                obsR = tuningdictright[(ri,p)]
            else:
                obsR = [] 
            if len(obsL)+len(obsR)>0:
                vals = np.concatenate((obsL, obsR))
                q1 = np.percentile(vals, 25)
                q3 = np.percentile(vals, 75)
                outlierhigh = vals>(1.5*(q3-q1)+q3)
                outlierlow = vals<(q1-1.5*(q3-q1))
                if sum(outlierhigh)==1:
                    valid = ~ outlierhigh
                    vals = vals[valid] #remove if single outlier
                obs[i, p] = np.mean(vals)
            else:
                obs[i, p] = np.nan
    return obs

regiontuningdictL = {}
regiontuningdictR = {}
lefttotal = np.concatenate((alldata[:, :neurons, :], leftchoicecells), axis=1)
righttotal = np.concatenate((alldata[:, neurons:, :], rightchoicecells), axis=1)
regiontuningdictL = get_tuningcurve(lefttotal, evidence, regiontuningdictL)
regiontuningdictR = get_tuningcurve(righttotal, evidence, regiontuningdictR)
poptuningcombined = jointdict_to_tuning(regiontuningdictL, regiontuningdictR)


colors = {16:'darkslateblue', 26:'dodgerblue', 36:'aqua', 46:'purple', 56:'fuchsia'}

#plot 2D population tuning curve
plt.figure()
plt.imshow(poptuningcombined, cmap='YlOrRd', vmin=0, vmax = 1, aspect='auto', interpolation='none')
plt.xticks([0, 5.5, 25.5, 45.5, 55.25, 65], labels=['-30', '0', 'cues', '200', 'delay', '300'])
plt.yticks([5, 15, 25], ['-10', '0', '10'])
for c, i in enumerate([16, 26, 36, 46, 56]):
    plt.axvline(i, color=colors[i], linewidth=2.5, linestyle = '--')
plt.title(' Preferred - Non Pref Population Tuning')
plt.colorbar()
plt.ylim([25,5])
plt.show()

#get cross-sections of population tuning curves
evs = np.arange(-15, 16)
plt.figure()
for c, i in enumerate([16, 26, 36, 46, 56][::-1]):
    plt.plot(evs, poptuningcombined[:, i][::-1], color=colors[i], linewidth = 3)
plt.title(' Preferred - Non Pref')
plt.xlim([-10, 10])
plt.ylim([0, 1.05])
plt.show()


#get tuning curves for individual cells
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
            firingtotals[:, p, e]=firingtotals[:, p, e]+data[t, :, p].T
            counts[:, p, e] = counts[:, p, e]+1
    countsfordivision = counts.copy()
    countsfordivision[np.where(counts==0)]=1
    firingavg = np.divide(firingtotals,countsfordivision)
    return firingavg, counts

evlevels = np.arange(-15, 16)

firingavg, counts = get_average(np.concatenate((leftchoicecells, rightchoicecells), axis=1), evidence, 2*17*8, evlevels, pos)
undersampled = np.where(counts[0, :, :]<1)

countchoice = 17*8

plt.figure()
nLs = [6]
for i, nL in enumerate(nLs):
    FAL = firingavg[nL, :, :]
    FAL[undersampled] = np.nan
    activepL = np.argmax(np.nanmean(FAL, axis=1))
    plt.plot(-1*evlevels, FAL[activepL, :], color = 'blue')
nRs = [6+countchoice]
for i, nR in enumerate(nRs):
    FAR = firingavg[nR, :, :]
    FAR[undersampled] = np.nan
    activepR = np.argmax(np.nanmean(FAR, axis=1))
    plt.plot(-1*evlevels, FAR[activepR, :], color = 'red')
plt.title('early cells')
plt.xlim([-6, 6])
plt.ylim([0, 10.5])
plt.show()

plt.figure()
nLs = [28]
for i, nL in enumerate(nLs):
    FAL = firingavg[nL, :, :]
    FAL[undersampled] = np.nan
    activepL = np.argmax(np.nanmean(FAL, axis=1))
    plt.plot(-1*evlevels, FAL[activepL, :], color = 'blue')
nRs = [28+countchoice]
for i, nR in enumerate(nRs):
    FAR = firingavg[nR, :, :]
    FAR[undersampled] = np.nan
    activepR = np.argmax(np.nanmean(FAR, axis=1))
    plt.plot(-1*evlevels, FAR[activepR, :], color = 'red')
plt.title('late cells')
plt.xlim([-10, 10])
plt.ylim([0, 10.5])
plt.show()

plt.figure()
nLs = [110]
for i, nL in enumerate(nLs):
    FAL = firingavg[nL, :, :]
    FAL[undersampled] = np.nan
    activepL = np.argmax(np.nanmean(FAL, axis=1))
    plt.plot(-1*evlevels, FAL[activepL, :], color = 'blue')
nRs = [110+countchoice]
for i, nR in enumerate(nRs):
    FAR = firingavg[nR, :, :]
    FAR[undersampled] = np.nan
    activepR = np.argmax(np.nanmean(FAR, axis=1))
    plt.plot(-1*evlevels, FAR[activepR, :], color = 'red')
plt.title('delay cells')
plt.xlim([-10, 10])
plt.ylim([0, 10.5])
plt.show()

