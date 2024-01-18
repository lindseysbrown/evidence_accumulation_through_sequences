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

alldata = np.load('bumpmodel-full-feedforwardinside-faster.npy')

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
nLs = [323, 325, 327, 329, 331]
for i, nL in enumerate(nLs):
    FAL = firingavg[nL, :, :]
    FAL[undersampled] = np.nan
    activepL = np.argmax(np.nanmean(FAL, axis=1))
    plt.plot(evlevels, FAL[activepL, :], color = 'blue', alpha=(5-i)/5)

nRs = [333, 335, 337, 339, 341]
for i, nR in enumerate(nRs):
    FAR = firingavg[nR, :, :]
    FAR[undersampled] = np.nan
    activepR = np.argmax(np.nanmean(FAR, axis=1))
    plt.plot(evlevels, FAR[activepR, :], color = 'red', alpha=(i+1)/5)

plt.xlim([-12, 12])
plt.xlabel('evidence')

plt.savefig('LRSwitch-MultipleTuningCurvesBump-FeedforwardInside.pdf', transparent = True)

'''
#get evidence tuning curve at active position
activepL = np.argmax(np.nanmean(FAL, axis=1))
activepR = np.argmax(np.nanmean(FAR, axis=1))

ax[0].set_title('left neuron')
ax[0].set_ylabel('average activity')
ax[0].set_xlim([-15, 15])
ax[0].set_ylim([0, 1])

ax[1].plot(evlevels, FAR[activepR, :], color = 'red')
ax[1].set_title('example right neuron')
ax[1].set_xlabel('evidence')
ax[1].set_ylabel('average activity')
ax[1].set_xlim([-15, 15])
ax[1].set_ylim([0, 1])

plt.tight_layout()
plt.savefig('TuningCurvesTiledBump.pdf', transparent = True)
'''

'''
#define transparent colormaps
from matplotlib.colors import ListedColormap
cmap = plt.cm.Purples
purp = cmap(np.arange(cmap.N))
purp[:,-1] = np.linspace(0, 1, cmap.N)
purp = ListedColormap(purp)

cmap = plt.cm.Blues
blue = cmap(np.arange(cmap.N))
blue[:,-1] = np.linspace(0, 1, cmap.N)
blue = ListedColormap(blue)

cmap = plt.cm.Greens
green = cmap(np.arange(cmap.N))
green[:,-1] = np.linspace(0, 1, cmap.N)
green = ListedColormap(green)

cmap = plt.cm.Reds
red = cmap(np.arange(cmap.N))
red[:,-1] = np.linspace(0, 1, cmap.N)
red = ListedColormap(red)

cmap = plt.cm.Oranges
orange = cmap(np.arange(cmap.N))
orange[:,-1] = np.linspace(0, 1, cmap.N)
orange = ListedColormap(orange)

cmap = plt.cm.Greys
grey = cmap(np.arange(cmap.N))
grey[:,-1] = np.linspace(0, 1, cmap.N)
grey = ListedColormap(grey)

#get overlaid receptive fields
cmaplist = [green, red, blue, orange, grey, purp]
ns = [227+35, 260+35, 264+35, 294+35, 297+35, 300+35]

plt.figure()
for i, m in enumerate(cmaplist):
    FA = firingavg[ns[i], :, :]
    FA[undersampled] = np.nan
    plt.imshow(FA, cmap=m, aspect = 'auto')
plt.plot([15, 16, 17, 18, 19, 20, 21, 21], [1400, 1500, 1510, 1750, 1760, 1900, 1910, 2000], color = 'white')
plt.plot([15, 14, 13, 14, 15, 14, 13, 13], [1400, 1500, 1510, 1750, 1760, 1900, 1910, 2000], color = 'black')
plt.ylim([1400, 2000])
plt.xticks([5, 15, 25], labels = ['-10', '0', '10'])
plt.xlabel('evidence')
plt.yticks([1400, 1600, 1800, 2000], labels = ['110', '130', '150', '170'])
plt.ylabel('position')
plt.savefig('RFsBump.pdf', transparent = True)
'''