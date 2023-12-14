# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 11:53:49 2022

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

#parameters
a = 1
b = a
c = 0
Plength = 20
optostrength = 1

T = 300

#initialize neural rings
neurons = 17

#set up connection matrix
W = a*np.identity(neurons*2)

for i in range(1, neurons):
    #feedfoward connections
    W[i, i-1] = b
    W[i+neurons, i-1+neurons] = b
    #inhibitory connections
    W[i+neurons, i] = -c
    W[i, i+neurons] = -c


def P(t):
    pos = np.zeros((neurons,))
    pos[int(np.floor(t/Plength))] = T
    return np.concatenate((pos, pos))

def O(t, i):
    optoL = np.zeros((neurons,))
    optoR = np.zeros((neurons,))
    optoL[i] = optostrength
    return np.concatenate((optoL, optoR))  

def I(t, Lcues, Rcues):
    Lcue = np.abs(t-Lcues)
    if min(Lcue)<.5:
        IL = 4*np.ones((neurons,))
    else:
        IL = np.zeros((neurons,))
    Rcue = np.abs(t-Rcues)
    if min(Rcue)<.5:
        IR = 4*np.ones((neurons,))
    else:
        IR = np.zeros((neurons,))    
    return np.concatenate((IL-IR, IR-IL))

def correct(sol, Lcues, Rcues):
    if len(Lcues)>len(Rcues):
        return sol[-1,16]>sol[-1, 33]
    return sol[-1, 16]<sol[-1, 33]

def simulate(Lcues, Rcues, optoi, input_noise = False, Inoise = .67):
    #reset simulation
    Lchain = np.zeros((neurons,))
    Rchain = np.zeros((neurons,))
    
    Lchain[0] = 65
    Rchain[0] = 65
    
    Lcues= Lcues+30
    Rcues = Rcues+30
    
    if input_noise:
        Lkeep = np.random.uniform(0, 1, size = len(Lcues))
        Lcues[Lkeep<Inoise] =  500
        Rkeep = np.random.uniform(0, 1, size = len(Rcues))
        Rcues[Rkeep<Inoise] =  500
        
    

    def chain(y, t):
        dydt = -a*y+np.maximum(W@y+P(t)+I(t, Lcues, Rcues)-T, 0)
        return dydt

    def optochain(y, t):
        dydt = -a*y+np.maximum(W@y+P(t)+O(t, optoi)+I(t, Lcues, Rcues)-T, 0)
        return dydt


    y0 = np.concatenate((Lchain, Rchain))
    
    t = np.linspace(0, 330, 3301)
    
    sol = odeint(chain, y0, t, hmax=5)
    optosol = odeint(optochain, y0, t, hmax=5)
    return sol, optosol

#code for getting sequence plots identical to neural data
pthresh = .1
region_threshold = .25 #.5 Ryan
region_width = 4
base_thresh = 0 #3 Ryan

def get_regions(data, M, threshold, width, base_thresh):
    upreg = np.where(data>(threshold*M))[0]
    baseline = np.mean(data[~upreg])
    regions = []
    if len(upreg)==0:
        return regions
    last = upreg[0]
    i=1
    currentreg = [last]
    while i<(len(upreg)-1):
        curr = upreg[i]
        if curr == last+1:
            currentreg.append(curr)
        else:
            if len(currentreg)>width:
                regions.append(currentreg)
            currentreg = [curr]
        last = curr        
        i=i+1
    if len(currentreg)>width:
        if np.mean(data[currentreg])>base_thresh*baseline:
            regions.append(currentreg)
    return regions

def divide_LR(alldata, leftchoices, rightchoices, pthresh, region_threshold, region_width, basethresh):
    '''

    Parameters
    ----------
    alldata : neural data (trials x neuron x position)
    leftchoices : trials in which the animal went left
    rightchoices : trials in which the animal went right

    Returns
    -------
    indices of left preferrring neurons, indices of right preferring neurons, 
    and indices of neurons with no significant difference in response between
    the two choices

    '''
    avgdata = np.mean(alldata, axis=0) #neurons x position
    
    #transform data by substracting the minimum
    avgdata = avgdata - np.reshape(np.min(avgdata, axis=1), (-1, 1))
    
    left_neurons = []
    right_neurons = []
    split_neurons = [] #neurons are still task modulated but not significant choice selective
    nonmod_neurons = [] #neurons with no peaks
    

    maxfire = np.max(avgdata, axis=1)
    rightfires = alldata[rightchoices, :, :]
    leftfires = alldata[leftchoices, :, :]
    for i, m in enumerate(maxfire):
        upregions = get_regions(avgdata[i, :], m, region_threshold, region_width, base_thresh)
        left = False
        right = False
        for region in upregions:
            leftfiring = leftfires[:, i, region]
            leftactivity = np.mean(np.round(leftfiring, 2), axis=1)
            rightfiring = rightfires[:, i, region]
            rightactivity = np.mean(np.round(rightfiring, 2), axis=1)
            tval, pval = stats.ttest_ind(leftactivity, rightactivity)
            if pval <2*pthresh:
                if np.round(np.mean(leftactivity), 3)>np.round(np.mean(rightactivity),3):
                    left = True
                if np.round(np.mean(rightactivity), 3)>np.round(np.mean(leftactivity),3):
                    right = True
        if not (right and left):
            if right:
                right_neurons.append(i)
            elif left:
                left_neurons.append(i)
            else:
                if len(upregions)>0:
                    split_neurons.append(i)
                else:
                    nonmod_neurons.append(i) 
        else:
            if len(upregions)>0:
                split_neurons.append(i)
            else:
                nonmod_neurons.append(i)                  
    return np.array(left_neurons), np.array(right_neurons), np.array(split_neurons), np.array(nonmod_neurons)

LCellLChoice = np.zeros((1, 3301))
LCellRChoice = np.zeros((1, 3301))

RCellLChoice = np.zeros((1, 3301))
RCellRChoice = np.zeros((1, 3301))

SCellLChoice = np.zeros((1, 3301))
SCellRChoice = np.zeros((1, 3301))

NCellLChoice = np.zeros((1, 3301))
NCellRChoice = np.zeros((1, 3301))


alldata = np.load('competinginputs1000.npy')
lchoices = np.load('lchoices.npy')
rchoices = np.load('rchoices.npy')

avgdata = np.mean(alldata[:, :, :], axis=0) #neurons x position

leftis, rightis, splitis, nonis = divide_LR(alldata, lchoices, rchoices, pthresh, region_threshold, region_width, base_thresh)


maxchanges = np.zeros((len(leftis), 2*neurons))

for i, l in enumerate(leftis):
    Lcues = np.array([500])
    Rcues = np.array([500])
    sol, optosol = simulate(Lcues, Rcues, l)

    delta = optosol-sol
    
    increases = np.max(delta, axis=0)
    decreases = np.min(delta, axis=0)
    
    maxchanges[i] = increases+decreases


leftdata = alldata[lchoices, :, :]
rightdata = alldata[rchoices, :, :]

leftcellsleftchoice = np.mean(leftdata[:, leftis, :], axis=0)
leftcellsrightchoice = np.mean(rightdata[:, leftis, :], axis=0)

rightcellsleftchoice = np.mean(leftdata[:, rightis, :], axis=0)
rightcellsrightchoice = np.mean(rightdata[:, rightis, :], axis=0)

splitcellsleftchoice = np.mean(leftdata[:, splitis, :], axis=0)
splitcellsrightchoice = np.mean(rightdata[:, splitis, :], axis=0)
    
LCellLChoice = np.vstack((LCellLChoice, leftcellsleftchoice))
LCellRChoice = np.vstack((LCellRChoice, leftcellsrightchoice))

RCellLChoice = np.vstack((RCellLChoice, rightcellsleftchoice))
RCellRChoice = np.vstack((RCellRChoice, rightcellsrightchoice))

SCellLChoice = np.vstack((SCellLChoice, splitcellsleftchoice))
SCellRChoice = np.vstack((SCellRChoice, splitcellsrightchoice))
    

LCellLChoice = LCellLChoice[1:, :]
LCellRChoice = LCellRChoice[1:, :]
   
newLis = np.argsort(np.argmax((LCellLChoice+LCellRChoice)/2, axis=1))

maxchanges = maxchanges[newLis, :]
maxchangesleft = maxchanges[:, leftis]
maxchangesright = maxchanges[:, rightis]
maxchangessplit = maxchanges[:, splitis]


#resort data so always in sequence order
LCellLChoice = LCellLChoice[newLis, :]
LCellRChoice = LCellRChoice[newLis, :]
maxchangesleft = maxchangesleft[:, newLis]

RCellLChoice = RCellLChoice[1:, :]
RCellRChoice = RCellRChoice[1:, :]
   
newRis = np.argsort(np.argmax((RCellRChoice+RCellLChoice)/2, axis=1))

#resort data so always in sequence order
RCellLChoice = RCellLChoice[newRis, :]
RCellRChoice = RCellRChoice[newRis, :]
maxchangesright = maxchangesright[:, newRis]

SCellLChoice = SCellLChoice[1:, :]
SCellRChoice = SCellRChoice[1:, :]
   
newSis = np.argsort(np.argmax((SCellLChoice+SCellRChoice)/2, axis=1))

#resort data so always in sequence order
SCellLChoice = SCellLChoice[newSis, :]
SCellRChoice = SCellRChoice[newSis, :]
maxchangesplit = maxchangessplit[:, newSis]
   

#nonnormalized plots
fig, ax = plt.subplots(1, 3,  gridspec_kw={'width_ratios': [len(newLis), len(newRis), len(newSis)]})

ax[0].imshow(maxchangesleft, cmap = 'bwr', aspect='equal', vmin=-20, vmax=20)
ax[0].set_xlabel('left cells')
ax[0].set_ylabel('stimulated left cell')
ax[0].set_xticks([])
ax[0].set_yticks([])

ax[1].imshow(maxchangesright, cmap = 'bwr', aspect='equal', vmin=-20, vmax=20)
ax[1].set_xlabel('right cells')
ax[1].set_xticks([])
ax[1].set_yticks([])

ax[2].imshow(maxchangessplit, cmap = 'bwr', aspect='equal', vmin=-20, vmax=20)
ax[2].set_xlabel('non-pref. cells')
ax[2].set_xticks([])
#ax[2].set_xticklabels(['-30', '0', 'cues', '200', 'delay', '300'])
#ax[2].set_xlabel('position (cm)')
ax[2].set_yticks([])

plt.savefig('CompetingInputsOptoHeatmap.pdf', transparent= True)

leftresp = maxchangesleft[4, :]
rightresp = maxchangesright[4, :]
changes = np.concatenate((leftresp, rightresp))

colors = []
for c in changes:
    if c<0:
        colors.append('blue')
    else:
        colors.append('red')
    
plt.figure()
#plt.axvspan(0, len(leftresp), color = 'red', alpha = .5)
#plt.axvspan(len(leftresp), len(changes), color = 'blue', alpha = .5)
plt.bar(np.arange(len(changes)), changes, color = colors, align = 'edge', width=1)
plt.axvspan(4,5, facecolor='yellow', alpha=.4, edgecolor=None)
plt.xlim([0, len(changes)])
plt.ylim([-12, 22])
plt.plot(np.arange(len(changes)), np.zeros(len(changes)), color = 'k', linestyle = '--')
plt.vlines(len(leftresp), -12, 22, color='k', linestyle = '--')
plt.ylabel('change from baseline', fontsize=24)
plt.xticks([len(leftresp)/2, len(leftresp)+len(rightresp)/2], labels = ['left neurons', 'right neurons'], fontsize=24)
plt.xlabel('sorted by position', fontsize=24)
plt.tight_layout()
plt.savefig('BarOptoInputs.pdf', transparent=True)