# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 13:42:39 2021

@author: lindseyb
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd
from scipy import stats
import pickle

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Arial']})

trialdata = pd.read_pickle('trialdata.pkl')

#initialize neural rings
neurons = 35
timepoints = 17
Cring = np.zeros((neurons*timepoints,))

P0 = 20
T = 300

a = 1.1
b =2
w0 = 3

optostrength = 1

#set up synaptic weights
def phi(i):
    return i*2*np.pi/neurons

def F(x):
    return .5*(1+np.tanh(x))

def O(t, i):
    opto = np.zeros((neurons*timepoints,))
    opto[i] = optostrength
    return opto

W = np.zeros((neurons*timepoints, neurons*timepoints))

bump = np.zeros((neurons, neurons))
for i in range(neurons):
    for j in range(neurons):
        bump[i, j] = w0*(np.cos(phi(i)-phi(j))-.9)

W[:neurons,:neurons] = bump

for i in range(1, timepoints):
    W[neurons*i:neurons*(i+1), neurons*i:neurons*(i+1)]=bump
    
feedforward = np.zeros((neurons*timepoints, neurons*timepoints))
i, j = np.indices(W.shape)
feedforward[i==j+neurons] = a*b

def P(t):
    P = np.zeros(neurons*timepoints)
    i = int(np.floor(t/P0))
    P[neurons*i:neurons*(i+1)]=T
    return P

def I(t, C, Lcues, Rcues):
    Lcue = np.abs(t-Lcues)
    if min(Lcue)<.5:
        IL = np.roll(C, -1)
    else:
        IL = np.zeros((neurons*timepoints,))
    Rcue = np.abs(t-Rcues)
    if min(Rcue)<.5:
        IR = np.roll(C, 1)
    else:
        IR = np.zeros((neurons*timepoints,)) 
    return IL+IR

def correct(sol, Lcues, Rcues):
    imax = np.argmax(sol[-1, :])
    if len(Lcues)>len(Rcues):
        return imax<(neurons*timepoints-.5*neurons-1)
    return imax>(neurons*timepoints-.5*neurons)

def simulate(Lcues, Rcues, optoi, input_noise = False, Inoise = .67):
    #reset simulation
    Cring = np.zeros((neurons*timepoints,))
    
    Cring[16] = .6
    Cring[17] = .7
    Cring[18] = .6
    
    Lcues= Lcues+30
    Rcues = Rcues+30

    if input_noise:
        Lkeep = np.random.uniform(0, 1, size = len(Lcues))
        Lcues[Lkeep<Inoise] =  500
        Rkeep = np.random.uniform(0, 1, size = len(Rcues))
        Rcues[Rkeep<Inoise] =  500
    

    def ring(y, t):
        dydt = -a*y+np.heaviside(P(t)-T, 1)*(F(W@y+feedforward@y+5*I(t, y, Lcues, Rcues)))
        return dydt
    
    def optoring(y, t):
        dydt = -a*y+np.heaviside(P(t)-T, 1)*(F(W@y+feedforward@y+5*I(t, y, Lcues, Rcues)+5*O(t, optoi)))
        return dydt
    
    y0 = Cring
    t = np.linspace(0, 330, 3301)
    
    sol = odeint(ring, y0, t, hmax=5)
    optosol = odeint(optoring, y0, t, hmax=5)
    
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


alldata = np.load('bumpmodel-full-feedforwardinside-faster.npy')
lchoices = np.load('lchoices.npy')
rchoices = np.load('rchoices.npy')

avgdata = np.mean(alldata[:, :, :], axis=0) #neurons x position

leftis, rightis, splitis, nonis = divide_LR(alldata, lchoices, rchoices, pthresh, region_threshold, region_width, base_thresh)


maxchanges = np.zeros((len(leftis), neurons*timepoints))

for i, l in enumerate(leftis):
    Lcues = np.array([500])
    Rcues = np.array([500])
    sol, optosol = simulate(Lcues, Rcues, l)

    delta = optosol-sol
    
    increases = np.max(delta, axis=0)
    decreases = np.min(delta, axis=0)
    
    maxchanges[i] = increases+decreases

evidencechanges = maxchanges.copy()
with open('bumpetuning.pkl', 'rb') as handle:
    evtuning = pickle.load(handle)

neuronsatev = {}
for n in evtuning.keys():
    e = evtuning[n]
    if e in neuronsatev:
        neuronsatev[e] = neuronsatev[e] + [n]
    else:
        neuronsatev[e] = [n]

evlevels = sorted(list(neuronsatev.keys()))

evavgchange = np.zeros((len(leftis), len(evlevels)))           
for i, l in enumerate(leftis):
    lev = evtuning[l]
    ltier = np.floor(l/35)*35
    for e, evval in enumerate(evlevels):
        nlist = np.array(neuronsatev[evval])
        nlist = nlist[nlist>ltier]
        evavgchange[i, e] = np.mean(maxchanges[i, nlist])
        

    


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
   
#newLis = np.argsort(np.argmax((LCellLChoice+LCellRChoice)/2, axis=1))
newLis = np.argsort(leftis)

maxchanges = maxchanges[newLis, :]
evidencechanges = evidencechanges[newLis, :]
evavgchange = evavgchange[newLis, :]
maxchangesleft = maxchanges[:, leftis]
maxchangesright = maxchanges[:, rightis]
maxchangessplit = maxchanges[:, splitis]

evidencechangesleft = evidencechanges[:, leftis]
evidencechangesright = evidencechanges[:, rightis]
evidencechangescombined = np.concatenate((evidencechangesleft, evidencechangesright), axis=1)

#resort evidence data to sort by evidence tuning
tuningleft = np.array([evtuning[ecell] for ecell in leftis])
tuningright = np.array([evtuning[ecell] for ecell in rightis])
tuningcombined = np.concatenate((tuningleft, tuningright))
newis = np.argsort(tuningcombined)
evidencechangescombined = evidencechangescombined[:, newis]


#resort data so always in sequence order
LCellLChoice = LCellLChoice[newLis, :]
LCellRChoice = LCellRChoice[newLis, :]
maxchangesleft = maxchangesleft[:, newLis]

RCellLChoice = RCellLChoice[1:, :]
RCellRChoice = RCellRChoice[1:, :]
   
#newRis = np.argsort(np.argmax((RCellRChoice+RCellLChoice)/2, axis=1))
newRis = np.argsort(rightis)

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

ax[0].imshow(maxchangesleft, cmap = 'bwr', aspect='equal', vmin=-1, vmax=1)
ax[0].set_xlabel('left cells')
ax[0].set_ylabel('stimulated left cell')
ax[0].set_xticks([])
ax[0].set_yticks([])

ax[1].imshow(maxchangesright, cmap = 'bwr', aspect='equal', vmin=-1, vmax=1)
ax[1].set_xlabel('right cells')
ax[1].set_xticks([])
ax[1].set_yticks([])

ax[2].imshow(maxchangessplit, cmap = 'bwr', aspect='equal', vmin=-1, vmax=1)
ax[2].set_xlabel('non-pref. cells')
ax[2].set_xticks([])
#ax[2].set_xticklabels(['-30', '0', 'cues', '200', 'delay', '300'])
#ax[2].set_xlabel('position (cm)')
ax[2].set_yticks([])

plt.savefig('PlanarBumpOptoHeatmap-FFinside.pdf', transparent= True)

leftresp = maxchangesleft[78, :]
rightresp = maxchangesright[78, :]
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
plt.bar(np.arange(len(changes)), changes, color = 'k', align = 'edge', width=1)
#plt.axvspan(76,77, facecolor='yellow', alpha=.4, edgecolor=None)
plt.axvline(99, color = 'grey', linestyle='--')
plt.axvline(116, color ='grey', linestyle = '--')
plt.xlim([0, len(changes)])
plt.ylim([-0.75, 1.25])
plt.plot(np.arange(len(changes)), np.zeros(len(changes)), color = 'k', linestyle = '--')
plt.vlines(len(leftresp), -12, 22, color='k', linestyle = '--')
plt.ylabel('change from baseline', fontsize=24)
plt.xticks([len(leftresp)/2, len(leftresp)+len(rightresp)/2], labels = ['left neurons', 'right neurons'], fontsize=24)
plt.xlabel('sorted by position', fontsize=24)
plt.tight_layout()
plt.savefig('NEWBarOptoBump-truesort-FFinside.pdf', transparent=True)


echanges = evidencechangescombined[76, :]

ecolors = []
for c in echanges:
    if c<0:
        ecolors.append('blue')
    else:
        ecolors.append('red')
  
cell = np.where(newis==78)[0][0] #not sure this is actually correct cell still...     
plt.figure()
plt.bar(np.arange(len(changes)), echanges, color = ecolors, align = 'edge', width=1)
plt.axvspan(cell, cell+1, facecolor='yellow', alpha=.9, edgecolor=None)
plt.ylabel('change from baseline', fontsize=24)
plt.xlabel('neurons sorted by evidence', fontsize = 24)
plt.savefig('BarOptoBump-EvIndividualCells-FFinside.pdf')

echanges2 = evavgchange[78, :]
ecolors2 = []
for c in echanges2:
    if c<0:
        ecolors2.append('blue')
    else:
        ecolors2.append('red')

#ecell = evtuning[leftis[newLis[76]]]
plt.figure()
plt.bar(-1*np.array(evlevels), echanges2, color = 'k', align = 'edge', width=1)
plt.ylim([-.75, 1.25])
#plt.axvspan(ecell,ecell+1, facecolor='yellow', alpha=.4, edgecolor=None)
plt.ylabel('change from baseline', fontsize=24)
plt.xlabel('evidence', fontsize = 24)
plt.savefig('LRSwitch-NEWBarOptoBump-Ev-FFinside.pdf', transparent=True)

#inset
plt.figure()
plt.bar(np.arange(len(changes[99:116])), changes[99:116], color = 'k', align='edge', width=1)
plt.savefig('BarOpto-inset-FFinside.pdf')





