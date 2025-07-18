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


trialdata = pd.read_pickle('ExampleData/trialdata.pkl')
scale=.5

#parameters
a = 1
b = .2
c = a
e = a-b
P0 = 20
baseline = 0
T = 300
externalI = 20*scale

m = .2
h = 4

#initialize neural rings
neurons = 12

#set up connection matrix
W = b*np.identity(neurons*2)

for i in range(1, neurons):
    #feedfoward connections
    W[i, i-1] = c
    W[i+neurons, i-1+neurons] = c
    #inhibitory connections
    W[i+neurons, i] = -e
    W[i, i+neurons] = -e


#build a set of postion gates to tile the time of the trial, with a different signal for each neuron
Pdivs = np.sort(np.round(np.random.uniform(0, 330, size  = neurons)))
gaps = np.diff(Pdivs)
shorts = np.where(gaps<5)[0] #require that each neuron's active position is at least 5 cm

while len(shorts)>0:
    for s in shorts:
        Pdivs[s+1] = Pdivs[s+1]+5
    Pdivs = np.sort(np.round(np.random.uniform(0, 330, size  = neurons)))
    gaps = np.diff(Pdivs)
    shorts = np.where(gaps<5)[0]            

#define position gating signal based on divisions between neuron's active positions
def P(t, Pdivs):
    '''
    Position gate is active for the neuron corresponding to the division of position selectivity containing p(t)
    '''
    pos = np.zeros((neurons,))
    try:
        i0 = np.where(t<Pdivs)[0][0]
    except:
        i0 = -1
    pos[i0] = T+externalI+pos[i0]
    return np.concatenate((pos, pos))

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
    return np.concatenate((IL, IR))

def correct(sol, Lcues, Rcues):
    if len(Lcues)>len(Rcues):
        return sol[-1,16]>sol[-1, 33]
    return sol[-1, 16]<sol[-1, 33]

def simulate(Lcues, Rcues, input_noise = False, Inoise = .67):
    #reset simulation
    Lchain = np.zeros((neurons,))
    Rchain = np.zeros((neurons,))
    
    Lchain[0] = baseline
    Rchain[0] = baseline
   
    #first 30 cm prior to cue onset
    Lcues = Lcues+30
    Rcues = Rcues+30
    
    if input_noise:
        Lkeep = np.random.uniform(0, 1, size = len(Lcues))
        Lcues[Lkeep<Inoise] =  500
        Rkeep = np.random.uniform(0, 1, size = len(Rcues))
        Rcues[Rkeep<Inoise] =  500
    
    
    def chain(y, t):
        dydt = -a*y+np.maximum(W@y+P(t, Pdivs)+scale*I(t, Lcues, Rcues)-T, 0)
        return dydt



    y0 = np.concatenate((Lchain, Rchain))
    
    t = np.linspace(0, 330, 3301)
    
    sol = odeint(chain, y0, t, hmax=5)
    return sol

#without noise in input
leftsol = np.zeros((3301, neurons*2))
rightsol = np.zeros((3301, neurons*2))
alldata = np.zeros((3301, neurons*2))
lchoices = []
rchoices = []
psychometric = {}
    
for t in range(len(trialdata)):
    print(t)
    Lcues = trialdata['leftcues'][t]
    Lcues = np.append(Lcues, 500)
    Rcues = trialdata['rightcues'][t]
    Rcues = np.append(Rcues, 500)
    sol = simulate(Lcues, Rcues)
    delta = len(Lcues)-len(Rcues)
    
    if len(Lcues)>len(Rcues):
        leftsol = np.dstack((leftsol, sol))
    
    if len(Rcues)>len(Lcues):
        rightsol = np.dstack((rightsol, sol))
    
    #collect all data to parallel neural analysis
    alldata = np.dstack((alldata, sol))
    
    wentleft = 1*(sol[-1, neurons-1]>sol[-1, 2*neurons-1])
    wentright = 1*(sol[-1, neurons-1]<sol[-1, 2*neurons-1])
    if wentleft:
        lchoices.append(t)
    if wentright:
        rchoices.append(t)
        
    if delta in psychometric:
        psychometric[delta].append(wentleft)
    else:
        psychometric[delta] = [wentleft]



psychometric.pop(0, 0)
cuediffs = sorted(psychometric.keys())
perf = [np.mean(psychometric[c]) for c in cuediffs]

#remove initialized zero array        
alldata = alldata[:, :, 1:]

#code for plotting sequences
pthresh = .1
region_threshold = .25
region_width = 4
base_thresh = 0

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


# =============================================================================
#     Parameters
#     ----------
#     alldata : neural data (trials x neuron x position)
#     leftchoices : trials in which the animal went left
#     rightchoices : trials in which the animal went right
# 
#     Returns
#     -------
#     indices of left preferrring neurons, indices of right preferring neurons, 
#     and indices of neurons with no significant difference in response between
#     the two choices
# =============================================================================

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


alldata = alldata.T #transpose for right dimensions for sequence plot


avgdata = np.mean(alldata[:, :, :], axis=0) #neurons x position

leftis, rightis, splitis, nonis = divide_LR(alldata, lchoices, rchoices, pthresh, region_threshold, region_width, base_thresh)

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

#resort data so always in sequence order
LCellLChoice = LCellLChoice[newLis, :]
LCellRChoice = LCellRChoice[newLis, :]

RCellLChoice = RCellLChoice[1:, :]
RCellRChoice = RCellRChoice[1:, :]
   
newRis = np.argsort(np.argmax((RCellRChoice+RCellLChoice)/2, axis=1))

#resort data so always in sequence order
RCellLChoice = RCellLChoice[newRis, :]
RCellRChoice = RCellRChoice[newRis, :]

SCellLChoice = SCellLChoice[1:, :]
SCellRChoice = SCellRChoice[1:, :]
   
newSis = np.argsort(np.argmax((SCellLChoice+SCellRChoice)/2, axis=1))

#resort data so always in sequence order
SCellLChoice = SCellLChoice[newSis, :]
SCellRChoice = SCellRChoice[newSis, :]

NCellLChoice = NCellLChoice[1:, :]
NCellRChoice = NCellRChoice[1:, :]
   
newNis = np.argsort(np.argmax((NCellLChoice+NCellRChoice)/2, axis=1))

#resort data so always in sequence order
NCellLChoice = NCellLChoice[newNis, :]
NCellRChoice = NCellRChoice[newNis, :]

#nonnormalized plots
fig, ax = plt.subplots(3, 2, gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [len(newLis), len(newRis), len(newSis)]})

VM = 12.5

ax[0, 0].imshow(LCellLChoice, cmap = 'Greys', aspect='auto', vmin = 0, vmax = VM)
ax[0, 0].set_title('left choice trials')
ax[0, 0].set_ylabel('left pref.')
ax[0,0].set_xticks([])
ax[0,0].set_yticks([])

ax[0, 1].imshow(LCellRChoice, cmap = 'Greys', aspect='auto', vmin = 0, vmax = VM)
ax[0, 1].set_title('right choice trials')
ax[0,1].set_xticks([])
ax[0,1].set_yticks([])

ax[1, 0].imshow(RCellLChoice, cmap = 'Greys', aspect='auto', vmin = 0, vmax = VM)
ax[1, 0].set_ylabel('right pref.')
ax[1,0].set_xticks([])
ax[1,0].set_yticks([])

ax[1, 1].imshow(RCellRChoice, cmap = 'Greys', aspect='auto', vmin = 0, vmax = VM)
ax[1,1].set_xticks([])
ax[1,1].set_yticks([])

ax[2, 0].imshow(SCellLChoice, cmap = 'Greys', aspect='auto', vmin = 0, vmax = VM)
ax[2, 0].set_ylabel('non-pref.')
ax[2,0].set_xticks([0, 300, 1300, 2300, 2800, 3300])
ax[2,0].set_xticklabels(['-30', '0', 'cues', '200', 'delay', '300'])
ax[2,0].set_xlabel('position (cm)')
ax[2,0].set_yticks([])

im = ax[2, 1].imshow(SCellRChoice, cmap = 'Greys', aspect='auto', vmin = 0, vmax = VM)
ax[2,1].set_xticks([0, 300, 1300, 2300, 2800, 3300])
ax[2,1].set_xticklabels(['-30', '0', 'cues', '200', 'delay', '300'])
ax[2,1].set_xlabel('position (cm)')
ax[2,1].set_yticks([])
cbar = fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.95)
plt.suptitle('NonNormalized Sequences')
plt.show()



#plot position gating input
times = np.linspace(-30, 300, 3301)
Pvals = np.zeros((len(times),2*neurons))
for i, t in enumerate(times):
    Pvals[i, :] = P(t+30, Pdivs)


plt.figure()
plt.plot(times, Pvals, color = 'cyan', label = 'i-1')
plt.axhline(T, color = 'gray', linestyle = ':')
plt.ylim([0, 320])
plt.show()