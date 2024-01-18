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

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Arial']})

trialdata = pd.read_pickle('trialdata.pkl') #file containing data for set of trials with positions of left and right cues on each trial

#initialize neural rings
neurons = 35
timepoints = 17
Cring = np.zeros((neurons*timepoints,))

#set up parameters
P0 = 20 #postion width
T = 300 #threshold
a = 1.1 #decay rate
b = 0.088 #feedforward synaptic connections
w0 = 0.12 #parameterization of within layer cosine connectivity
w1 = -0.9 #parameterization of within layer cosine connectivity

#activation function
def F(x):
    return 25*.5*(1+np.tanh(x)) #q=25

#set up synaptic weights
def phi(i):
    return i*2*np.pi/neurons

W = np.zeros((neurons*timepoints, neurons*timepoints)) #matrix of all synaptic weights

bump = np.zeros((neurons, neurons)) #matrix for synaptic weights within a layer, used for block structure
for i in range(neurons):
    for j in range(neurons):
        bump[i, j] = w0*(np.cos(phi(i)-phi(j))+w1)

W[:neurons,:neurons] = bump

for i in range(1, timepoints):
    W[neurons*i:neurons*(i+1), neurons*i:neurons*(i+1)]=bump
    
feedforward = np.zeros((neurons*timepoints, neurons*timepoints)) #feedforward connections between neurons
i, j = np.indices(W.shape)
feedforward[i==j+neurons] = b

#position gating signal
def P(t):
    P = np.zeros(neurons*timepoints)
    i = int(np.floor(t/P0))
    P[neurons*i:neurons*(i+1)]=T #assign all neurons at position layer to threshold level
    return P

#external input function
def I(t, C, Lcues, Rcues):
    '''
    === inputs ===
    t: current position
    C: activity level of cells in layer
    Lcues: list of positions with left towers
    Rcues: list of positions with right towers
    '''
    Lcue = np.abs(t-Lcues)
    if min(Lcue)<.5: #positive 1_left signal if cue within .5cm
        IL = np.roll(C, -1) #multiply by activity of evidence accumulation cells, shifted to get shifter neuron activity
    else:
        IL = np.zeros((neurons*timepoints,))
    Rcue = np.abs(t-Rcues) 
    if min(Rcue)<.5: 
        IR = np.roll(C, 1) #same for right cues but opposite shift
    else:
        IR = np.zeros((neurons*timepoints,)) 
    return IL+IR #input sum of right and left input

def correct(sol, Lcues, Rcues):
    '''
    determine whether animal makes correct decision depending on location of the peak

    === inputs ===
    sol: integrated solution
    Lcues: list of positions of left cues
    Rcues: list of positions of right cues
    '''
    imax = np.argmax(sol[-1, :]) #neuron with maximal activity at the last timepoint
    if len(Lcues)>len(Rcues):
        return imax<(neurons*timepoints-.5*neurons-1) #neuron with maximal activity should be to the left of center on trial with more left cues
    return imax>(neurons*timepoints-.5*neurons)

def simulate(Lcues, Rcues, input_noise = False, Inoise = .67):
    '''
    simulate a trial for a set of left cue positions (Lcues) and right cue positions (Rcues)
    '''
    #reset simulation
    Cring = np.zeros((neurons*timepoints,))
    
    #initialize bump to the center of the first position layer
    Cring[16] = 15
    Cring[17] = 17.5
    Cring[18] = 15
    
    #add onto cues to account for 30cm precue region
    Lcues= Lcues+30
    Rcues = Rcues+30

    if input_noise: #if considering noise in input, randomly remove cues with probabiliy given by Inoise
        Lkeep = np.random.uniform(0, 1, size = len(Lcues))
        Lcues[Lkeep<Inoise] =  500
        Rkeep = np.random.uniform(0, 1, size = len(Rcues))
        Rcues[Rkeep<Inoise] =  500
    

    def ring(y, t):
        '''
        differential equation for evolution of the position gated bump attractor from Eq. (2)
        '''
        dydt = -a*y+np.heaviside(P(t)-T, 1)*(F(W@y+feedforward@y+.2*I(t, y, Lcues, Rcues)))
        return dydt
    
    y0 = Cring
    t = np.linspace(0, 330, 3301)
    
    sol = odeint(ring, y0, t, hmax=5) #numerically integrate the solution using scipy.odeint
    return sol

#without noise in input
alldata = np.zeros((3301, neurons*timepoints))
lchoices = []
rchoices = []
psychometric = {}
    
for t in range(len(trialdata)): #loop over all trials in trialdata
    Lcues = trialdata['leftcues'][t]
    Lcues = np.append(Lcues, 500) #append cue to the end to avoid empty arrays
    Rcues = trialdata['rightcues'][t]
    Rcues = np.append(Rcues, 500)
    sol = simulate(Lcues, Rcues)
    delta = len(Rcues)-len(Lcues) #final cue difference
    
    #collect all data
    alldata = np.dstack((alldata, sol))
    
    #keep track of trials for which left or right decisions were made
    wentleft = 1*(sol[-1, 16]>sol[-1, 33])
    wentright = 1*(sol[-1, 16]<sol[-1, 33])
    if wentleft:
        lchoices.append(t) 
    if wentright:
        rchoices.append(t)

    #track whether the final decision was left or right for different differences in cues    
    if delta in psychometric:
        psychometric[delta].append(wentleft)
    else:
        psychometric[delta] = [wentleft]


#calculate pyschometric curve
psychometric.pop(0, 0)
cuediffs = sorted(psychometric.keys())
perf = [np.mean(psychometric[c]) for c in cuediffs]

#remove initialized zero array        
alldata = alldata[:, :, 1:]

#save all data for use in future analysis
np.save('bumpmodel.npy', alldata)

#code for getting sequence plots identical to neural data
pthresh = .1
region_threshold = .25 #.5 Ryan
region_width = 4
base_thresh = 0 #3 Ryan

def get_regions(data, M, threshold, width, base_thresh):
    ''''
    identify peaks in neural data defined as regions with activity of at least threshold*M, where M is the maximum of average firing at each position, and width of at least width
    '''
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
            leftactivity = np.mean(np.round(leftfiring, 3), axis=1)
            rightfiring = rightfires[:, i, region]
            rightactivity = np.mean(np.round(rightfiring, 3), axis=1)
            tval, pval = stats.ttest_ind(leftactivity, rightactivity)
            if pval <2*pthresh:
                if np.mean(leftactivity)>np.mean(rightactivity):
                    left = True
                else:
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

#nonnormalized plots
fig, ax = plt.subplots(3, 2, gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [len(newLis), len(newRis), len(newSis)]})

ax[0, 0].imshow(LCellLChoice, cmap = 'Greys', aspect='auto')
ax[0, 0].set_title('left choice trials')
ax[0, 0].set_ylabel('left pref.')
ax[0,0].set_xticks([])
ax[0,0].set_yticks([])

ax[0, 1].imshow(LCellRChoice, cmap = 'Greys', aspect='auto')
ax[0, 1].set_title('right choice trials')
ax[0,1].set_xticks([])
ax[0,1].set_yticks([])

ax[1, 0].imshow(RCellLChoice, cmap = 'Greys', aspect='auto')
ax[1, 0].set_ylabel('right pref.')
ax[1,0].set_xticks([])
ax[1,0].set_yticks([])

ax[1, 1].imshow(RCellRChoice, cmap = 'Greys', aspect='auto')
ax[1,1].set_xticks([])
ax[1,1].set_yticks([])

ax[2, 0].imshow(SCellLChoice, cmap = 'Greys', aspect='auto')
ax[2, 0].set_ylabel('non-pref.')
ax[2,0].set_xticks([0, 300, 1300, 2300, 2800, 3300])
ax[2,0].set_xticklabels(['-30', '0', 'cues', '200', 'delay', '300'])
ax[2,0].set_xlabel('position (cm)')
ax[2,0].set_yticks([])

ax[2, 1].imshow(SCellRChoice, cmap = 'Greys', aspect='auto')
ax[2,1].set_xticks([0, 300, 1300, 2300, 2800, 3300])
ax[2,1].set_xticklabels(['-30', '0', 'cues', '200', 'delay', '300'])
ax[2,1].set_xlabel('position (cm)')
ax[2,1].set_yticks([])
plt.suptitle('NonNormalized Sequences')
plt.show()

    




