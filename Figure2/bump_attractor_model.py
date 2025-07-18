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

#load trial data on which to test the model, lists of left and right cues for each trial
trialdata = pd.read_pickle('ExampleData/trialdata.pkl')

#initialize neural ring
neurons = 35
timepoints = 1 #single ring for all positions
Cring = np.zeros((neurons*timepoints,))

#set parameters
a = 1.1
w0 = 3

#set up synaptic weights
def phi(i): #map each neuron to an angle
    return i*2*np.pi/neurons

def F(x): #activation function
    return .5*(1+np.tanh(x))

W = np.zeros((neurons*timepoints, neurons*timepoints)) #synaptic connection matrix

for i in range(neurons):
    for j in range(neurons):
        W[i, j] = w0*(np.cos(phi(i)-phi(j))-.9) #define weights between different neurons with cosine connectivity



def I(t, C, Lcues, Rcues):
    '''
    calculate external input to the ring
    t: position
    C: current state of the ring
    Lcues: positions of left cues
    Rcues: positions of right cues
    '''
    Lcue = np.abs(t-Lcues)
    if min(Lcue)<.5: #left cue only has an effect if within .5cm of cue onset
        IL = np.roll(C, -1) #input to each neuron from its left neighbor
    else:
        IL = np.zeros((neurons*timepoints,)) #no input from the left cue

    #same for right cues
    Rcue = np.abs(t-Rcues)
    if min(Rcue)<.5:
        IR = np.roll(C, 1)
    else:
        IR = np.zeros((neurons*timepoints,)) 
    return IL+IR #total external input from left and right cues

def correct(sol, Lcues, Rcues):
    '''
    returns whether the final solution gives the correct behavior for a given set of cues
    '''
    imax = np.argmax(sol[-1, :])
    if len(Lcues)>len(Rcues): #if there are more left cues than right cues, center of the peak should be left of the midpoint
        return imax<(neurons*timepoints-.5*neurons-1)
    return imax>(neurons*timepoints-.5*neurons)

def simulate(Lcues, Rcues):
    '''
    Simulate a single trial for a set of left and right cues
    '''

    #reset simulation
    Cring = np.zeros((neurons*timepoints,))
    
    #initialize bump of activity at the center
    Cring[16] = .6
    Cring[17] = .7
    Cring[18] = .6
    
    #remap cues to account for negative positions
    Lcues= Lcues+30
    Rcues = Rcues+30
   
   #differential equation for the bump attractor
    def ring(y, t):
        dydt = -a*y+(F(W@y+5*I(t, y, Lcues, Rcues)))
        return dydt
    
    y0 = Cring
    t = np.linspace(0, 330, 3301)
    
    #integrate the differential equation
    sol = odeint(ring, y0, t, hmax=5)
    return sol


#initialize arrays to save outputs from simulation
alldata = np.zeros((3301, neurons*timepoints))
lchoices = []
rchoices = []

#simulate each trial
for t in range(len(trialdata)):
    Lcues = trialdata['leftcues'][t]
    Lcues = np.append(Lcues, 500)
    Rcues = trialdata['rightcues'][t]
    Rcues = np.append(Rcues, 500)
    sol = simulate(Lcues, Rcues)
    
    #collect all data to parallel neural analysis
    alldata = np.dstack((alldata, sol))
    
    imax = np.argmax(sol[-1, :])
    wentleft = imax<(neurons*timepoints-.5*neurons-1)
    wentright = imax>(neurons*timepoints-.5*neurons)
    if wentleft:
        lchoices.append(t)
    if wentright:
        rchoices.append(t)

#remove initialized zero array        
alldata = alldata[:, :, 1:]

#method from Koay et al. 2022 for determining choice-selectivity of neurons
pthresh = .1
region_threshold = .25
region_width = 4
base_thresh = 0

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


#generate sequence plots for simulated data
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

#divide the data into left and right preferring cells
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

#normalize data
for i in range(len(newLis)):
    M = np.max([np.max(LCellLChoice[i, :]), np.max(LCellRChoice[i, :])])
    m = np.min([np.min(LCellLChoice[i, :]), np.min(LCellRChoice[i, :])])
    LCellLChoice[i, :] = (LCellLChoice[i, :]-m)/(M-m)
    LCellRChoice[i, :] = (LCellRChoice[i, :]-m)/(M-m)
    
for i in range(len(newRis)):
    M = np.max([np.max(RCellLChoice[i, :]), np.max(RCellRChoice[i, :])])
    m = np.min([np.min(RCellLChoice[i, :]), np.min(RCellRChoice[i, :])])
    RCellLChoice[i, :] = (RCellLChoice[i, :]-m)/(M-m)
    RCellRChoice[i, :] = (RCellRChoice[i, :]-m)/(M-m)

for i in range(len(newSis)):
    M = np.max([np.max(SCellLChoice[i, :]), np.max(SCellRChoice[i, :])])
    m = np.min([np.min(SCellLChoice[i, :]), np.min(SCellRChoice[i, :])])
    SCellLChoice[i, :] = (SCellLChoice[i, :]-m)/(M-m)
    SCellRChoice[i, :] = (SCellRChoice[i, :]-m)/(M-m)

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
plt.suptitle('Normalized Sequences')




