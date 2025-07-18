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
from sklearn.preprocessing import minmax_scale

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Arial']})

#load data for trials of lists of left and right cues for each trial
trialdata = pd.read_pickle('ExampleData/trialdata.pkl')

#parameters
a = 1 #decay rate
b = .2 #self-excitation
c = 0 #feedforward connections, none for this model
e = a-b #inhibition
baseline = 0 #starting condition
T = 300 #threshold
externalI = 40 #external signal

#initialize neural populations
neurons = 17

#set up connection matrix
W = b*np.identity(neurons*2) #synaptic connection matrix

for i in range(0, neurons):
    #feedfoward connections
    W[i, i-1] = c
    W[i+neurons, i-1+neurons] = c
    #inhibitory connections
    W[i+neurons, i] = -e
    W[i, i+neurons] = -e


def P(t):
    return T*np.ones((2*neurons,))+externalI #always receive baseline external signal at all timepoints

def I(t, Lcues, Rcues):
    '''
    calculate external input to each population
    t: position
    Lcues: positions of left cues
    Rcues: positions of right cues
    '''
    Lcue = np.abs(t-Lcues)
    if min(Lcue)<.5: #left cue only has an effect if within .5cm of cue onset
        IL = 4*np.ones((neurons,))
    else:
        IL = np.zeros((neurons,))

    #same for leftcues
    Rcue = np.abs(t-Rcues)
    if min(Rcue)<.5:
        IR = 4*np.ones((neurons,))
    else:
        IR = np.zeros((neurons,))    
    return np.concatenate((IL, IR)) #cue input to each population

def correct(sol, Lcues, Rcues):
    if len(Lcues)>len(Rcues): #if more left than right cues, correct if more activity in left population
        return sol[-1,16]>sol[-1, 33]
    return sol[-1, 16]<sol[-1, 33]

def simulate(Lcues, Rcues):
    #reset simulation
    Lchain = baseline*np.ones((neurons,))
    Rchain = baseline*np.ones((neurons,))
       
    #first 30 cm prior to cue onset
    Lcues = Lcues+30
    Rcues = Rcues+30
    
    #differential equation
    def chain(y, t):
        dydt = -a*y+np.maximum(W@y+P(t)+I(t, Lcues, Rcues)-T, 0)
        return dydt



    y0 = np.concatenate((Lchain, Rchain))
    
    t = np.linspace(0, 330, 3301)
    
    #integrate differential equation
    sol = odeint(chain, y0, t, hmax=1)
    return sol

#set up arrays to save data from simulations
leftsol = np.zeros((3301, neurons*2))
rightsol = np.zeros((3301, neurons*2))
lchoices = []
rchoices = []

#simulate each trial    
for t in range(len(trialdata)):
    Lcues = trialdata['leftcues'][t]
    Lcues = np.append(Lcues, 500)
    Rcues = trialdata['rightcues'][t]
    Rcues = np.append(Rcues, 500)
    sol = simulate(Lcues, Rcues)

    
    if len(Lcues)>len(Rcues):
        leftsol = np.dstack((leftsol, sol))
    
    if len(Rcues)>len(Lcues):
        rightsol = np.dstack((rightsol, sol))
    
    
    wentleft = 1*(sol[-1, 16]>sol[-1, 33])
    wentright = 1*(sol[-1, 16]<sol[-1, 33])
    if wentleft:
        lchoices.append(t)
    if wentright:
        rchoices.append(t)
        






#remove initialized zero array        
leftsol = leftsol[:, :, 1:]
rightsol = rightsol[:, :, 1:]


#get average activity
LCellLChoice = np.mean(leftsol, axis=2).T[:neurons] #left cells are the first half of the neurons
RCellLChoice = np.mean(leftsol, axis=2).T[neurons:] #right cells are the second half of the neurons
LCellRChoice = np.mean(rightsol, axis=2).T[:neurons]
RCellRChoice = np.mean(rightsol, axis=2).T[neurons:]

#normalize activity to [0,1]
for i in range(neurons):
    M = np.max([np.max(LCellLChoice[i, :]), np.max(LCellRChoice[i, :])])
    m = np.min([np.min(LCellLChoice[i, :]), np.min(LCellRChoice[i, :])])
    LCellLChoice[i, :] = (LCellLChoice[i, :]-m)/(M-m)
    LCellRChoice[i, :] = (LCellRChoice[i, :]-m)/(M-m)
    
for i in range(neurons):
    M = np.max([np.max(RCellLChoice[i, :]), np.max(RCellRChoice[i, :])])
    m = np.min([np.min(RCellLChoice[i, :]), np.min(RCellRChoice[i, :])])
    RCellLChoice[i, :] = (RCellLChoice[i, :]-m)/(M-m)
    RCellRChoice[i, :] = (RCellRChoice[i, :]-m)/(M-m)


fig, ax = plt.subplots(2, 2, gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [neurons, neurons]})

ax[0, 0].imshow(LCellLChoice, cmap = 'Greys', aspect='auto', vmin=.05, vmax=.95)
ax[0, 0].set_title('left choice trials')
ax[0, 0].set_ylabel('left pref.')
ax[0,0].set_xticks([])
ax[0,0].set_yticks([])

ax[0, 1].imshow(LCellRChoice, cmap = 'Greys', aspect='auto', vmin=.05, vmax=.95)
ax[0, 1].set_title('right choice trials')
ax[0,1].set_xticks([])
ax[0,1].set_yticks([])

ax[1, 0].imshow(RCellLChoice, cmap = 'Greys', aspect='auto', vmin=.05, vmax=.95)
ax[1, 0].set_ylabel('right pref.')
ax[1,0].set_xticks([])
ax[1,0].set_yticks([])
ax[1,0].set_xticks([0, 300, 1300, 2300, 2800, 3300])
ax[1,0].set_xticklabels(['-30', '0', 'cues', '200', 'delay', '300'])
ax[1,0].set_xlabel('position (cm)')

ax[1, 1].imshow(RCellRChoice, cmap = 'Greys', aspect='auto', vmin=.05, vmax=.95)
ax[1,1].set_xticks([])
ax[1,1].set_yticks([])
ax[1,1].set_xticks([0, 300, 1300, 2300, 2800, 3300])
ax[1,1].set_xticklabels(['-30', '0', 'cues', '200', 'delay', '300'])
ax[1,1].set_xlabel('position (cm)')




