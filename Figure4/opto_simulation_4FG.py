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
optostrength = .2 #strength of optogenetic excitation

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

def O(t, i):
    opto = np.zeros((neurons*timepoints,))
    opto[i] = optostrength
    return opto

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

def simulate(Lcues, Rcues, optoi, input_noise = False, Inoise = .67):
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
        dydt = -a*y+np.heaviside(P(t)-T, 1)*(F(W@y+feedforward@y+5*I(t, y, Lcues, Rcues))) #differential equation without optogenetic excitation
        return dydt
    
    def optoring(y, t):
        dydt = -a*y+np.heaviside(P(t)-T, 1)*(F(W@y+feedforward@y+I(t, y, Lcues, Rcues)+O(t, optoi))) #modified differential equation with optogenetic excitation
        return dydt
    
    y0 = Cring
    t = np.linspace(0, 330, 3301)
    
    sol = odeint(ring, y0, t, hmax=5)
    optosol = odeint(optoring, y0, t, hmax=5)
    
    return sol, optosol

#code for getting sequence plots, based on method from Koay et al. (equivalent to left vs. right chains in model but more general for neural data)
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


alldata = np.load('bumpmodel.npy') #load model simulated data
lchoices = np.load('lchoices.npy')
rchoices = np.load('rchoices.npy')

#use original simulation to sort cells into right and left preferring
leftis, rightis, splitis, nonis = divide_LR(alldata, lchoices, rchoices, pthresh, region_threshold, region_width, base_thresh)

maxchanges = np.zeros((len(leftis), neurons*timepoints))

#for each left preferring cell, simulate model with excitation to that cell
for i, l in enumerate(leftis):
    Lcues = np.array([500])
    Rcues = np.array([500])
    sol, optosol = simulate(Lcues, Rcues, l)

    delta = optosol-sol #difference in activity with and without stimulation
    
    increases = np.max(delta, axis=0) #find maximum positive change
    decreases = np.min(delta, axis=0) #find maximum negative change
    
    maxchanges[i] = increases+decreases

evidencechanges = maxchanges.copy()
with open('bumpetuning.pkl', 'rb') as handle: #load dictionary of neuron to preferred evidence level
    evtuning = pickle.load(handle)

#find neurons at each evidence level to determine changes by evidence level
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
    ltier = np.floor(l/35)*35 #find which position layer the stimulated neuron is in
    for e, evval in enumerate(evlevels):
        nlist = np.array(neuronsatev[evval])
        nlist = nlist[nlist>ltier] #only average neural activity at later positions than the stimulated neuron
        evavgchange[i, e] = np.mean(maxchanges[i, nlist])
        
#reorder data by choice preference
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

leftresp = maxchangesleft[78, :]
rightresp = maxchangesright[78, :]
changes = np.concatenate((leftresp, rightresp))

#plot individual neuron changes for all neurons, sorted by position, then sorted by evidence level within the position    
plt.figure()
plt.bar(np.arange(len(changes)), changes, color = 'k', align = 'edge', width=1)
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
plt.show()

#inset
plt.figure()
plt.bar(np.arange(len(changes[99:116])), changes[99:116], color = 'k', align='edge', width=1) #subset of cells within same position, sorted by evidence
plt.show()

#plot responses at each evidence level
echanges2 = evavgchange[78, :]
ecolors2 = []
for c in echanges2:
    if c<0:
        ecolors2.append('blue')
    else:
        ecolors2.append('red')

plt.figure()
plt.bar(-1*np.array(evlevels), echanges2, color = 'k', align = 'edge', width=1)
plt.ylim([-.75, 1.25])
plt.ylabel('change from baseline', fontsize=24)
plt.xlabel('evidence', fontsize = 24)
plt.show()







