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

trialdata = pd.read_pickle('ExampleData/trialdata.pkl') #file containing data for set of trials with positions of left and right cues on each trial

#initialize neural rings
neurons = 35
timepoints = 17
Cring = np.zeros((neurons*timepoints,))

#set up parameters
P0 = 20 #postion width
T = 300 #threshold
X = 0.04 #external current
a = 1.1 #decay rate
w0 = 0.12 #parameterization of within layer cosine connectivity
w1 = -1 #parameterization of within layer cosine connectivity
w0p = 0.12 #parameterization for cosine connectivity across positions
w1p = -1 #parameterization for cosine connectivity across positions

#activation function
def F(x):
    return 25*.5*(1+np.tanh(x)) #q=25

#set up synaptic weights
def phi(i):
    return i*2*np.pi/neurons

def phip(i):
    return i*2*np.pi/(timepoints)

W = np.zeros((neurons*timepoints, neurons*timepoints)) #matrix of all synaptic weights

bump = np.zeros((neurons, neurons)) #matrix for synaptic weights within a layer, used for block structure
for i in range(neurons):
    for j in range(neurons):
        bump[i, j] = w0*(np.cos(phi(i)-phi(j))+w1)

W[:neurons,:neurons] = bump
for i in range(1, timepoints):
    W[neurons*i:neurons*(i+1), neurons*i:neurons*(i+1)]=bump


Wpos = np.zeros((neurons*timepoints, neurons*timepoints)) #position connections between neurons
posbump = np.zeros((timepoints, timepoints))
for i in range(timepoints):
    for j in range(timepoints):
       Wpos[(i*neurons):((i+1)*neurons), (j*neurons):((j+1)*neurons)] = w0p*(np.cos(phip(i)-phip(j))+w1p)
        # posbump[i, j] = w0p*(np.cos(phip(i)-phip(j))+w1p)


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

def Iv(t, C, v):
    '''
    calculate external input to the ring
    t: position
    C: current state of the ring
    v: velocity
    '''
    if v(t)<0: 
        I = np.roll(C, -35) #input to each neuron from its left neighbor
    elif v(t)>0:
        I = np.roll(C, 35)
    else:
        I = np.zeros((neurons*timepoints,)) 
    return np.abs(v(t)/50)*I

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

def simulate(Lcues, Rcues, v, input_noise = False, Inoise = .67):
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
        dydt = -a*y+(F(W@y+Wpos@y+.2*I(t, y, Lcues, Rcues)+.045*Iv(t, y, v))) 
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
deltas = []
peaks = []

def v(t):
    if t<150:
        return 1
    if t<225:
        return -1.5
    if t<300:
        return 0
    else:
        return 1
    
for t in range(len(trialdata)):#loop over all trials in trialdata
    Lcues = trialdata['leftcues'][t]
    Lcues = [22, 30, 50, 107, 135, 189]
    Lcues = np.append(Lcues, 500) #append cue to the end to avoid empty arrays
    Rcues = trialdata['rightcues'][t]
    Rcues = [55, 67]
    Rcues = np.append(Rcues, 500)
    sol = simulate(Lcues, Rcues, v)
    delta = len(Rcues)-len(Lcues) #final cue difference

    #collect all data
    alldata = np.dstack((alldata, sol))
    
    deltas.append(delta)
    peaks.append(np.argmax(alldata[-1, 560:, -1]))

plt.figure()
plt.scatter(deltas, peaks)
plt.xlabel('evidence')
plt.ylabel('bump location in final layer')

alldata = alldata[:, :, 1:]
alldata = alldata.T

np.save('jointbumpsol.npy', alldata) #save data for further analysis

