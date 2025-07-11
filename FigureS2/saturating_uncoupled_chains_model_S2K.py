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
a = 1 #rate of decay and self-excitation
b = a #feedforward connections
Plength = 20

T = 300

#initialize neural rings
neurons = 17

#set up connection matrix
W = a*np.identity(neurons*2)

for i in range(1, neurons):
    #feedfoward connections
    W[i, i-1] = b
    W[i+neurons, i-1+neurons] = b

#position gating signal
def P(t):
    '''
    Position gating signal, equal to T when neuron is active, assuming constant velocity
    '''
    pos = np.zeros((neurons,))
    pos[int(np.floor(t/Plength))] = T
    return np.concatenate((pos, pos))

def I(t, Lcues, Rcues):
    '''
    External input drive from cues, that contribute a square pulse if a cue occurs within .5cm of the current position

    ===INPUTS===
    t: current position
    Lcues: list of positions at which left towers occur
    Rcues: list of positions at which right towers occur
    '''
    Lcue = np.abs(t-Lcues)
    if min(Lcue)<.5:
        IL = np.ones((neurons,))
    else:
        IL = np.zeros((neurons,))
    Rcue = np.abs(t-Rcues)
    if min(Rcue)<.5:
        IR = np.ones((neurons,))
    else:
        IR = np.zeros((neurons,))    
    return np.concatenate((IL-IR, IR-IL))

def correct(sol, Lcues, Rcues):
    '''
    Returns boolean of whether the final neuron in the chain with more inputs had greater firing rate
    '''
    if len(Lcues)>len(Rcues):
        return sol[-1,16]>sol[-1, 33]
    return sol[-1, 16]<sol[-1, 33]

def simulate(Lcues, Rcues, input_noise = False, Inoise = .67):
    '''
    Simulate a single trial for a set of input towers

    ===INPUTS===
    Lcues: list of positions at which left towers occur
    Rcues: list of positions at which right towers occur

    '''
    #reset simulation
    Lchain = np.zeros((neurons,))
    Rchain = np.zeros((neurons,))
    
    Lchain[0] = 8
    Rchain[0] = 8
    
    Lcues= Lcues+30
    Rcues = Rcues+30
    
    if input_noise: #option for not integrating some cues in the case of input noise
        Lkeep = np.random.uniform(0, 1, size = len(Lcues))
        Lcues[Lkeep<Inoise] =  500
        Rkeep = np.random.uniform(0, 1, size = len(Rcues))
        Rcues[Rkeep<Inoise] =  500

    def chain(y, t): #differntial equation
        dydt = -a*y+np.maximum(W@y+P(t)+I(t, Lcues, Rcues)-T, 0) #Eq. (S2)
        saturated = (y>13) & (dydt>0) #test if firing rate above 13
        dydt[saturated] = 0 #anywhere that is saturated with positive derivative, does not increase further
        return dydt

    y0 = np.concatenate((Lchain, Rchain))
    
    t = np.linspace(0, 330, 3301)
    
    sol = odeint(chain, y0, t, hmax=5) #numerically integrate with scipy's odeint
    return sol

#without noise in input
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
    
    #collect all data to parallel neural analysis
    alldata = np.dstack((alldata, sol))



alldata = alldata[:, :, 1:]

#save solution for future analysis
np.save('saturatinginputssol.npy', alldata)
