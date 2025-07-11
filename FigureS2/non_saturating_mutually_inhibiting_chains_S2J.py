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
scale=.5

#parameters
a = 1
b = .2
c = a
e = a-b
P0 = 20
baseline = 0
T = 300
externalI = 40*scale

m = .2
h = 4

#initialize neural rings
neurons = 17

#set up connection matrix
W = b*np.identity(neurons*2)

for i in range(1, neurons):
    #feedfoward connections
    W[i, i-1] = c
    W[i+neurons, i-1+neurons] = c
    #inhibitory connections
    W[i+neurons, i] = -e
    W[i, i+neurons] = -e


def P(t):
    pos = np.zeros((neurons,))
    i0 = int(np.floor(t/P0))
    pos[i0] = T+externalI
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
    
    #Lcues=np.array([500])
    #Rcues = np.array([500])
    
    def chain(y, t):
        dydt = -a*y+np.maximum(W@y+P(t)+scale*I(t, Lcues, Rcues)-T, 0)
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
    #Lcues = [10, 20, 35, 46, 55, 78, 89, 93, 100, 106, 111, 150]
    Lcues = np.append(Lcues, 500)
    Rcues = trialdata['rightcues'][t]
    #Rcues = []
    Rcues = np.append(Rcues, 500)
    sol = simulate(Lcues, Rcues)
    delta = len(Lcues)-len(Rcues)
    
    #collect all data to parallel neural analysis
    alldata = np.dstack((alldata, sol))



alldata = alldata[:, :, 1:]

np.save('nonsaturatingMIsol.npy', alldata)
