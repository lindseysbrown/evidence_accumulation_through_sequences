# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 11:53:49 2022

@author: lindseyb
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Arial']})

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


#initialize neural rings
neurons = 17

#set up connection matrix
W = b*np.identity(neurons*2)

for i in range(1, neurons):
    #feedfoward connections
    W[i, i-1] = c
    W[i+neurons, i-1+neurons] = c

for i in range(neurons):
    #inhibitory connections
    W[i+neurons, i] = -e
    W[i, i+neurons] = -e

#position gating signal
def P(t):
    pos = np.zeros((neurons,))
    i0 = int(np.floor(t/P0))
    pos[i0] = T+externalI
    return np.concatenate((pos, pos))

#input signal
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

#simulate differential equation
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
        dydt = -a*y+np.maximum(W@y+P(t)+scale*I(t, Lcues, Rcues)-T, 0)
        return dydt



    y0 = np.concatenate((Lchain, Rchain))
    
    t = np.linspace(0, 330, 3301)
    
    sol = odeint(chain, y0, t, hmax=5)
    return sol

#without noise in input
alldata = np.zeros((3301, neurons*2))
lchoices = []
rchoices = []
psychometric = {}
    
for t in range(1): #simulate a single example trial
    Lcues = [32, 55, 72] #fixed set of left cues
    Lcues = np.append(Lcues, 500)
    Rcues = [60] #fixed set of right cues
    Rcues = np.append(Rcues, 500)
    sol = simulate(Lcues, Rcues)
    delta = len(Lcues)-len(Rcues)
    
    
    #collect all data to parallel neural analysis
    alldata = np.dstack((alldata, sol))
    
    wentleft = 1*(sol[-1, 16]>sol[-1, 33])
    wentright = 1*(sol[-1, 16]<sol[-1, 33])
    if wentleft:
        lchoices.append(t)
    if wentright:
        rchoices.append(t)
        
    if delta in psychometric:
        psychometric[delta].append(wentleft)
    else:
        psychometric[delta] = [wentleft]

#define range over which to plot data
start = 700
stop = 1100

#remove initialized zero array        
alldata = alldata[:, :, 1:]

times = np.linspace(-30, 300, 3301)
Ivals = np.zeros((len(times),2*neurons))
Pvals = np.zeros((len(times),2*neurons))
for i, t in enumerate(times):
    Ivals[i, :] = scale*I(t, Lcues, Rcues)
    Pvals[i, :] = P(t+30)

#plot activity of individual neurons
plt.figure()
plt.axvspan(40, 50, color = 'cyan', alpha = .2)
plt.axvspan(70, 80, color = 'cyan', alpha = .4)
plt.plot(times[start:stop], alldata[:, 3][start:stop], color = 'blue', linestyle = 'dashdot')
plt.plot(times[start:stop], alldata[:, 4][start:stop], color = 'blue')
plt.plot(times[start:stop], alldata[:, 5][start:stop], color = 'blue', linestyle = '--')
plt.plot(times[start:stop], alldata[:, 20][start:stop], color = 'red', linestyle = 'dashdot')
plt.plot(times[start:stop], alldata[:, 21][start:stop], color = 'red')
plt.plot(times[start:stop], alldata[:, 22][start:stop], color = 'red', linestyle = '--')
plt.axvline(55, color = 'orange', linestyle = '--')
plt.axvline(60, color = 'purple', linestyle = '--')
plt.axvline(72, color = 'orange', linestyle = '--')
plt.xlim([40, 80])
plt.ylim([0, 10])
plt.show()


#plot input current values
plt.figure()
plt.axvspan(40, 50, color = 'cyan', alpha = .2)
plt.axvspan(70, 80, color = 'cyan', alpha = .4)
plt.plot(times[start:stop], Ivals[:, 0][start:stop], color = 'orange')
plt.plot(times[start:stop], Ivals[:, 17][start:stop], color = 'purple')
plt.axvline(55, color = 'orange', linestyle = '--')
plt.axvline(60, color = 'purple', linestyle = '--')
plt.axvline(72, color = 'orange', linestyle = '--')
plt.xlim([40, 80])
plt.ylim([0, 2.1])
plt.show()


#plot position gate values
plt.figure()
plt.axvspan(40, 50, color = 'cyan', alpha = .2)
plt.axvspan(70, 80, color = 'cyan', alpha = .4)
plt.plot(times[start:stop], Pvals[:, 4][start:stop], color = 'cyan', label = 'i-1')
plt.plot(times[start:stop], Pvals[:, 3][start:stop], linestyle = 'dashdot', color = 'cyan', linewidth = 3, label = 'i')
plt.plot(times[start:stop], Pvals[:, 5][start:stop], linestyle = '--', color = 'cyan', linewidth = 2.5, label = 'i+1')
plt.legend()
plt.axhline(T, color = 'gray', linestyle = ':')
plt.ylim([0, 320])
plt.xlim([40, 80])
plt.show()
