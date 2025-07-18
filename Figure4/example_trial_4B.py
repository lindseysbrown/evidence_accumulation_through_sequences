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


#initialize neural rings
neurons = 35
timepoints = 17
Cring = np.zeros((neurons*timepoints,))

cmtos = 1

P0 = 20
T = 300*cmtos

a = 1.1*cmtos
b = .088
w0 = .12

#set up synaptic weights
def phi(i):
    return i*2*np.pi/neurons

def F(x):
    return 12.5*(1+np.tanh(x/cmtos))*cmtos

W = np.zeros((neurons*timepoints, neurons*timepoints))

bump = np.zeros((neurons, neurons))
for i in range(neurons):
    for j in range(neurons):
        bump[i, j] = w0*(np.cos(phi(i)-phi(j))-1) #w1=.9

W[:neurons,:neurons] = bump

for i in range(1, timepoints):
    W[neurons*i:neurons*(i+1), neurons*i:neurons*(i+1)]=bump
    
feedforward = np.zeros((neurons*timepoints, neurons*timepoints))
i, j = np.indices(W.shape)
feedforward[i==j+neurons] = a*b/cmtos

#position gating signal
def P(t):
    P = np.zeros(neurons*timepoints)
    i = int(np.floor(t/P0*cmtos))
    P[neurons*i:neurons*(i+1)]=T
    return P+.04

#input signal
def I(t, C, Lcues, Rcues):
    Lcue = np.abs(t-Lcues)
    if min(Lcue)<.5/cmtos:
        IL = np.roll(C, -1)
    else:
        IL = np.zeros((neurons*timepoints,))
    Rcue = np.abs(t-Rcues)
    if min(Rcue)<.5/cmtos:
        IR = np.roll(C, 1)
    else:
        IR = np.zeros((neurons*timepoints,)) 
    return (IL+IR)*cmtos

def correct(sol, Lcues, Rcues):
    imax = np.argmax(sol[-1, :])
    if len(Lcues)>len(Rcues):
        return imax<(neurons*timepoints-.5*neurons-1)
    return imax>(neurons*timepoints-.5*neurons)

#differential equation to simulate
def simulate(Lcues, Rcues, input_noise = False, Inoise = .67):
    #reset simulation
    Cring = np.zeros((neurons*timepoints,))
    
    Cring[16] = 15
    Cring[17] = 17.5
    Cring[18] = 15
    
    Lcues= Lcues+30/cmtos
    Rcues = Rcues+30/cmtos

    if input_noise:
        Lkeep = np.random.uniform(0, 1, size = len(Lcues))
        Lcues[Lkeep<Inoise] =  500
        Rkeep = np.random.uniform(0, 1, size = len(Rcues))
        Rcues[Rkeep<Inoise] =  500
    

    def ring(y, t):
        dydt = -a*y+(F(W@y+feedforward@y+.2*I(t, y, Lcues, Rcues)+P(t)-T))
        return dydt
    
    y0 = Cring
    t = np.linspace(0, 330/cmtos, 3301)
    
    sol = odeint(ring, y0, t, hmax=5/cmtos)
    return sol

#set up simulation
alldata = np.zeros((3301, neurons*timepoints))
    
for t in range(1): #simulate for a single example trial
    Lcues = [32, 55, 72] #fixed set of left cues
    Lcues = np.append(Lcues, 500)
    Rcues = [60] #fixed set of right cues
    Rcues = np.append(Rcues, 500)
    sol = simulate(Lcues/cmtos, Rcues/cmtos)
    delta = len(Lcues)-len(Rcues)

#range over which to plot sample trial data
start = 700
stop = 1100

times = np.linspace(-30, 300, 3301)

Ivals = np.zeros((len(times),timepoints*neurons))
Pvals = np.zeros((len(times),timepoints*neurons))
for i, t in enumerate(times):
    Ivals[i, :] = I(t, sol[i, :], Lcues, Rcues)
    Pvals[i, :] = P(t+30)

alldata = sol
evs = np.arange(-17, 18)

#plot position gate inputs
plt.figure()
plt.axvspan(40, 50, color = 'cyan', alpha = .2)
plt.axvspan(70, 80, color = 'cyan', alpha = .4)
plt.plot(times[start:stop], Pvals[:, 140][start:stop], color = 'cyan', label = 'i-1')
plt.plot(times[start:stop], Pvals[:, 105][start:stop], linestyle = 'dashdot', color = 'cyan', linewidth = 3, label = 'i')
plt.plot(times[start:stop], Pvals[:, 175][start:stop], linestyle = '--', color = 'cyan', linewidth = 2.5, label = 'i+1')
plt.legend()
#plt.axhline(T, color = 'gray', linestyle = ':')
plt.ylim([0, 310])
plt.xlim([40, 80])
plt.show()

range1 = alldata[start:stop, 112:132]
range2 = alldata[start:stop, 147: 167]
range3 = alldata[start:stop, 180: 200]
newdat = np.concatenate((range1, range2, range3), axis=1)

#plot activity of bump cells as heatmap
plt.figure()
plt.axvspan(0, 100, color = 'cyan', alpha = .2)
plt.axvspan(300, 400, color = 'cyan', alpha = .4)
plt.imshow(newdat.T, cmap = 'Greys', interpolation = 'none', vmin = 0, vmax = 15, origin = 'lower')
plt.xticks([0, 400], [40, 80])
plt.yticks([0, 20, 40, 60])
plt.axvline(150, color = 'orange', linestyle = '--')
plt.axvline(200, color = 'purple', linestyle = '--')
plt.axvline(320, color = 'orange', linestyle = '--')
plt.axhline(20, color = 'black', linewidth = .5)
plt.axhline(40, color = 'black', linewidth = .5)
plt.ylabel('Neuron')
plt.xlabel('position (cm)')
plt.xlim([0, 400])
plt.show()




