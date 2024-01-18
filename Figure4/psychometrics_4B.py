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
from scipy.stats import sem

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Arial']})

trialdata = pd.read_pickle('trialdata.pkl')

#initialize neural rings
neurons = 35
timepoints = 17
Cring = np.zeros((neurons*timepoints,))

P0 = 20
T = 300

a = 1.1
b=2
w0 = 3

#set up synaptic weights
def phi(i):
    return i*2*np.pi/neurons

def F(x):
    return .5*(1+np.tanh(x))

W = np.zeros((neurons*timepoints, neurons*timepoints))

bump = np.zeros((neurons, neurons))
for i in range(neurons):
    for j in range(neurons):
        bump[i, j] = w0*(np.cos(phi(i)-phi(j))-.9)

W[:neurons,:neurons] = bump

for i in range(1, timepoints):
    W[neurons*i:neurons*(i+1), neurons*i:neurons*(i+1)]=bump
    
feedforward = np.zeros((neurons*timepoints, neurons*timepoints))
i, j = np.indices(W.shape)
feedforward[i==j+neurons] = a*b

def P(t):
    P = np.zeros(neurons*timepoints)
    i = int(np.floor(t/P0))
    P[neurons*i:neurons*(i+1)]=T
    return P

def I(t, C, Lcues, Rcues):
    Lcue = np.abs(t-Lcues)
    if min(Lcue)<.5:
        IL = np.roll(C, -1)
    else:
        IL = np.zeros((neurons*timepoints,))
    Rcue = np.abs(t-Rcues)
    if min(Rcue)<.5:
        IR = np.roll(C, 1)
    else:
        IR = np.zeros((neurons*timepoints,)) 
    return IL+IR

def correct(sol, Lcues, Rcues):
    imax = np.argmax(sol[-1, :])
    if len(Lcues)>len(Rcues):
        return imax<(neurons*timepoints-.5*neurons-.5)
    return imax>(neurons*timepoints-.5*neurons)

def simulate(Lcues, Rcues, input_noise = False, Inoise = .67):
    #reset simulation
    Cring = np.zeros((neurons*timepoints,))
    
    Cring[16] = .6
    Cring[17] = .7
    Cring[18] = .6
    
    Lcues= Lcues+30
    Rcues = Rcues+30

    if input_noise:
        Lkeep = np.random.uniform(0, 1, size = len(Lcues))
        Lcues[Lkeep<Inoise] =  500
        Rkeep = np.random.uniform(0, 1, size = len(Rcues))
        Rcues[Rkeep<Inoise] =  500
    

    def ring(y, t):
        dydt = -a*y+np.heaviside(P(t)-T, 1)*(F(W@y+feedforward@y+5*I(t, y, Lcues, Rcues)))
        return dydt
    
    y0 = Cring
    t = np.linspace(0, 330, 3301)
    
    sol = odeint(ring, y0, t, hmax=5)
    return sol

def rebin(p):
    pnew= {}
    for i in range(-14, 14, 3):
        pnew[i] = []
        try:
            pnew[i] = pnew[i]+p[i-1]
        except:
            pnew[i] = pnew[i]
        try:
            pnew[i] = pnew[i]+p[i]
        except:
            pnew[i] = pnew[i]
        try:
            pnew[i] = pnew[i]+p[i+1]
        except:
            pnew[i] = pnew[i]
        if len(pnew[i])==0:
            pnew.pop(i)
    return pnew

alldata = np.load('bumpmodel-full-feedforwardinside-faster.npy')
perfnonoise = {}

for t in range(len(trialdata)):
    Lcues = trialdata['leftcues'][t]
    Rcues = trialdata['rightcues'][t]
    sol = alldata[t, :, :].T
    delta = len(Lcues)-len(Rcues)
       

    imax = np.argmax(sol[-1, -1*neurons:])
    wentleft = imax<(.5*neurons-1)
    wentright = imax>(.5*neurons)
    
    if delta ==1:
        plt.figure()
        plt.plot(sol[-1, -1*neurons:])
        plt.axvline(17, c = 'grey', linestyle = '--')
    
    if delta in perfnonoise:
        perfnonoise[delta].append(wentleft)
    else:
        perfnonoise[delta] = [wentleft]

perfnonoise.pop(0)
cuediffsnonoise = sorted(perfnonoise.keys())
perf_nonoise = [np.mean(perfnonoise[c]) for c in cuediffsnonoise]


#with noise in input
psychometrics = []

for i in range(25):
    psychometricerror = {}
    
    samples = np.random.choice(np.arange(1000), 150)
        
    for t in samples:
        Lcues = trialdata['leftcues'][t]
        Lcues = np.append(Lcues, 500)
        Rcues = trialdata['rightcues'][t]
        Rcues = np.append(Rcues, 500)
        sol = simulate(Lcues, Rcues, input_noise = True)
        delta = len(Lcues)-len(Rcues)
        
        
        imax = np.argmax(sol[-1, :])
        wentleft = imax<(neurons*timepoints-.5*neurons-.5)
        wentright = imax>(neurons*timepoints-.5*neurons-.5)
        if (not wentleft and not wentright):
            r = np.random.rand()
            wentleft = r<.5
        if delta in psychometricerror:
            psychometricerror[delta].append(wentleft)
        else:
            psychometricerror[delta] = [wentleft]
    
    psychometricerror.pop(0, 0)
    psychometrics = psychometrics + [psychometricerror]

rebinpsychometrics = [rebin(p) for p in psychometrics]
perfs = []
for p in rebinpsychometrics:
    cuediffs = sorted(p.keys())
    perf = [(c, np.mean(p[c])) for c in cuediffs] 
    perfs = perfs+perf
performance = {}
for a in perfs:
    if a[0] in performance:
        performance[a[0]].append(a[1])
    else:
        performance[a[0]] = [a[1]]
cuediffs = sorted(performance.keys())
meanperf = [np.mean(performance[c]) for c in cuediffs]
sdperf = [sem(performance[c]) for c in cuediffs]

plt.figure()
plt.plot(-1*np.array(cuediffsnonoise), 1-np.array(perf_nonoise), color='black', label = 'No Noise')
plt.errorbar(-1*np.array(cuediffs), 1-np.array(meanperf), yerr = sdperf, color = 'red', label = 'Input Noise')
plt.xlabel('#R - #L')
plt.ylabel('Model Performance')
plt.ylim([0,1])
plt.xlim([-15, 15])
plt.axvline(x=0, color = 'grey', linestyle = 'dashed') 
plt.axhline(y=0.5, color = 'grey', linestyle = 'dashed')  
plt.legend()
plt.savefig('LRSwitch-PsychometricPlanarBump-FeedforwardInside.pdf', transparent = True)

