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
from scipy.stats import sem

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Arial']})


trialdata = pd.read_pickle('trialdata.pkl')

#parameters
a = 1
b = .2
c = a
e = a-b
P0 = 20
baseline = 0
T = 300
externalI = 20
optostrength = 1

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
        dydt = -a*y+np.maximum(W@y+P(t)+I(t, Lcues, Rcues)-T, 0)
        return dydt



    y0 = np.concatenate((Lchain, Rchain))
    
    t = np.linspace(0, 330, 3301)
    
    sol = odeint(chain, y0, t, hmax=5)
    return sol

#without noise in input
#leftsol = np.zeros((3301, neurons*2))
#rightsol = np.zeros((3301, neurons*2))
#alldata = np.zeros((3301, neurons*2))
#lchoices = []
#rchoices = []
psychometric = {}
    
for t in range(len(trialdata)):
    Lcues = trialdata['leftcues'][t]
    Lcues = np.append(Lcues, 500)
    Rcues = trialdata['rightcues'][t]
    Rcues = np.append(Rcues, 500)
    sol = simulate(Lcues, Rcues)
    delta = len(Lcues)-len(Rcues)
    
    #if len(Lcues)>len(Rcues):
     #   leftsol = np.dstack((leftsol, sol))
    
    #if len(Rcues)>len(Lcues):
     #   rightsol = np.dstack((rightsol, sol))
    
    #collect all data to parallel neural analysis
    #alldata = np.dstack((alldata, sol))
    
    wentleft = 1*(sol[-1, 16]>sol[-1, 33])
    #wentright = 1*(sol[-1, 16]<sol[-1, 33])
    #if wentleft:
     #   lchoices.append(t)
    #if wentright:
     #   rchoices.append(t)
        
    if delta in psychometric:
        psychometric[delta].append(wentleft)
    else:
        psychometric[delta] = [wentleft]



psychometric.pop(0, 0)
cuediffsnonoise = sorted(psychometric.keys())
perfnonoise = [np.mean(psychometric[c]) for c in cuediffsnonoise]

#remove initialized zero array        
#leftsol = leftsol[:, :, 1:]
#rightsol = rightsol[:, :, 1:]
#alldata = alldata[:, :, 1:]



#plt.figure()
#plt.imshow(np.mean(rightsol,axis=2).T, aspect = 'auto', cmap = 'Greys', origin='lower')
#plt.title('Right Choice Trials')
#plt.xlabel('Position')
#plt.ylabel('Neuron')
#plt.savefig('SeqPlotRightCompetingChains.pdf', transparent=True)

#plt.figure()
#plt.imshow(np.mean(leftsol,axis=2).T, aspect = 'auto', cmap = 'Greys', origin='lower')
#plt.title('Left Choice Trials')
#plt.xlabel('Position')
#plt.ylabel('Neuron')
#plt.savefig('SeqPlotLeftCompetingChains.pdf', transparent=True)



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
        
        wentleft = 1*(sol[-1, 16]>sol[-1, 33])    
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
plt.plot(cuediffsnonoise, perfnonoise, color='black', label = 'No Noise')
plt.errorbar(cuediffs, meanperf, yerr = sdperf, color = 'red', label = 'Input Noise')
plt.ylim([0,1])
plt.xlim([-15, 15])
plt.axvline(x=0, color = 'grey', linestyle = 'dashed') 
plt.axhline(y=0.5, color = 'grey', linestyle = 'dashed')  
plt.xlabel('#L - #R')
plt.ylabel('Model Performance')
plt.legend()
plt.savefig('Oct3MIPsychometricCompetingChains.pdf', transparent = True)
