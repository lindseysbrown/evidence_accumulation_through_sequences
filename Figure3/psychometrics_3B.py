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


trialdata = pd.read_pickle('trialdata.pkl') #file containing data for set of trials with positions of left and right cues on each trial

#parameters
a = 1 #decay rate
b = .2 #self-excitation
c = a #feedforward excitation
e = a-b #mutual inhibition
P0 = 20 #position width
baseline = 0 #starting value
T = 300 #threshold
externalI = 10 #external drive

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
    '''
    Position gating signal, equal to T+externalI when neuron is active, assuming constant velocity
    '''
    pos = np.zeros((neurons,))
    i0 = int(np.floor(t/P0)) #determine which neuron is currently active
    pos[i0] = T+externalI
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
        IL = 2*np.ones((neurons,)) #f=2, strength of synaptic connection
    else:
        IL = np.zeros((neurons,))
    Rcue = np.abs(t-Rcues)
    if min(Rcue)<.5:
        IR = 2*np.ones((neurons,))
    else:
        IR = np.zeros((neurons,))    
    return np.concatenate((IL, IR)) #concatenate inputs to left chain and inputs to right chain

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
    
    #initialize chains to baseline level
    Lchain[0] = baseline
    Rchain[0] = baseline
   
    #first 30 cm prior to cue onset
    Lcues = Lcues+30
    Rcues = Rcues+30
    
    if input_noise: #option for not integrating some cues in the case of input noise
        Lkeep = np.random.uniform(0, 1, size = len(Lcues))
        Lcues[Lkeep<Inoise] =  500
        Rkeep = np.random.uniform(0, 1, size = len(Rcues))
        Rcues[Rkeep<Inoise] =  500
    
    def chain(y, t): #differential equation, Eq. (1)
        dydt = -a*y+np.maximum(W@y+P(t)+I(t, Lcues, Rcues)-T, 0)
        return dydt

    y0 = np.concatenate((Lchain, Rchain))
    
    t = np.linspace(0, 330, 3301)
    
    sol = odeint(chain, y0, t, hmax=5) #numerically integrate with scipy's odeint
    return sol

#get psychometric curve for 1000 trials with no noise in input
psychometric = {}    
for t in range(len(trialdata)): #loop over all trials in trialdata
    Lcues = trialdata['leftcues'][t]
    Lcues = np.append(Lcues, 500) #append cue to the end to avoid empty arrays
    Rcues = trialdata['rightcues'][t]
    Rcues = np.append(Rcues, 500)
    sol = simulate(Lcues, Rcues)
    delta = len(Lcues)-len(Rcues) #final cue difference
    
    #keep track of trials for which left or right decisions were made
    wentleft = 1*(sol[-1, 16]>sol[-1, 33])

    #track whether the final decision was left or right for different differences in cues     
    if delta in psychometric:
        psychometric[delta].append(wentleft)
    else:
        psychometric[delta] = [wentleft]

#calculate pyschometric curve        
psychometric.pop(0, 0)
cuediffsnonoise = sorted(psychometric.keys())
perfnonoise = [np.mean(psychometric[c]) for c in cuediffsnonoise]

def rebin(p):
    '''
    function that averages a dictionary of psychometric data collected for single evidence levels and rebins as groups of 3 evidence levels
    '''
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

for i in range(25): #for 25 sessions
    psychometricerror = {}
    
    samples = np.random.choice(np.arange(1000), 150) #randomly select 150 trials for session
        
    for t in samples: #loop over selected trials, simulating with input noise
        Lcues = trialdata['leftcues'][t]
        Lcues = np.append(Lcues, 500)
        Rcues = trialdata['rightcues'][t]
        Rcues = np.append(Rcues, 500)
        sol = simulate(Lcues, Rcues, input_noise = True) #set input noise to true for simulation with cues ignored
        delta = len(Lcues)-len(Rcues)
        
        wentleft = 1*(sol[-1, 16]>sol[-1, 33])    
        if delta in psychometricerror:
            psychometricerror[delta].append(wentleft)
        else:
            psychometricerror[delta] = [wentleft]
    
    psychometricerror.pop(0, 0)
    psychometrics = psychometrics + [psychometricerror]

#rebin evidence levels for noisy simulated sessions
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

#plot psychometric curves
plt.figure()
plt.plot(-1*np.array(cuediffsnonoise), 1-np.array(perfnonoise), color='black', label = 'No Noise') #psychometric data without noise
plt.errorbar(-1*np.array(cuediffs), 1-np.array(meanperf), yerr = sdperf, color = 'red', label = 'Input Noise') #psychometric data from noisy simulated sessions
plt.xlabel('#R - #L')
plt.ylim([0,1])
plt.xlim([-15, 15])
plt.axvline(x=0, color = 'grey', linestyle = 'dashed') 
plt.axhline(y=0.5, color = 'grey', linestyle = 'dashed')  
plt.ylabel('Model Performance')
plt.legend()

