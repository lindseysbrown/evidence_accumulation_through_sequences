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

trialdata = pd.read_pickle('trialdata.pkl') #file containing data for set of trials with positions of left and right cues on each trial

#initialize neural rings
neurons = 35
timepoints = 17
Cring = np.zeros((neurons*timepoints,))

#set up parameters
P0 = 20 #postion width
T = 300 #threshold
X = 0.04 #external input
a = 1.1 #decay rate
b = 0.088 #feedforward synaptic connections
w0 = 0.12 #parameterization of within layer cosine connectivity
w1 = -1 #parameterization of within layer cosine connectivity

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
    P[neurons*i:neurons*(i+1)]=T+X #assign all neurons at position layer to threshold level
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

def simulate(Lcues, Rcues, input_noise = False, Inoise = .67):
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
        dydt = -a*y+(F(W@y+feedforward@y+.2*I(t, y, Lcues, Rcues)+P(t)-T))
        return dydt
    
    y0 = Cring
    t = np.linspace(0, 330, 3301)
    
    sol = odeint(ring, y0, t, hmax=5) #numerically integrate the solution using scipy.odeint
    return sol

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

alldata = np.load('bumpmodel.npy') #reload simulated data
perfnonoise = {}

#get psychometric curve for 1000 trials with no noise in input
for t in range(len(trialdata)):
    Lcues = trialdata['leftcues'][t]
    Rcues = trialdata['rightcues'][t]
    sol = alldata[t, :, :].T
    delta = len(Lcues)-len(Rcues)
       

    imax = np.argmax(sol[-1, -1*neurons:]) #determine what simulated response was
    wentleft = imax<(.5*neurons-1) 
    wentright = imax>(.5*neurons)
    
    #track whether the final decision was left or right for different differences in cues   
    if delta in perfnonoise:
        perfnonoise[delta].append(wentleft)
    else:
        perfnonoise[delta] = [wentleft]

#calculate pyschometric curve  
perfnonoise.pop(0)
cuediffsnonoise = sorted(perfnonoise.keys())
perf_nonoise = [np.mean(perfnonoise[c]) for c in cuediffsnonoise]


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
plt.plot(-1*np.array(cuediffsnonoise), 1-np.array(perf_nonoise), color='black', label = 'No Noise')
plt.errorbar(-1*np.array(cuediffs), 1-np.array(meanperf), yerr = sdperf, color = 'red', label = 'Input Noise')
plt.xlabel('#R - #L')
plt.ylabel('Model Performance')
plt.ylim([0,1])
plt.xlim([-15, 15])
plt.axvline(x=0, color = 'grey', linestyle = 'dashed') 
plt.axhline(y=0.5, color = 'grey', linestyle = 'dashed')  
plt.legend()
plt.show()

