# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 11:53:49 2022

@author: lindseyb
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from skimage.measure import block_reduce
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Arial']})

#set colormap 
current_cmap = matplotlib.cm.get_cmap('YlOrRd')
current_cmap.set_bad(color='black', alpha=.3) 

#load set of left and right cues for simualtion trials
trialdata = pd.read_pickle('ExampleData/trialdata.pkl')
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
    #self-excitation
    W[i, i] = b+.001*i #increase weight of self-excitation to make chains unstable
    W[i+neurons, i+neurons] = b+.001*i
    #feedforward connections
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

#cue input signal
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
    
    #differential equation
    def chain(y, t):
        dydt = -a*y+np.maximum(W@y+P(t)+scale*I(t, Lcues, Rcues)-T, 0) #Eq. (1) but with unstable parameterization
        return dydt



    y0 = np.concatenate((Lchain, Rchain))
    
    t = np.linspace(0, 330, 3301)
    
    sol = odeint(chain, y0, t, hmax=5, h0 = .5)
    return sol



#without noise in input
leftsol = np.zeros((3301, neurons*2))
rightsol = np.zeros((3301, neurons*2))
alldata = np.zeros((3301, neurons*2))
lchoices = []
rchoices = []
psychometric = {}
    
for t in range(len(trialdata)):
    Lcues = trialdata['leftcues'][t]
    Lcues = np.append(Lcues, 500)
    Rcues = trialdata['rightcues'][t]
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



psychometric.pop(0, 0)
cuediffs = sorted(psychometric.keys())
perf = [np.mean(psychometric[c]) for c in cuediffs]

#remove initialized zero array        
alldata = alldata[:, :, 1:]

#function to determine cumulative evidence at each position
def get_cum_evidence(lefts, rights, pos):
    cum_ev = np.zeros(len(pos))
    for i, p in enumerate(pos):
        cum_ev[i] = np.sum(lefts<p)-np.sum(rights<p)
    return cum_ev

pos = np.arange(-30, 300, 5)
n_trials = 1000
n_pos = len(pos)
evidence = np.zeros((n_trials, n_pos))
for i in range(n_trials):
    evidence[i] = get_cum_evidence(trialdata['leftcues'][i], trialdata['rightcues'][i], pos)


#make plot of population average firing rate vs. position and evidence
def get_tuningcurve(data, evidence, tuningdict):
    '''
    build dictionary of neural firing rates in each position x evidence bin

    === inputs ===
    data: neural firing data at each position on each trial
    evidence: cumulative evidence at each position on each trial
    tuningdict: current dictionary of neural firing rates with keys tuples of evidence and position indices and values an array of observed firing rates

    === outputs ===
    tuningdict: updated dictionary of neural firing rates

    '''
    evs = np.arange(-15, 16)
    obs = np.zeros((len(evs), 66))
    for p in range(66):
        for i, e in enumerate(evs):
            validis = np.where(evidence[:, p]==e)[0]
            if len(validis)>2: #1: #2 or more trials
                if (i, p) in tuningdict:
                    tuningdict[(i, p)] = np.concatenate((tuningdict[(i, p)], np.nanmean(data[validis, :, p], axis=0)))
                else:
                    tuningdict[(i, p)] = np.mean(data[validis, :, p], axis=0)
    return tuningdict
    

def jointdict_to_tuning(tuningdictright, tuningdictleft):
    '''
    take tuning dictionaries and convert to population averages

    ===inputs===
    tuningdictright: dictionary with keys tuples of evidence and position and values lists of mean firing rate in that bin of each right preferring neuron
    tuningdictleft: dictionary with keys tuples of evidence and position and values lists of mean firing rate in that bin of each left preferring neuron

    ===outputs===
    array of average activity in preferred evidence by position bins
    '''
    evs = np.arange(-15, 16)
    obs = np.zeros((len(evs), 66))
    for p in range(66):
        for i, e in enumerate(evs):
            ri = np.where(evs==-1*e)[0][0]
            if (i, p) in tuningdictleft:
                obsL = tuningdictleft[(i,p)]
            else:
                obsL = []
            if (ri, p) in tuningdictright:
                obsR = tuningdictright[(ri,p)]
            else:
                obsR = [] 
            if len(obsL)+len(obsR)>0:
                vals = np.concatenate((obsL, obsR))
                q1 = np.percentile(vals, 25)
                q3 = np.percentile(vals, 75)
                outlierhigh = vals>(1.5*(q3-q1)+q3)
                outlierlow = vals<(q1-1.5*(q3-q1))
                if sum(outlierhigh)==1:
                    valid = ~ outlierhigh
                    vals = vals[valid] #remove if single outlier
                obs[i, p] = np.mean(vals)
            else:
                obs[i, p] = np.nan
    return obs

alldata = block_reduce(alldata.T, block_size = (1, 1, 50), func = np.mean)
alldata = alldata[:, :, :-1]

regiontuningdictL = {}
regiontuningdictR = {}
lefttotal = alldata[:, :neurons, :]
righttotal = alldata[:, neurons:, :]
regiontuningdictL = get_tuningcurve(lefttotal, evidence, regiontuningdictL)
regiontuningdictR = get_tuningcurve(righttotal, evidence, regiontuningdictR)
poptuningcombined = jointdict_to_tuning(regiontuningdictL, regiontuningdictR)


colors = {16:'darkslateblue', 26:'dodgerblue', 36:'aqua', 46:'purple', 56:'fuchsia'}

#plot 2D tuning curve
plt.figure()
plt.imshow(poptuningcombined, cmap='YlOrRd', vmin=0, vmax = .9, aspect='auto', interpolation='none')
plt.xticks([0, 5.5, 25.5, 45.5, 55.25, 65], labels=['-30', '0', 'cues', '200', 'delay', '300'])
#plt.yticks([0, 15, 30], ['-15', '0', '15'])
plt.yticks([5, 15, 25], ['-10', '0', '10'])
for c, i in enumerate([16, 26, 36, 46, 56]):
    plt.axvline(i, color=colors[i], linewidth=2.5, linestyle = '--')
plt.title(' Preferred - Non Pref Population Tuning')
plt.colorbar()
plt.ylim([25,5])
plt.show()

#plot crosssections of population tuning curve
evs = np.arange(-15, 16)
plt.figure()
for c, i in enumerate([16, 26, 36, 46, 56][::-1]):
    plt.plot(evs, poptuningcombined[:, i][::-1], color=colors[i], linewidth = 3)
plt.title(' Preferred - Non Pref')
plt.xlim([-10, 10])
plt.ylim([0, .9])
plt.show()

#plot evidence tuning curves for individual cells
def get_average(data, evidence, n_neurons, evs, poses):
    '''
    calculate average neural firing at each position and evidence bin
    === inputs ===
    data: neural firing data for each trial at each position
    evidence: cumulative evidence for each trial at each position
    n_neurons: number of neurons simulated
    evs: list of evidence bins
    poses: list of position bins
    '''
    firingtotals = np.zeros((n_neurons, len(poses), len(evs)))
    counts = np.zeros((n_neurons, len(poses), len(evs)))
    for e, ev in enumerate(evs):
        its, ips = np.where(evidence==ev)
        for (t, p) in zip(its, ips):
            firingtotals[:, p, e]=firingtotals[:, p, e]+data[t, :, p].T
            counts[:, p, e] = counts[:, p, e]+1
    countsfordivision = counts.copy()
    countsfordivision[np.where(counts==0)]=1
    firingavg = np.divide(firingtotals,countsfordivision)
    return firingavg, counts

evlevels = np.arange(-15, 16)

firingavg, counts = get_average(alldata, evidence, 2*neurons, evlevels, pos)
undersampled = np.where(counts[0, :, :]<1)

plt.figure()
nLs = [4]
for i, nL in enumerate(nLs):
    FAL = firingavg[nL, :, :]
    FAL[undersampled] = np.nan
    activepL = np.argmax(np.nanmean(FAL, axis=1))
    plt.plot(-1*evlevels, FAL[activepL, :], color = 'blue')
nRs = [21]
for i, nR in enumerate(nRs):
    FAR = firingavg[nR, :, :]
    FAR[undersampled] = np.nan
    activepR = np.argmax(np.nanmean(FAR, axis=1))
    plt.plot(-1*evlevels, FAR[activepR, :], color = 'red')
plt.title('early cells')
plt.xlim([-6, 6])
plt.ylim([0, 14])
plt.show()

plt.figure()
nLs = [8]
for i, nL in enumerate(nLs):
    FAL = firingavg[nL, :, :]
    FAL[undersampled] = np.nan
    activepL = np.argmax(np.nanmean(FAL, axis=1))
    plt.plot(-1*evlevels, FAL[activepL, :], color = 'blue')
nRs = [25]
for i, nR in enumerate(nRs):
    FAR = firingavg[nR, :, :]
    FAR[undersampled] = np.nan
    activepR = np.argmax(np.nanmean(FAR, axis=1))
    plt.plot(-1*evlevels, FAR[activepR, :], color = 'red')
plt.title('late cells')
plt.xlim([-10, 10])
plt.ylim([0, 14])
plt.show()

plt.figure()
nLs = [15]
for i, nL in enumerate(nLs):
    FAL = firingavg[nL, :, :]
    FAL[undersampled] = np.nan
    activepL = np.argmax(np.nanmean(FAL, axis=1))
    plt.plot(-1*evlevels, FAL[activepL, :], color = 'blue')
nRs = [32]
for i, nR in enumerate(nRs):
    FAR = firingavg[nR, :, :]
    FAR[undersampled] = np.nan
    activepR = np.argmax(np.nanmean(FAR, axis=1))
    plt.plot(-1*evlevels, FAR[activepR, :], color = 'red')
plt.title('delay cells')
plt.xlim([-10, 10])
plt.ylim([0, 14])
plt.show()
