# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 14:35:00 2023

@author: lindseyb
"""

import os
from scipy.io import loadmat
from scipy.io import savemat
import numpy as np
import h5py
import pickle
import matplotlib.pyplot as plt

from scipy import signal

binsize = 60 #60 frames = 1s
upfactor = 2

#for each neuron, read pickle file and remap to firing rate of correct size
def rolling_avg(a, windowsize):
    tot = np.cumsum(a, axis=1)
    diffs = tot[:, windowsize:]-tot[:, :-windowsize]
    diffs = diffs/windowsize
    h = int(windowsize/2)
    frs = np.zeros(np.shape(a))
    frs[:, h:np.shape(a)[1]-h] = diffs
    for w in range(h):
        frs[:, w] = np.mean(a[:, :w+h])
    for w in range(1, h+1):
        frs[:, -w] = np.mean(a[:, -(w+h):])
    return frs

def gauss_avg(a, windowsize):
    win = signal.windows.gaussian(windowsize, windowsize/4)
    tot = sum(win)
    frs = np.zeros(np.shape(a))
    for n in range(len(a)):
        frs[n,:] = signal.convolve(a[n], win, mode='same') / tot
    return frs
    

'''
def get_firing_rate(spiketimes, n_frames, binsize, upfactor):
    spikeoccurred = np.zeros((upfactor*n_frames,))
    spikeoccurred[spiketimes] = 1
    frs = rolling_avg(spikeoccurred, binsize)*60
    return frs
'''
      

data = pickle.load(open('../ACCCho/Suite2p/dFF_tetO_7_07302021_T10/cell0_fr2.p', 'rb'))


ACCpath = '../ACCCho/Suite2p'
ACCfiles = os.listdir(ACCpath)
for f in ACCfiles[:3]:
    if f.startswith('dFF_'):
        if os.path.isfile(os.path.join(ACCpath, f)):
            file = ACCpath+'/'+f
            mat = loadmat(file)
            n_neurons, n_frames = np.shape(mat['Output']['dFF'][0][0])
            FRs = np.zeros((n_neurons, n_frames))
            FRsgauss = np.zeros((n_neurons, n_frames))
            spikes = np.zeros((n_neurons, upfactor*n_frames))
            for n in range(n_neurons)[:5]:
                fileloc = ACCpath+'/'+f.split('.mat')[0]+'/cell'+str(n)+'_fr2.p'
                data = pickle.load(open(fileloc, 'rb'))
                spike_ts = data['fit']['spikes']
                spikeoccurred = np.zeros((upfactor*n_frames,))
                spikeoccurred[spike_ts] = 1
                spikes[n, :] = spikeoccurred
            frs = 60*rolling_avg(spikes, binsize)
            frs_gauss = 60*gauss_avg(spikes, binsize)
            for n in range(n_neurons):
            #frs = get_firing_rate(spikes, n_frames, binsize, upfactor)
                FRs[n, :] = np.mean(frs[n, :].reshape(-1, upfactor), axis=1)
                FRsgauss[n, :] = np.mean(frs_gauss[n, :].reshape(-1, upfactor), axis=1)
            #savemat(ACCpath+'/'+f.split('.mat')[0]+'/firingrate.mat', {'fr': FRs})
                
for i in range(5):
    dFF = mat['Output']['dFF'][0][0][i]
    fr = FRs[i]
    fig, (ax1, ax2) = plt.subplots(nrows=2)
    ax1.plot(dFF)
    ax1.set_ylabel('dFF')
    ax1.set_xticks([])
    ax2.plot(fr)
    ax2.set_ylabel('Firing Rate (Hz)')
    ax2.set_xlabel('Time')
    plt.suptitle('ACC Neuron '+str(i)+' Full Trace')
    
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3)
    ax1.plot(dFF[20000:20600])
    ax1.set_ylabel('dFF')
    ax1.set_xticks([])
    ax2.plot(fr[20000:20600])
    ax2.set_ylabel('Firing Rate (Hz) \n Rect. Wind.')
    ax2.set_xticks([])
    ax3.plot(FRsgauss[i, 20000:20600])
    ax3.set_ylabel('Firing Rate (Hz) \n Gauss. Wind.')
    ax3.set_xlabel('Time (s)')
    ax3.set_xticks([0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600])
    ax3.set_xticklabels(2*np.arange(0, 11))
    plt.suptitle('ACC Neuron '+str(i)+' 20 s')
    
    
    


