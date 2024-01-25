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

data = pickle.load(open('../ACCCho/Suite2p/dFF_tetO_7_07302021_T10/cell0_fr2.p', 'rb')) #load a cell with desired average firing rate

ACCpath = '../ACCCho/Suite2p' #path to files with fluorescence information
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
                data = pickle.load(open(fileloc, 'rb')) #load file of estimated firing rates
                spike_ts = data['fit']['spikes'] #get spike times
                spikeoccurred = np.zeros((upfactor*n_frames,))
                spikeoccurred[spike_ts] = 1
                spikes[n, :] = spikeoccurred
            frs = 60*rolling_avg(spikes, binsize) #get firing rate based on rolling average
            frs_gauss = 60*gauss_avg(spikes, binsize) #get firing rate based on gaussian
            for n in range(n_neurons): #do this for each cell in the session
                FRs[n, :] = np.mean(frs[n, :].reshape(-1, upfactor), axis=1)
                FRsgauss[n, :] = np.mean(frs_gauss[n, :].reshape(-1, upfactor), axis=1)
    
    
    


