# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 11:17:01 2023

@author: lindseyb
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
matplotlib.rcParams.update({'font.size': 18})
import pandas as pd
from sklearn.preprocessing import minmax_scale
import pickle


def plot_ndata(data, evs, poses, mue, mup, sigp, n, s, r, maze, idx):
    '''
    function to plot raw average firing rate in each position x evidence bin

    === inputs ===
    data: array of data consisting of neural firing rates (1st column), positions (2nd column), cumulative evidence (3rd column)
    evs: array of evidence bins
    poses: array of position bins
    mue: fit evidence mean
    mup: fit position mean
    sigp: fit position standard deviation
    n: neuron index
    s: recording session
    r: fit correlation
    maze: region of the maze (early, late, delay) 
    idx: sorted order position of fit correlation in that region of the maze

    === outputs ===
    figure showing raw firing rates in each position x evidence bin
    '''

    #load data
    frs = data[:, 0] #first column is neural data
    data = data[~np.isnan(frs), :] #remove entries where the firing rate is nan
    frs = data[:, 0] #reload non-nan firing rates
    pos = data[:, 1].reshape(-1, 1) #second column is position data
    ev = data[:, 2].reshape(-1, 1) #third column is evidence data
    frs = minmax_scale(frs) #rescale firing rates to [0,1]
           
    fig, ax = plt.subplots(1, 1)
         
    obs = np.zeros((len(evs), len(poses))) #initialize empty array for firing rate observations
    counts = np.zeros((len(evs), len(poses))) #initialize empty array to keep track of the number of times each bin is observed
    #for each evidence and position bin
    for i, e in enumerate(evs):
        for j, p in enumerate(poses):
            vp = pos[:, 0]==p #find data points at correct position
            ve = ev[:, 0]==e #find data points at correct evidence level
            os = vp & ve #all observations in correct bin
            counts[i][j] = sum(os) #number of observations in bin
            if sum(os)>1: #only plot if at least two observations
                obs[i][j] = np.mean(frs[os])
            else:
                obs[i][j] = np.nan 
    #get appropriate range for colorscale         
    vm = np.nanmin(obs)
    vM = np.nanmax(obs)
    ax.imshow(obs, cmap = 'YlOrRd', interpolation = 'none', vmin=vm, vmax=vM)
    #convert xlabels to corresponding position
    ax.set_xticks([0, 5.5, 25.5, 45.5, 55.25, 65])
    ax.set_xticklabels(['-30', '0', 'cues', '200', 'delay', '300'])
    #convert y labels to corresponding evidence
    ax.set_yticks([0, 15, 30])
    ax.set_yticklabels(['-15', '0', '15'])
    
    #plot lines to denote range used to calculate tuning curves
    ax.axvline(min((mup+.5*sigp+30)/5, 65), color = 'k', linestyle = '--')
    ax.axvline(max((mup-.5*sigp+30)/5, 0), color = 'k', linestyle = '--')

    #title and save out neuron raw firing rates     
    plt.suptitle('Neuron '+str(n)+' r='+str(np.round(r,3)))
    plt.savefig('ExamplesCriteria/'+'red'+region+maze+str(idx)+s+'neuron'+str(n)+'.pdf')
    

#set colormap with nan set to gray
current_cmap = matplotlib.cm.get_cmap('YlOrRd')
current_cmap.set_bad(color='black', alpha=.3) 

#define evidence bins 
evs = np.arange(-15, 16)


#load criteria cells
regions = ['ACC', 'HPC', 'DMS', 'RSC']

for region in regions:
    #define position bins corresponding to region
    if region in ['ACC', 'DMS']:
        poses = np.arange(-27.5, 302.5, 5)
    else:
        poses = np.arange(-30, 300, 5) 
    
    #open a list of cells that have been verified to meet non-outlier criteria (e.g. for cells with evidence SD<3, there is not exactly one outlier in the bin containing the mean evidence and mean position)
    with open(region+"-nonoutliercells.p", "rb") as fp:
        nstoplot = pickle.load(fp)
    
    #load file containing results of parameter fitting
    fitparams = pd.read_csv(region+'/paramfit/'+region+'allfitparams.csv')

    #only keep cells that meet non-outlier criteria
    keep = np.zeros(len(fitparams))
    for i in range(len(fitparams)):
        n = (fitparams['Neuron'].values[i], fitparams['Session'].values[i])
        if n in nstoplot:
            keep[i] = 1   
    fitparams['Keep'] = keep
    fitparams = fitparams[fitparams['Keep']>0]

    #early cue
    print('early')
    earlycells = fitparams[(fitparams['Mup']>0) & (fitparams['Mup']<100)] #cells whose mean position falls in the early cue region
    corrs = earlycells['Correlation'].values
    idxs = np.argsort(corrs)[::-1] #plot cells in order of decreasing correlation of fit
    idxs = idxs[:min(50, len(idxs))] #plot only as many as 50 examples
    for num,i in enumerate(idxs):
        print(num)
        params = earlycells.iloc[i]
        #determine which neuron is being plotted
        n = params['Neuron'] 
        s = params['Session']

        #load the corresponding data, consisting of an array of firing rate, position, and evidence values
        ndata = np.load(region+'/firingdata/'+s+'neuron'+str(n)+'.npy')

        #generate plot of the neural data
        plot_ndata(ndata, evs, poses, params['Mue'], params['Mup'], params['Sigp'], n, s, params['Correlation'], 'early', num)     
    

    #late cue, repeat for cells with mean position in the late cue region
    print('late')
    latecells = fitparams[(fitparams['Mup']>100) & (fitparams['Mup']<200)]
    corrs = latecells['Correlation'].values
    idxs = np.argsort(corrs)[::-1]
    idxs = idxs[:min(50, len(idxs))]
    for num,i in enumerate(idxs):
        print(num)
        params = latecells.iloc[i]
        n = params['Neuron']
        s = params['Session']
        ndata = np.load(region+'/firingdata/'+s+'neuron'+str(n)+'.npy')           
        plot_ndata(ndata, evs, poses, params['Mue'], params['Mup'], params['Sigp'], n, s, params['Correlation'], 'late', num)        
    
    
    #delay, repeat for cells with mean position in the delay region
    print('delay')
    delaycells = fitparams[fitparams['Mup']>200]
    corrs = delaycells['Correlation'].values
    idxs = np.argsort(corrs)[::-1]
    idxs = idxs[:min(50, len(idxs))]
    for num,i in enumerate(idxs):
        print(num)
        params = delaycells.iloc[i]
        n = params['Neuron']
        s = params['Session']
        ndata = np.load(region+'/firingdata/'+s+'neuron'+str(n)+'.npy')           
        plot_ndata(ndata, evs, poses, params['Mue'], params['Mup'], params['Sigp'], n, s, params['Correlation'], 'delay', num)
    
        
        
    