# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 11:17:01 2023

@author: lindseyb
"""

import numpy as np
from scipy.io import loadmat
from sklearn import decomposition
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
matplotlib.rcParams.update({'font.size': 18})
from scipy import stats
from scipy.stats import f_oneway, kurtosis, skew
import os
from scipy.optimize import curve_fit
import diptest
import pandas as pd
import seaborn as sns
from scipy.stats import f_oneway, kurtosis, skew, sem
from sklearn.preprocessing import minmax_scale
import pickle


def plot_ndata(data, evs, poses, mue, mup, sigp, n, s, r, maze, idx):
    frs = data[:, 0]
    data = data[~np.isnan(frs), :]
    frs = data[:, 0]
    pos = data[:, 1].reshape(-1, 1)
    ev = data[:, 2].reshape(-1, 1)
    frs = minmax_scale(frs)
           
    fig, ax = plt.subplots(1, 1)
         
    obs = np.zeros((len(evs), len(poses)))
    #sems = np.zeros((len(evs), len(poses)))
    counts = np.zeros((len(evs), len(poses)))
    for i, e in enumerate(evs):
        for j, p in enumerate(poses):
            vp = pos[:, 0]==p
            ve = ev[:, 0]==e
            os = vp & ve
            counts[i][j] = sum(os)
            if sum(os)>1:
                obs[i][j] = np.mean(frs[os])
                #sems[i][j] = sem(frs[os])
            else:
                obs[i][j] = np.nan
                #sems[i][j] = np.nan  
    ip = np.argmin(np.abs(mup-poses))
    ie = np.argmin(np.abs(mue-evs))            
    vm = np.nanmin(obs)
    vM = np.nanmax(obs)
    ax.imshow(obs, cmap = 'YlOrRd', interpolation = 'none', vmin=vm, vmax=vM)
    ax.set_xticks([0, 5.5, 25.5, 45.5, 55.25, 65])
    ax.set_xticklabels(['-30', '0', 'cues', '200', 'delay', '300'])
    ax.set_yticks([0, 15, 30])
    ax.set_yticklabels(['-15', '0', '15'])
    ax.axvline(min((mup+.5*sigp+30)/5, 65), color = 'k', linestyle = '--')
    ax.axvline(max((mup-.5*sigp+30)/5, 0), color = 'k', linestyle = '--')
    #ax.scatter(ip, ie, color = 'red', s=2)
            
    plt.suptitle('Neuron '+str(n)+' r='+str(np.round(r,3)))
    plt.savefig('ExamplesCriteria/'+'red'+region+maze+str(idx)+s+'neuron'+str(n)+'.pdf')
    

#current_cmap = matplotlib.cm.get_cmap('Purples')
#current_cmap.set_bad(color='black', alpha=.4) 
current_cmap = matplotlib.cm.get_cmap('YlOrRd')
current_cmap.set_bad(color='black', alpha=.3) 

 
evs = np.arange(-15, 16)


#load criteria cells
regions = ['ACC', 'HPC', 'DMS', 'RSC', 'V1']

for region in regions[3:4]:
    if region in ['ACC', 'DMS']:
        poses = np.arange(-27.5, 302.5, 5)
    else:
        poses = np.arange(-30, 300, 5) 
    
    
    with open(region+"-nonoutliercells.p", "rb") as fp:   #Pickling
        nstoplot = pickle.load(fp)
    


    fitparams = pd.read_csv(region+'/paramfit/'+region+'allfitparams-boundedmue-NEW.csv')

    keep = np.zeros(len(fitparams))
    for i in range(len(fitparams)):
        n = (fitparams['Neuron'].values[i], fitparams['Session'].values[i])
        if n in nstoplot:
            keep[i] = 1
            
    fitparams['Keep'] = keep
    
    fitparams = fitparams[fitparams['Keep']>0]

    corrs = fitparams['Correlation'].values
    idxs = np.argsort(corrs)[::-1]

    
    #early cue
    print('early')
    earlycells = fitparams[(fitparams['Mup']>0) & (fitparams['Mup']<100)]
    corrs = earlycells['Correlation'].values
    idxs = np.argsort(corrs)[::-1]
    idxs = idxs[:min(50, len(idxs))]
    for num,i in enumerate(idxs):
        print(num)
        params = earlycells.iloc[i]
        n = params['Neuron']
        s = params['Session']
        try:
            ndata = np.load(region+'/firingdata/'+s+'neuron'+str(n)+'.npy')
        except:
            ndata = np.load(region+'/firingdata-split/'+s+'neuron'+str(n)+'.npy') 
            
        plot_ndata(ndata, evs, poses, params['Mue'], params['Mup'], params['Sigp'], n, s, params['Correlation'], 'early', num)     
    
    #late cue
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
        try:
            ndata = np.load(region+'/firingdata/'+s+'neuron'+str(n)+'.npy')
        except:
            ndata = np.load(region+'/firingdata-split/'+s+'neuron'+str(n)+'.npy') 
            
        plot_ndata(ndata, evs, poses, params['Mue'], params['Mup'], params['Sigp'], n, s, params['Correlation'], 'late', num)        
    
    
    #delay
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
        try:
            ndata = np.load(region+'/firingdata/'+s+'neuron'+str(n)+'.npy')
        except:
            ndata = np.load(region+'/firingdata-split/'+s+'neuron'+str(n)+'.npy') 
            
        plot_ndata(ndata, evs, poses, params['Mue'], params['Mup'], params['Sigp'], n, s, params['Correlation'], 'delay', num)
    
        
        
    