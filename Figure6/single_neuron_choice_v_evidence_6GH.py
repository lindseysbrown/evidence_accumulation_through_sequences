# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 10:19:35 2023

@author: lindseyb
"""

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
matplotlib.rcParams.update({'font.size': 18})
import pandas as pd
import pickle
from scipy.stats import f, sem

#demo
demo = True #True to run with example datasets, False to run for full data in structured folders

if not demo:
    regions = ['ACC', 'DMS', 'HPC', 'RSC']
else:
    regions = ['ACC']

percentevsig = {}

#set up position bins
poses = np.arange(-30, 300, 5)

#parameters
#0: beta_ev, 1:beta_choice, 2:beta_i, 3:F_ev, 4:F_choice, 5: p_ev

with open('ExampleData/numtotaltrials.pkl', 'rb') as handle:
    numcorrecttrials = pickle.load(handle) #dictionary of number of trials in each session

plt.figure()
for region in regions:
    if not demo:
        fitparams = np.load(region+'/paramfit/'+region+'allfitparams-linearencoding.npy') #load saved parameters from running encoding model
        fitneurondata = pd.read_csv(region+'/paramfit/'+region+'allfitparams-linearencoding-neuroninfo.csv') #load neuron, session information corresponding to parameters

    else:
        fitparams = np.load('ExampleData/ACCsingleneuronencoding.npy')
        fitneurondata = pd.read_csv('ExampleData/ACCsingleneuronencoding-neuroninfo.csv')
    
    #initialize zero matrix for each neuron of F-statistic needed for it to be significant
    sigF = np.zeros(len(fitneurondata))
    
    for j in range(len(sigF)):
        session = fitneurondata['Session'].values[j] #find current session
        ncorrect = numcorrecttrials[session] #find number of samples on that session
        sigF[j] = f.ppf(.95, 1, ncorrect-3) #find significant f statistic for that session

   
    sessionlist = list(set(fitneurondata['Session'].values))
    
    #initialize arrays to track whether for each neuron at each position, the evidence and choice parameters are significant
    evsig = np.zeros((66, len(sessionlist)))
    choicesig = np.zeros((66, len(sessionlist)))
    
    #normalized by position tuning of cells
    mups = []
    sigps = []
    if not demo:
        fitparamsgauss = pd.read_csv(region+'/paramfit/'+region+'allfitparams.csv') #load fit joint gaussian parameters
    else:
        fitparamsgauss = pd.read_csv('ExampleData/ACCparamfitexample.csv')
    for s, n in zip(fitneurondata['Session'].values, fitneurondata['Index'].values): #for each neuron find position parameters
        try:
            mups.append(fitparamsgauss['Mup'][(fitparamsgauss.Neuron==n) & (fitparamsgauss.Session==s)].iloc[0])
            sigps.append(fitparamsgauss['Sigp'][(fitparamsgauss.Neuron==n) & (fitparamsgauss.Session==s)].iloc[0])
        except:
            mups.append(-500)
            sigps.append(0)

    mups = np.array(mups)
    sigps = np.array(sigps)

    #initialize arrays to track the precentage of active neurons at each position with signficant evidence and choice parameters       
    evsignorm = np.zeros((66, len(sessionlist)))
    choicesignorm = np.zeros((66, len(sessionlist)))  
    
    for k, sess in enumerate(sessionlist):
        validsession = np.where(fitneurondata['Session'].values == sess)[0]
        
        sigFsession = sigF[validsession]
        
        print(len(sigFsession))
        if len(sigFsession)>50: #only consider sessions with at least 50 neurons
            for i in range(len(evsig)): #loop over positions
                evsig[i, k] = np.mean(fitparams[i, 3, validsession]>sigFsession) #check if F-stat for evidence parameter is significant, fraction of all cells
                choicesig[i, k] = np.mean(fitparams[i, 4, validsession]>sigFsession)#check if F-stat for choice parameter is significant, fraction of all cells
        else:
            evsig[:, k] = np.nan
            choicesig[:, k] = np.nan

        #get the position tuning parameters associated with this session    
        fitparamssession = fitparams[:, :, validsession]
        mupssession = mups[validsession]
        sigpssession = sigps[validsession]
        
        if len(sigFsession)>50:
            for i in range(len(evsignorm)):
                validis = np.where(np.abs(mupssession-poses[i])<sigpssession)[0] #only consider neurons with position mean within 1 position standard deviation as active
                evsignorm[i, k] = np.mean(fitparamssession[i, 3, validis]>sigFsession[validis]) #check if F-stat for evidence parameter is significant, fraction of active cells
                choicesignorm[i, k] = np.mean(fitparamssession[i, 4, validis]>sigFsession[validis]) #check if F-stat for choice parameter is significant, fraction of active cells
        else:
            evsignorm[:, k] = np.nan
            choicesignorm[:, k] = np.nan            
   
    #plot results
    plt.figure()
    plt.plot(poses, np.nanmean(evsignorm, axis=1), color = 'darkorange', label = 'Evidence', linewidth=3) #mean across sessions
    plt.fill_between(poses, np.nanmean(evsignorm, axis=1)-sem(evsignorm, axis=1, nan_policy='omit'), np.nanmean(evsignorm, axis=1)+sem(evsignorm, axis=1, nan_policy='omit'), alpha = .5, color = 'darkorange') #standard error by session
    plt.plot(poses, np.nanmean(choicesignorm, axis=1), color = 'red', label='choice', linewidth=3) #mean across sessions
    plt.fill_between(poses, np.nanmean(choicesignorm, axis=1)-sem(choicesignorm, axis=1, nan_policy='omit'), np.nanmean(choicesignorm, axis=1)+sem(choicesignorm, axis=1, nan_policy='omit'), alpha = .5, color = 'red') #standard error by session
    plt.title(region)
    plt.xlabel('Position')
    plt.ylabel('Fraction Position Active Neurons \n with Significant Tuning')
    plt.legend()
    plt.ylim([0, .5])
    plt.axvline(0, color = 'k', linestyle = '--')
    plt.axvline(200, color = 'k', linestyle = '--')  
    plt.axhline(.05, color='grey', linestyle = '--')
    plt.legend()
    plt.show()

    

            



