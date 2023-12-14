# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 10:19:35 2023

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
import pickle
from scipy.stats import f, sem

from matplotlib_venn import venn2, venn3


regions = ['ACC', 'DMS', 'HPC', 'RSC', 'V1']

percentevsig = {}

poses = np.arange(-30, 300, 5)

#parameters
#0: beta_ev, 1:beta_choice, 2:beta_i, 3:F_ev, 4:F_choice, 5: p_ev

with open('numcorrecttrials.pkl', 'rb') as handle:
    numcorrecttrials = pickle.load(handle)

plt.figure()
for region in regions:
    fitparams = np.load(region+'/paramfit/'+region+'allfitparams-linearencoding-signedevidence.npy')
    fitneurondata = pd.read_csv(region+'/paramfit/'+region+'allfitparams-linearencoding-neuroninfo-signedevidence.csv')
    
    sigF = np.zeros(len(fitneurondata))
    
    for j in range(len(sigF)):
        session = fitneurondata['Session'].values[j]
        ncorrect = numcorrecttrials[session]
        sigF[j] = f.ppf(.95, 1, ncorrect-3)

   
    sessionlist = list(set(fitneurondata['Session'].values))
    
    evsig = np.zeros((66, len(sessionlist)))
    choicesig = np.zeros((66, len(sessionlist)))
    
    #normalized by position tuning of cells
    mups = []
    sigps = []
    fitparamsgauss = pd.read_csv(region+'/paramfit/'+region+'allfitparams-boundedmue-NEW.csv')
    fitparamsevbeyond = pd.read_csv(region+'/paramfit/'+region+'allfitparams-boundedmue-choicematched.csv')
    for s, n in zip(fitneurondata['Session'].values, fitneurondata['Index'].values):
        try:
            mups.append(fitparamsgauss['Mup'][(fitparamsgauss.Neuron==n) & (fitparamsgauss.Session==s)].iloc[0])
            sigps.append(fitparamsgauss['Sigp'][(fitparamsgauss.Neuron==n) & (fitparamsgauss.Session==s)].iloc[0])
        except:
            mups.append(-500)
            sigps.append(0)

    mups = np.array(mups)
    sigps = np.array(sigps)
    #get cells significant by other fitting method
    pvals = fitparamsgauss['Pval'].values
    corr = fitparamsgauss['Correlation'].values
    evsigN = fitparamsgauss['Neuron'].values[(pvals<.05) & (corr>.1)]
    evsigSession = fitparamsgauss['Session'].values[(pvals<.05) & (corr>.1)]
    evsiggauss = [(evsigN[i], evsigSession[i]) for i in range(len(evsigN))]
    
    pvals = fitparamsevbeyond['Pval'].values
    corr = fitparamsevbeyond['Correlation'].values
    evsigN = fitparamsevbeyond['Neuron'].values[(pvals<.05) & (corr>.1)]
    evsigSession = fitparamsevbeyond['Session'].values[(pvals<.05) & (corr>.1)]
    evsigbeyond = [(evsigN[i], evsigSession[i]) for i in range(len(evsigN))]
    
    evsignorm = np.zeros((66, len(sessionlist)))
    evsigkde = np.zeros((66, len(sessionlist)))
    choicesignorm = np.zeros((66, len(sessionlist)))
    choicesigkde = np.zeros((66, len(sessionlist)))
    validicount = np.zeros((66, len(sessionlist)))
    sigevtuningactive = np.zeros(len(fitneurondata))
    
    
    
    
    for k, sess in enumerate(sessionlist):
        validsession = np.where(fitneurondata['Session'].values == sess)[0]
        
        sigFsession = sigF[validsession]
        
        print(len(sigFsession))
        if len(sigFsession)>50:
            for i in range(len(evsig)):
                evsig[i, k] = np.mean(fitparams[i, 3, validsession]>sigFsession)
                choicesig[i, k] = np.mean(fitparams[i, 4, validsession]>sigFsession)
        else:
            evsig[:, k] = np.nan
            choicesig[:, k] = np.nan
            
        fitparamssession = fitparams[:, :, validsession]
        mupssession = mups[validsession]
        sigpssession = sigps[validsession]
        
        if len(sigFsession)>50:
            for i in range(len(evsignorm)):
                validis = np.where(np.abs(mupssession-poses[i])<sigpssession)[0]
                evsignorm[i, k] = np.mean(fitparamssession[i, 3, validis]>sigFsession[validis])
                choicesignorm[i, k] = np.mean(fitparamssession[i, 4, validis]>sigFsession[validis])
        else:
            evsignorm[:, k] = np.nan
            choicesignorm[:, k] = np.nan            
   
        
    plt.figure()
    plt.plot(poses, np.nanmean(evsig, axis=1), color = 'indigo', label = 'Evidence', linewidth=3)
    plt.fill_between(poses, np.nanmean(evsig, axis=1)-sem(evsig, axis=1, nan_policy='omit'), np.nanmean(evsig, axis=1)+sem(evsig, axis=1, nan_policy='omit'), alpha = .5, color = 'indigo')
    plt.plot(poses, np.nanmean(choicesig, axis=1), color='mediumpurple', label = 'sign(Evidence)', linewidth=3)
    plt.fill_between(poses, np.nanmean(choicesig, axis=1)-sem(choicesig, axis=1, nan_policy='omit'), np.nanmean(choicesig, axis=1)+sem(choicesig, axis=1, nan_policy='omit'), alpha = .5, color = 'mediumpurple')
    plt.title(region)
    plt.xlabel('Position')
    plt.ylabel('Fraction Neurons with Significant Tuning')
    #plt.legend(ncol =3)
    plt.ylim([0, .3])
    plt.axvline(0, color = 'k', linestyle = '--')
    plt.axvline(200, color = 'k', linestyle = '--')
    plt.legend()
        
        
    
    plt.figure()
    plt.plot(poses, np.nanmean(evsignorm, axis=1), color = 'darkorange', label = 'Evidence', linewidth=3)
    plt.fill_between(poses, np.nanmean(evsignorm, axis=1)-sem(evsignorm, axis=1, nan_policy='omit'), np.nanmean(evsignorm, axis=1)+sem(evsignorm, axis=1, nan_policy='omit'), alpha = .5, color = 'darkorange')
    plt.plot(poses, np.nanmean(choicesignorm, axis=1), color = 'red', label='sign(Evidence)', linewidth=3)
    plt.fill_between(poses, np.nanmean(choicesignorm, axis=1)-sem(choicesignorm, axis=1, nan_policy='omit'), np.nanmean(choicesignorm, axis=1)+sem(choicesignorm, axis=1, nan_policy='omit'), alpha = .5, color = 'red')
    plt.title(region)
    plt.xlabel('Position')
    plt.ylabel('Fraction Position Active Neurons \n with Significant Tuning')
    #plt.legend(ncol =3)
    plt.ylim([0, .3])
    plt.axvline(0, color = 'k', linestyle = '--')
    plt.axvline(200, color = 'k', linestyle = '--')  
    plt.axhline(.05, color='grey', linestyle = '--')
    plt.legend()
    plt.savefig('Figure4Plots/LinearEvidenceEncoding/'+region+'-bysession-signedev-5sig.pdf')

    

            



