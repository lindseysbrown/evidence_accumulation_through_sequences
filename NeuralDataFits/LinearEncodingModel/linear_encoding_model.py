# -*- coding: utf-8 -*-
"""
Created on Thu May 12 16:59:04 2022

@author: lindseyb
"""

import numpy as np
from scipy import stats
from sklearn.preprocessing import minmax_scale
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import sys

#demo
demo = True #True to run with example datasets, False to run for full data in structured folders

def modelfit(frs, ev, choices):
    '''
    fit a linear model to predict firing rates (frs) from cumulative evidences (ev) and signed evidence(ie, current best choice; choices)

    === outputs ===
    beta_ev: coefficient on evidence in the full model
    beta_choice: coefficient on choice in the full model
    beta_i: intercept in the full model
    F_ev: f statistic for the evidence coefficient
    F_choice: f statistic for the choice coefficient
    '''
    #fit full linear regression model using scikit learn
    lr = LinearRegression()
    X = np.concatenate((ev, choices), axis=1)
    lr.fit(X, frs)
    beta_ev, beta_choice = lr.coef_
    beta_i = lr.intercept_
    preds = lr.predict(X)
    SS1 = len(frs)*mean_squared_error(preds, frs)
    
    #fit model with only choice as regressor
    lr2 = LinearRegression()
    lr2.fit(choices, frs)  
    preds2 = lr2.predict(choices)
    SS2 = len(frs)*mean_squared_error(preds2, frs)
    F_ev = (SS2-SS1)/(SS1/(len(frs)-3)) #calculate f-statistic
    
    #fit model with only evidence as regressor
    lr3 = LinearRegression()
    lr3.fit(ev, frs)
    preds3 = lr3.predict(ev)
    SS3 = len(frs)*mean_squared_error(preds3, frs)
    F_choice = (SS3-SS1)/(SS1/(len(frs)-3)) #calculate f-statistic
    
    return beta_ev, beta_choice, beta_i, F_ev, F_choice

arguments = sys.argv[1:]
infile = arguments[0]
infile = infile.split('.')[0]+'.npy'

if not demo:
    neuron = int(infile.split('neuron')[1].split('.')[0])
else:
    neuron = 153

sigma=.1

# preprocessing to remove nandata
data = np.load(infile) #load data array with firing rates (1st column), positions (2nd column), cumulative evidences (3rd column)
frs = data[:, 0]
pos = data[:, 1].reshape(-1, 1)
ev = data[:, 2].reshape(-1, 1)

#remove nan firing rate datapoints
frsnonnan = frs[~np.isnan(frs)] 
frsnonnan = minmax_scale(frsnonnan)
posnonnan = pos[~np.isnan(frs)].reshape(-1, 1)
evnonnan = ev[~np.isnan(frs)].reshape(-1, 1)

#current best choice predictor is sign of evidence
choicesnonnan = .5*(np.sign(evnonnan)+1)

#array to save fit parameters at each position
params = np.zeros((66, 6))

poses = pos[:66]
for i, p in enumerate(poses): #iterate over all positions
    validps = np.where(posnonnan==p)[0] #find data at position
    
    beta_ev, beta_choice, beta_i, F_ev, F_choice = modelfit(frsnonnan[validps], evnonnan[validps], choicesnonnan[validps]) #fit linear regression model at position
       
    Fs_comp = np.zeros(50)
    
    if np.mean(Fs_comp)>F_ev:
        params[i, :] = [beta_ev, beta_choice, beta_i, F_ev, F_choice, 1] #not a significant fit if correlation is negative
    else:    
        stat = stats.ttest_1samp(Fs_comp, F_ev)
        params[i, :] = [beta_ev, beta_choice, beta_i, F_ev, F_choice, stat.pvalue] #test if correlation is significantly better than 0

if not demo:
    region, fdata, neuron = infile.split('/')    
    np.save(region+'/paramfit-linearencoding/'+neuron, params) #save results for future analysis
else:
    print(params)



                
