# -*- coding: utf-8 -*-
"""
Created on Thu May 12 16:59:04 2022

@author: lindseyb
"""

import numpy as np
from scipy import stats
import os
from scipy.optimize import curve_fit
from sklearn.preprocessing import minmax_scale
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import sys
import matplotlib.pyplot as plt


def modelfit(frs, ev, choices):
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
    F_ev = (SS2-SS1)/(SS1/(len(frs)-3))
    
    #fit model with only evidence as regressor
    lr3 = LinearRegression()
    lr3.fit(ev, frs)
    preds3 = lr3.predict(ev)
    SS3 = len(frs)*mean_squared_error(preds3, frs)
    F_choice = (SS3-SS1)/(SS1/(len(frs)-3))
    
    return beta_ev, beta_choice, beta_i, F_ev, F_choice

def pseudofit(frs, ev, choice):
    beta_ev, beta_choice, beta_i, F, F_choice = modelfit(frs, ev, choice)
    return F

arguments = sys.argv[1:]
infile = arguments[0]
infile = infile.split('.')[0]+'.npy'

# preprocessing to remove nandata
data = np.load(infile)
frs = data[:, 0]
#frs = minmax_scale(frs)
pos = data[:, 1].reshape(-1, 1)
ev = data[:, 2].reshape(-1, 1)
choices = np.zeros(len(frs)).reshape(-1, 1)
frsnonnan = frs[~np.isnan(frs)]
frsnonnan = minmax_scale(frsnonnan)
posnonnan = pos[~np.isnan(frs)].reshape(-1, 1)
evnonnan = ev[~np.isnan(frs)].reshape(-1, 1)

i = 65
while i<len(frs):
    choices[i-65:i+1] = np.sign(ev[i])
    i = i+66

choicesnonnan = choices[~np.isnan(frs)]   

evsets = np.load('evidencegenerated.npy')

params = np.zeros((66, 6))

poses = pos[:66]
for i, p in enumerate(poses):
    validps = np.where(posnonnan==p)[0]
    #print(len(validps))
    
    
    beta_ev, beta_choice, beta_i, F_ev, F_choice = modelfit(frsnonnan[validps], evnonnan[validps], choicesnonnan[validps])
    
    if (i>20) and (i<25):
        print(len(frsnonnan[validps]))
        print(beta_choice)
    
    Fs_comp = np.zeros(50)
    for j in range(50):
        evgen = evsets[j, :len(frs)]
        evgen = evgen[~np.isnan(frs)]
        F_comp = pseudofit(frsnonnan[validps], evgen[validps].reshape(-1, 1), choicesnonnan[validps])
        Fs_comp[j] = F_comp
    
    if np.mean(Fs_comp)>F_ev:
        params[i, :] = [beta_ev, beta_choice, beta_i, F_ev, F_choice, 1]
    else:    
        stat = stats.ttest_1samp(Fs_comp, F_ev)
        params[i, :] = [beta_ev, beta_choice, beta_i, F_ev, F_choice, stat.pvalue]
        #print(params[i, :])

region, fdata, neuron = infile.split('/')    
np.save(region+'/paramfit-linearencoding-nochoicematch/'+neuron, params)



                
