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
import sys
import matplotlib.pyplot as plt
import pandas as pd



def gauss(x, mu, sig, a, c):
    x = x.reshape(-1)
    return a*np.exp((-(x-mu)**2)/(2*sig**2))+c

def logistic(x, k, x0, a, c):
    x = x.reshape(-1)
    return a/(1+np.exp(-k*(x-x0)))+c

def fit_logistic(x, y, p0, bl, bu):
    popt, pcov = curve_fit(logistic, x, y, p0=p0, bounds = (bl, bu))
    return popt
    
def fit_gauss(x, y, p0, bl, bu):
    popt, pcov = curve_fit(gauss, x, y, p0 = p0, bounds = (bl, bu))
    return popt

def pseudofit(frs, pos, ev, mup, sigp):
    lower = np.max([(mup-.5*sigp), 0])
    validis = (pos>lower)&(pos<(mup+.5*sigp))
    validis = validis.reshape(-1)
    frs = frs[validis]
    ev = ev[validis]
    m = np.min(ev)
    M = np.max(ev)
    try:
        mue, sige, a, c = fit_gauss(ev, frs, [0, 5, 1, 0], [m, 0, 0, 0], [M, 30, 10, 1])
        preds = gauss(ev, mue, sige, a, c)
        rgauss, p = stats.pearsonr(preds.reshape(-1), frs)
    except:
        rgauss = 0
        
    try:
        k, x0, a, c = fit_logistic(ev, frs, [0, 0, 1, 0], [-1, -15, 0, 0], [1, 15, 10, 1])
        preds = logistic(ev, k, x0, a, c)
        rlog, p = stats.pearsonr(preds.reshape(-1), frs)
    except:
        rlog = 0
    
    return rgauss, rlog

arguments = sys.argv[1:]
infile = arguments[0]
infile = infile.split('.')[0]+'.npy'

region, fdata, neuron = infile.split('/') 
s, n = neuron.split('neuron')
n = int(n.split('.npy')[0])

# preprocessing to remove nandata
data = np.load(infile)
frs = data[:, 0]
data = data[~np.isnan(frs), :]
frs = data[:, 0]

frs = minmax_scale(frs)
pos = data[:, 1].reshape(-1, 1)
ev = data[:, 2].reshape(-1, 1)


fitdata = pd.read_csv(region+'/paramfit-boundedmue/'+region+'allfitparams-boundedmue-NEW.csv')
try:
    mup = fitdata['Mup'][(fitdata['Session']==s)&(fitdata['Neuron']==n)].iloc[0]
    sigp = fitdata['Sigp'][(fitdata['Session']==s)&(fitdata['Neuron']==n)].iloc[0] 
except:
    mup = pos[np.argmax(frs)]
    sigp = 20


#pseudosession fits
rs_pear_gauss = np.zeros(50)
rs_pear_log = np.zeros(50)
evsets = np.load('evidencegenerated.npy')

for i in range(50):
    evgen = evsets[i, :len(frs)]
    r_pear_gauss, r_pear_log = pseudofit(frs, pos, evgen.reshape(-1, 1), mup, sigp) 
    rs_pear_gauss[i] = r_pear_gauss
    rs_pear_log[i] = r_pear_log

#get gaussian and logistic fit
lower = np.max([(mup-.5*sigp), 0])
validis = (pos>lower)&(pos<(mup+.5*sigp))
validis = validis.reshape(-1)
frs = frs[validis]
ev = ev[validis]
m = np.min(ev)
M = np.max(ev)

try:
    mue, sige, a, c = fit_gauss(ev, frs, [0, 5, 1, 0], [m, 0, 0, 0], [M, 30, 10, 1])
    preds = gauss(ev, mue, sige, a, c)
    rgauss, p = stats.pearsonr(preds.reshape(-1), frs)

except:
    rgauss = 0
    mue = np.nan
    sige = np.nan
    a = np.nan
    c = np.nan

    
try:
    k, x0, alog, clog = fit_logistic(ev, frs, [0, 0, 1, 0], [-1, -15, 0, 0], [1, 15, 10, 1])
    preds = logistic(ev, k, x0, a, c)
    rlog, p = stats.pearsonr(preds.reshape(-1), frs)

except:
    rlog = 0
    k = np.nan
    x0 = np.nan
    alog = np.nan
    clog = np.nan


#comparison to pseudosessions
if np.mean(rs_pear_gauss)>rgauss:
    pgauss = 1
else:    
    stat = stats.ttest_1samp(rs_pear_gauss[~np.isnan(rs_pear_gauss)], rgauss)
    pgauss = stat.pvalue
  
if np.mean(rs_pear_log)>rlog:
    plog = 1
else:    
    stat = stats.ttest_1samp(rs_pear_log[~np.isnan(rs_pear_log)], rlog)
    plog = stat.pvalue
    
params = np.array([rgauss, pgauss, mue, sige, a, c, rlog, plog, k, x0, alog, clog, mup, sigp])

region, fdata, neuron = infile.split('/')    
np.save(region+'/paramfit-evonly/'+neuron, params)




                
