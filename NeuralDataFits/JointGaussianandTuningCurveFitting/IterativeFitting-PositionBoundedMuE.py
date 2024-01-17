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

pthresh = .05
region_threshold = .25 #.5 Ryan
region_width = 4
base_thresh = 0 #3 Ryan

blp = [-50, 0, 0, 0]
bup = [350, 200, 10, 1]

ble = [-20, 0, 0, 0]
bue = [20, 30, 10, 1]

def gauss(X, mu, sig, a, b):
    x = X[:, 0]
    y = X[:, 1]
    return a*y*np.exp((-(x-mu)**2)/(2*sig**2))+b
    
def fit_gauss(x, y, p0, bl, bu):
    popt, pcov = curve_fit(gauss, x, y, p0 = p0, bounds = (bl, bu))
    return popt

def get_ebounds(mup, pos, ev):
    validps = np.where(np.abs(pos-mup)<5)[0]
    if len(validps)>0:
        erange = ev[validps]
        bl = np.min(erange)-1
        bu = np.max(erange)+1 #previously 1.3 times
        if bl == bu: #ensure there is a range
            bl = bl-1
            bu = bu+1
    else:
        if mup<50:
            bu = 1
            bl = -1
        else:
            bu = 15
            bl = -15
    return [bl, 0, 0, 0], [bu, 30, 10, 1]

def full_function(pos, ev, mu_p, sig_p, mu_e, sig_e, a, b):
    return a*np.exp((-(pos-mu_p)**2)/(2*sig_p**2))*np.exp((-(ev-mu_e)**2)/(2*sig_e**2))+b

def get_cum_evidence(lefts, rights, pos):
    cum_ev = np.zeros(len(pos))
    for i, p in enumerate(pos):
        cum_ev[i] = np.sum(lefts<p)-np.sum(rights<p)
    return cum_ev

def get_regions(data, M, threshold, width, base_thresh):
    upreg = np.where(data>(threshold*M))[0]
    baseline = np.nanmean(data[~upreg])
    regions = []
    if len(upreg)==0:
        return regions
    last = upreg[0]
    i=1
    currentreg = [last]
    while i<(len(upreg)-1):
        curr = upreg[i]
        if curr == last+1:
            currentreg.append(curr)
        else:
            if len(currentreg)>width:
                regions.append(currentreg)
            currentreg = [curr]
        last = curr        
        i=i+1
    if len(currentreg)>width:
        if np.nanmean(data[currentreg])>base_thresh*baseline:
            regions.append(currentreg)
    return regions

def divide_LR(alldata, leftchoices, rightchoices, pthresh, region_threshold, region_width, basethresh):
    '''

    Parameters
    ----------
    alldata : neural data (trials x neuron x position)
    leftchoices : trials in which the animal went left
    rightchoices : trials in which the animal went right

    Returns
    -------
    indices of left preferrring neurons, indices of right preferring neurons, 
    and indices of neurons with no significant difference in response between
    the two choices

    '''
    avgdata = np.nanmean(alldata, axis=0) #neurons x position
    
    #transform data by substracting the minimum
    avgdata = avgdata - np.reshape(np.nanmin(avgdata, axis=1), (-1, 1))
    
    left_neurons = []
    right_neurons = []
    split_neurons = [] #neurons are still task modulated but not significant choice selective
    nonmod_neurons = [] #neurons with no peaks
    

    maxfire = np.nanmax(avgdata, axis=1)
    rightfires = alldata[rightchoices, :, :]
    leftfires = alldata[leftchoices, :, :]
    for i, m in enumerate(maxfire):
        upregions = get_regions(avgdata[i, :], m, region_threshold, region_width, base_thresh)
        left = False
        right = False
        for region in upregions:
            leftfiring = leftfires[:, i, region]
            leftactivity = np.nanmean(leftfiring, axis=1)
            rightfiring = rightfires[:, i, region]
            rightactivity = np.nanmean(rightfiring, axis=1)
            tval, pval = stats.ttest_ind(leftactivity, rightactivity)
            if pval <2*pthresh:
                if np.nanmean(leftactivity)>np.nanmean(rightactivity):
                    left = True
                else:
                    right = True
        if not (right and left):
            if right:
                right_neurons.append(i)
            elif left:
                left_neurons.append(i)
            else:
                if len(upregions)>0:
                    split_neurons.append(i)
                else:
                    nonmod_neurons.append(i) 
        else:
            if len(upregions)>0:
                split_neurons.append(i)
            else:
                nonmod_neurons.append(i)                  
    return np.array(left_neurons), np.array(right_neurons), np.array(split_neurons), np.array(nonmod_neurons)

def pseudofit(frs, pos, ev):
    s = np.ones((len(frs),1))
    dat = np.concatenate((pos, s), axis=1)
    mu0 = pos[np.argmax(frs)]
    a0 = np.max(frs)
    mu_p, sig_p, a, b = fit_gauss(dat, frs, [mu0, 20, a0, 0], blp, bup)
    
    iters = 0
    
    mu_e = ev[np.argmax(frs)]
    sig_e = 5
    
    while iters<25:
        
        #evidence fitting
        posscaling = gauss(np.concatenate((pos, s), axis=1), mu_p, sig_p, 1, 0)
        dat = np.concatenate((ev, posscaling.reshape(-1, 1)), axis=1)
        try:
            ble, bue = get_ebounds(mu_p, pos, ev)
            if mu_e< ble[0]:
                mu_e = ble+.01
            if mu_e>bue[0]:
                mu_e = bue-.01            
            mu_e, sig_e, a, b = fit_gauss(dat, frs, [mu_e, sig_e, a, b], ble, bue)
        except:
            print('poor evidence fit iteration:'+str(iters))
        
        #position fitting
        evscaling = gauss(np.concatenate((ev, s), axis =1), mu_e, sig_e, 1, 0)
        dat = np.concatenate((pos, evscaling.reshape(-1, 1)), axis=1)
        try:
            mu_p, sig_p, a, b = fit_gauss(dat, frs, [mu_p, sig_p, a, b], blp, bup)
        except:
            print('poor position fit iteration:'+str(iters))
        
        iters = iters+1
        
        
    preds = full_function(pos, ev, mu_p, sig_p, mu_e, sig_e, a, b)
    r, p = stats.pearsonr(preds.reshape(-1), frs) 
    
    return r

arguments = sys.argv[1:]
infile = arguments[0]
infile = infile.split('.')[0]+'.npy'

# preprocessing to remove nandata
data = np.load(infile)
frs = data[:, 0]
data = data[~np.isnan(frs), :]
frs = data[:, 0]

frs = minmax_scale(frs)
pos = data[:, 1].reshape(-1, 1)
ev = data[:, 2].reshape(-1, 1)

# initialize position fit
s = np.ones((len(frs),1))
dat = np.concatenate((pos, s), axis=1)
mu0 = pos[np.argmax(frs)]
a0 = np.max(frs)
mu_p, sig_p, a, b = fit_gauss(dat, frs, [mu0, 20, a0, 0], blp, bup)

deltaparam = np.inf
iters = 0

mu_e = ev[np.argmax(frs)]
sig_e = 5

deltas = []
errors = []
perror = []

paramsprev = np.zeros((4, 1))

while iters<25:
    paramprev = np.array([mu_p, sig_p, mu_e, sig_e])
    paramsprev = np.concatenate((paramsprev, paramprev.reshape(-1, 1)), axis=1)
    
    #evidence fitting
    posscaling = gauss(np.concatenate((pos, s), axis=1), mu_p, sig_p, 1, 0)
    dat = np.concatenate((ev, posscaling.reshape(-1, 1)), axis=1)
    try:
        ble, bue = get_ebounds(mu_p, pos, ev)
        if mu_e< ble[0]:
            mu_e = ble+.01
        if mu_e>bue[0]:
            mu_e = bue-.01
        mu_e, sig_e, a, b = fit_gauss(dat, frs, [mu_e, sig_e, a, b], ble, bue)
    except:
        print('poor evidence fit in iter'+str(iters))
    
    #position fitting
    evscaling = gauss(np.concatenate((ev, s), axis =1), mu_e, sig_e, 1, 0)
    dat = np.concatenate((pos, evscaling.reshape(-1, 1)), axis=1)
    try:
        mu_p, sig_p, a, b = fit_gauss(dat, frs, [mu_p, sig_p, a, b], blp, bup)
    except:
        print('poor position fit in iter'+str(iters))
    
    paramnew = np.array([mu_p, sig_p, mu_e, sig_e])
    
    deltaparam = np.sum((paramnew-paramprev)**2)
    deltas.append(deltaparam)
    
    #perror.append(np.sum((paramnew-trueparam)**2))
    
    preds = full_function(pos, ev, mu_p, sig_p, mu_e, sig_e, a, b)
    error = np.sum((frs-preds)**2)
    errors.append(error)
    iters = iters+1
    r, p = stats.pearsonr(preds.reshape(-1), frs)

rs_pear = np.zeros(50)
evsets = np.load('evidencegenerated.npy')

for i in range(50):
    evgen = evsets[i, :len(frs)]
    r_pear = pseudofit(frs, pos, evgen.reshape(-1, 1)) 
    rs_pear[i] = r_pear

if np.mean(rs_pear)>r:
    paramnew = np.concatenate((paramnew, np.array([r, 1])))
else:    
    stat = stats.ttest_1samp(rs_pear, r)
    paramnew = np.concatenate((paramnew, np.array([r, stat.pvalue])))


region, fdata, neuron = infile.split('/')    
np.save(region+'/paramfit-boundedmue/'+neuron, paramnew)



                
