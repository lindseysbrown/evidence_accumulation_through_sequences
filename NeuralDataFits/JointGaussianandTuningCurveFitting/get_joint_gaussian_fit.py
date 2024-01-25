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

#bounds on position parameters
blp = [-50, 0, 0, 0] #lower bounds
bup = [350, 200, 10, 1] #upper bounds

#initial bounds on evidence parameters
ble = [-20, 0, 0, 0] #lower bounds
bue = [20, 30, 10, 1] #upper bounds

def gauss(X, mu, sig, a, b):
    '''
    returns gaussian function with scaling

    === inputs ===
    X: array consisting of current data set for which fitting gaussian parameters (1st column) and scaled values of other variable (2nd column)
    mu: mean
    sig: standard deviation
    a: scale of gaussian function
    b: vertical shift of gaussian function
    '''
    x = X[:, 0]
    y = X[:, 1]
    return a*y*np.exp((-(x-mu)**2)/(2*sig**2))+b
    
def fit_gauss(x, y, p0, bl, bu):
    '''
    use scipy's curve_fit to fit the parameters (a, mu, sig, b) of a gaussian function of the form a*y*exp((-(x-mu)**2)/(2*sig**2))+b

    ===inputs===
    x: input data to gaussian, array consisting of current data set for which fitting gaussian parameters (1st column) and scaled values of other variable (2nd column)
    y: desired output values for each input
    p0: initial guess at the parameters
    bl: lower bounds on the parameters
    bu: upper bounds on the parameters

    ===outputs===
    popt: array containing the fit parameters (mu, sig, a, b)
    '''
    popt, pcov = curve_fit(gauss, x, y, p0 = p0, bounds = (bl, bu))
    return popt

def get_ebounds(mup, pos, ev):
    '''
    get bounds to use for evidence to correspond to observed range at the current fit position
    === inputs ===
    mup: current fit position mean
    pos: list of positions
    ev: list of cumulative evidence at each position in pos

    === outputs ===
    list of lower bounds on evidence parameters
    list of upper bounds on evidence parameters 
    '''
    validps = np.where(np.abs(pos-mup)<5)[0] #get positions within same position bin as fit mean position
    if len(validps)>0:
        erange = ev[validps] #find evidence levels at the mean position
        bl = np.min(erange)-1 #set lower bound based on minimum of observed evidence levels
        bu = np.max(erange)+1 #set upper bound based on maximum of observed evidence levels
        if bl == bu: #ensure there is a range
            bl = bl-1
            bu = bu+1
    else: #if fit position beyond bounds of maze
        if mup<50: #evidence mean in [-1, 1] if positino mean before maze starts
            bu = 1
            bl = -1
        else: #evidence mean in [-15,15] if position mean beyond the end of the maze
            bu = 15
            bl = -15
    return [bl, 0, 0, 0], [bu, 30, 10, 1]

def full_function(pos, ev, mu_p, sig_p, mu_e, sig_e, a, b):
    '''
    evaluate function with fit parameters
    === inputs ===
    pos: positions
    ev: cumulative evidence level
    mu_p: position mean
    sig_p: position standard deviation
    mu_e: evidence mean
    sig_e: evidence standard deviation
    a: scaling of gaussian multiplication
    b: vertical shift
    '''
    return a*np.exp((-(pos-mu_p)**2)/(2*sig_p**2))*np.exp((-(ev-mu_e)**2)/(2*sig_e**2))+b #aP(p)E(e)+b

def pseudofit(frs, pos, ev):
    '''
    repeat fit procedure for pseudosession with firing rates (frs) and positions (pos) with a randomly generated set of cumulative evidence (ev)

    === outputs ===
    correlation between firing rate and predicted firing rates from parameters fit to pseudosession
    '''
    #initial position fit
    s = np.ones((len(frs),1)) #for first iteration, assume flat evidence tuning
    dat = np.concatenate((pos, s), axis=1)
    mu0 = pos[np.argmax(frs)] #initial guess for the mean is where the maximum firing rate occurs
    a0 = np.max(frs)
    mu_p, sig_p, a, b = fit_gauss(dat, frs, [mu0, 20, a0, 0], blp, bup) #get first fit of position parameters
    
    iters = 0
    
    #initial guess for evidence parameters
    mu_e = ev[np.argmax(frs)]
    sig_e = 5
    
    while iters<25: #repeat for 25 iterations
        
        #evidence fitting
        posscaling = gauss(np.concatenate((pos, s), axis=1), mu_p, sig_p, 1, 0) #find modulated position scaling P(p)
        dat = np.concatenate((ev, posscaling.reshape(-1, 1)), axis=1)
        try:
            ble, bue = get_ebounds(mu_p, pos, ev) #get evidence bounds based on current position tuning
            if mu_e< ble[0]: #move current estimate of mu_e to be within bounds
                mu_e = ble+.01
            if mu_e>bue[0]:
                mu_e = bue-.01            
            mu_e, sig_e, a, b = fit_gauss(dat, frs, [mu_e, sig_e, a, b], ble, bue) #get evidence parameters
        except:
            print('poor evidence fit iteration:'+str(iters))
        
        #position fitting
        evscaling = gauss(np.concatenate((ev, s), axis =1), mu_e, sig_e, 1, 0) #find modulated evidence scaling E(e)
        dat = np.concatenate((pos, evscaling.reshape(-1, 1)), axis=1)
        try:
            mu_p, sig_p, a, b = fit_gauss(dat, frs, [mu_p, sig_p, a, b], blp, bup) #get position parameters
        except:
            print('poor position fit iteration:'+str(iters))
        
        iters = iters+1
        
        
    preds = full_function(pos, ev, mu_p, sig_p, mu_e, sig_e, a, b) #get predictions for fit parameters
    r, p = stats.pearsonr(preds.reshape(-1), frs) #get corrleation between predictions and observed firing rates
    
    return r

arguments = sys.argv[1:]
infile = arguments[0]
infile = infile.split('.')[0]

# preprocessing to remove nandata
data = np.load(infile) #load array of firing rates (1st column), positions (2nd column), and cumulative evidence (3rd column)
frs = data[:, 0]
data = data[~np.isnan(frs), :] #remove data where firing rate is nan
frs = data[:, 0]

frs = minmax_scale(frs) #rescale firing rates so that a and b can be bounded
pos = data[:, 1].reshape(-1, 1) #get positions
ev = data[:, 2].reshape(-1, 1) #get cumulative evidences

# initialize position fit
s = np.ones((len(frs),1)) #for first iteration, assume flat evidence tuning
dat = np.concatenate((pos, s), axis=1)
mu0 = pos[np.argmax(frs)] #initial guess for the mean is where the maximum firing rate occurs
a0 = np.max(frs)
mu_p, sig_p, a, b = fit_gauss(dat, frs, [mu0, 20, a0, 0], blp, bup) #get first fit of position parameters

deltaparam = np.inf #track how much parameters chagne between iterations
iters = 0

#initial guess for evidence parameters
mu_e = ev[np.argmax(frs)] 
sig_e = 5

deltas = []
errors = []
perror = []

paramsprev = np.zeros((4, 1))

while iters<25: #iteratively fit postion and evidence parameters for 25 iterations
    paramprev = np.array([mu_p, sig_p, mu_e, sig_e])
    paramsprev = np.concatenate((paramsprev, paramprev.reshape(-1, 1)), axis=1) #keep track of previous parameter fits
    
    #evidence fitting
    posscaling = gauss(np.concatenate((pos, s), axis=1), mu_p, sig_p, 1, 0) #find modulated position scaling P(p)
    dat = np.concatenate((ev, posscaling.reshape(-1, 1)), axis=1)
    try:
        ble, bue = get_ebounds(mu_p, pos, ev) #get evidence bounds based on current position tuning
        if mu_e< ble[0]: #move current estimate of mu_e to be within bounds
            mu_e = ble+.01
        if mu_e>bue[0]:
            mu_e = bue-.01
        mu_e, sig_e, a, b = fit_gauss(dat, frs, [mu_e, sig_e, a, b], ble, bue) #get evidence parameters
    except:
        print('poor evidence fit in iter'+str(iters))
    
    #position fitting
    evscaling = gauss(np.concatenate((ev, s), axis =1), mu_e, sig_e, 1, 0) #find modulated evidence scaling E(e)
    dat = np.concatenate((pos, evscaling.reshape(-1, 1)), axis=1)
    try:
        mu_p, sig_p, a, b = fit_gauss(dat, frs, [mu_p, sig_p, a, b], blp, bup) #get position parameters
    except:
        print('poor position fit in iter'+str(iters))
    
    paramnew = np.array([mu_p, sig_p, mu_e, sig_e])
    
    deltaparam = np.sum((paramnew-paramprev)**2)
    deltas.append(deltaparam)
    
    #evaluate the fit for the full function
    preds = full_function(pos, ev, mu_p, sig_p, mu_e, sig_e, a, b) #predicted firing rates from fit parameters
    error = np.sum((frs-preds)**2)
    errors.append(error)
    iters = iters+1
    r, p = stats.pearsonr(preds.reshape(-1), frs) #correlation between predicted and true firing rates

rs_pear = np.zeros(50)
evsets = np.load('evidencegenerated.npy') #load generated evidence for pseudosessions

for i in range(50): 
    evgen = evsets[i, :len(frs)]
    r_pear = pseudofit(frs, pos, evgen.reshape(-1, 1)) #fit parameters for each pseudosession and get fit correlation
    rs_pear[i] = r_pear

#test if true fit is significantly better than pseudosession fits
if np.mean(rs_pear)>r: 
    paramnew = np.concatenate((paramnew, np.array([r, 1]))) #if r is worse than the mean of the pseudosessions set pval to 1
else:    
    stat = stats.ttest_1samp(rs_pear, r)
    paramnew = np.concatenate((paramnew, np.array([r, stat.pvalue]))) #determine significance by t-test


region, fdata, neuron = infile.split('/')    
np.save(region+'/paramfit/'+neuron, paramnew) #save parameters for use in further analysis



                
