# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 10:19:35 2023

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
import seaborn as sns
from scipy.stats import sem
from sklearn.preprocessing import minmax_scale
import pickle

#demo
demo = True #True to run with example datasets, False to run for full data in structured folders

percentagefit = True #whether to use only top 80% of neurons for the plot
checkobs = True #whether to check for the number of observations for narrowly tuned cells
checkvar = True #whether to perform additional checks beyond count for narrowly tuned cells
outliervar = True #whether to use outliers as additional check (otherwise checks based on standard error of observations)

def get_obs(data, evs, poses, mue, mup):
    '''
    function to determined the number of observations, standard error, and number of outliers in the bin containing the fit evidence mean and fit position mean

    === inputs ===
    data: array of data with firing rates (1st column), positions (2nd column), and cumulative evidence (3rd column)
    evs: array of evidence bins
    poses: array of position bins
    mue: fit evidence mean
    mup: fit position mean
    '''

    #load data
    frs = data[:, 0]
    data = data[~np.isnan(frs), :] #remove data for which firing rate is nan
    frs = data[:, 0] #firing rates, 1st column
    pos = data[:, 1].reshape(-1, 1) #position, 2nd column
    ev = data[:, 2].reshape(-1, 1) #cumulative evidence, 3rd column
    frs = minmax_scale(frs)
       
    p = poses[np.argmin(np.abs(poses-mup))] #find nearest position bin to fit position mean
    e = evs[np.argmin(np.abs(evs-mue))] #find nearest evidence bin to fit evidence mean

    #only consider observations within the bin containing fit position and evidence mean
    vp = pos[:, 0]==p
    ve = ev[:, 0]==e
    os = vp & ve
    
    if outliervar: #if testing for number of outliers
        if sum(os)>0:
            #find number of outliers based on interquartile range
            q1 = np.percentile(frs[os], 25)
            q3 = np.percentile(frs[os], 75)
            n_outlier = len(np.where(frs[os]>(1.5*(q3-q1)+q3))[0])
            return sum(os), np.mean(frs[os]), n_outlier
        else:
            return sum(os), np.nan, 1 #if insufficient number of observations for percentiles, return 1 outlier
        

    return sum(os), np.mean(frs[os]), sem(frs[os]) #number of obs is the total meeting the criteria, mean in bin, standard error in bin
    
if not demo:
    regions = ['ACC', 'DMS', 'HPC', 'RSC']
else:
    regions = ['ACC']

for region in regions:

    #define evidence bins
    evs = np.arange(-15, 16)
    
    #define position bins corresponding to region
    if region in ['ACC', 'DMS']:
        poses = np.arange(-27.5, 302.5, 5)
    else:
        poses = np.arange(-30, 300, 5)    
    
    #load parameters from joint gaussian fit
    if not demo:
        fitparams = pd.read_csv(region+'/paramfit/'+region+'allfitparams.csv')
    else:
        fitparams = pd.read_csv('ExampleData/ACCparamfitexample.csv')
    
    #use fit pvalues to determine which fits were significant and consider only those cells
    pvals = fitparams['Pval'].values
    signs = fitparams['Neuron'].values[pvals<.05]
    sigsessions = fitparams['Session'].values[pvals<.05]
    allsigneurons = [(signs[i], sigsessions[i]) for i in range(len(signs))]


    #initialize needed data for these polots
    rs = np.zeros(len(allsigneurons))
    mups = np.zeros(len(allsigneurons))
    sigps = np.zeros(len(allsigneurons))
    mues = np.zeros(len(allsigneurons))
    normmues = np.zeros(len(allsigneurons))
    normsiges = np.zeros(len(allsigneurons))
    obs = np.zeros(len(allsigneurons))
    sems = np.zeros(len(allsigneurons))
    siges = np.zeros(len(allsigneurons))
    counts = 0
    percentagecounts = 0
    
    #normalize fit evidence mean and fit evidence standard deviation based on range of evidence observed at the mean
    mue = fitparams['Mue'].values
    sige = fitparams['Sige'].values
    muenorm = np.zeros(len(mue))
    sigenorm = np.zeros(len(mue))
    for i, e in enumerate(mue):
        if e<0: #normalize by maximum magnitude of evidence of the same sign
            muenorm[i] = mue[i]/np.abs(fitparams['MinE'].values[i])
        else:
            muenorm[i] = mue[i]/np.abs(fitparams['MaxE'].values[i])
        sigenorm[i] = sige[i]/(fitparams['MaxE'].values[i]-fitparams['MinE'].values[i]) #normalize by evidence range
    fitparams['NormMue'] = muenorm
    fitparams['NormSige'] = sigenorm
    
    for i, neuron in enumerate(allsigneurons): #for each neuron with significant evidence tuning
        n = neuron[0]
        s = neuron[1]            

        #load parameters corresponding to the neuron
        try:
            params = fitparams[(fitparams['Neuron']==n)& (fitparams['Session']==s)].iloc[0]
            r = params['Correlation']
            if np.isnan(r):
                r = 0
        except:
            r = 0
        rs[i] = r
        mups[i] = params['Mup'] #fit position mean
        sigps[i] = params['Sigp'] #fit position standard deviation
        normmues[i] = params['NormMue'] #normalized evidence mean
        normsiges[i] = params['NormSige'] #normalized evidence standard deviation
        mues[i] = params['Mue'] #fit evidence mean
        siges[i] = params['Sige'] #fit evidence standard deviations
        
        if params['Sige']<3 and not demo: #for narrowly tuned cells
            #get gaussian fit data
            ndata = np.load(region+'/firingdata/'+s+'neuron'+str(n)+'.npy')
            
            obval, mval, semval = get_obs(ndata, evs, poses, params['Mue'], params['Mup']) #get the observations in bin containing the fit mean evidence and position
            obs[i] = obval #number of observations
            sems[i] = semval #number of outliers if outliervar = True    
        else:
            obs[i] = np.inf #do not need to test number of observations for wide evidecne fits
            

    if percentagefit:                    
        q = np.percentile(rs, 20) #determine lowerbound on the top 80% of fits
    else:
        q = 0
        
    if checkobs: #lower bound to have at least 4 observations
        o = 3
    else:
        o=0
    
    if checkvar:
        if outliervar:
            maxsig = 1 #if testing for outliers, cannot be exactly one outlier
        else:
            maxsig = .2 #if testing for variance must be less than .2
    else:
        maxsig = np.inf
        
    #only consider cells that meet all criteria
    if outliervar:
        print(region+'>'+str(q))
        print(len(mups[(rs>q)&(obs>o)]))    
        mups = mups[(rs>q)&(obs>o)&(sems!=maxsig)]
        sigps = sigps[(rs>q)&(obs>o)&(sems!=maxsig)]
        normmues = normmues[(rs>q)&(obs>o)&(sems!=maxsig)]
        normsiges = normsiges[(rs>q)&(obs>o)&(sems!=maxsig)]
        mues = mues[(rs>q)&(obs>o)&(sems!=maxsig)]
        siges = siges[(rs>q)&(obs>o)&(sems!=maxsig)]
        semsnew = sems[(rs>q)&(obs>o)&(sems!=maxsig)]
        rsnew = rs[(rs>q)&(obs>o)&(sems!=maxsig)]
        print(len(mups))
        
    else:
        print(region+'>'+str(q))
        print(len(mups[(rs>q)&(obs>o)]))    
        mups = mups[(rs>q)&(obs>o)&(sems<maxsig)]
        sigps = sigps[(rs>q)&(obs>o)&(sems<maxsig)]
        normmues = normmues[(rs>q)&(obs>o)&(sems<maxsig)]
        normsiges = normsiges[(rs>q)&(obs>o)&(sems<maxsig)]
        mues = mues[(rs>q)&(obs>o)&(sems<maxsig)]
        siges = siges[(rs>q)&(obs>o)&(sems<maxsig)]
        semsnew = sems[(rs>q)&(obs>o)&(sems<maxsig)]
        rsnew = rs[(rs>q)&(obs>o)&(sems<maxsig)]
        print(len(mups))        
    
    #save out cells that meet criteria for later analyses
    if outliervar:
        newsigis = np.arange(len(allsigneurons))[(rs>q)&(obs>o)&(sems!=maxsig)]
        newsigneurons = []
        for idx in newsigis:
            newsigneurons.append(allsigneurons[idx])
        with open(region+'-nonoutliercells.p', "wb") as fp:   
            pickle.dump(newsigneurons, fp)
    else:
        newsigis = np.arange(len(allsigneurons))[(rs>q)&(obs>o)&(sems<maxsig)]
        newsigneurons = []
        for idx in newsigis:
            newsigneurons.append(allsigneurons[idx])
        with open(region+'-lowsemcells.p', "wb") as fp:   
            pickle.dump(newsigneurons, fp)
    
    #histogram of fit position means
    plt.figure()
    sns.histplot(mups, bins=np.arange(-50, 350, 15), stat='density', color = 'green', edgecolor = None, alpha=.3, kde=True, line_kws={'linewidth':3})
    plt.xlim([-50, 350])
    plt.ylabel('density')
    plt.title(region + r' $\mu_p$')
    plt.show()
 
    #histogram of fit position standard deviation
    plt.figure()
    sns.histplot(sigps, bins=np.arange(0, 200, 10), stat='density', color = 'green', edgecolor = None, alpha=.3, kde=True, line_kws={'linewidth':3})
    plt.xlim([0, 200])
    plt.ylabel('density')
    plt.title(region + r' $\sigma_p$')
    plt.show()

    #histogram of normalized fit evidence means
    plt.figure()
    #correct for side of evidence
    sns.histplot(-1*normmues[(np.abs(normmues)<2)&~np.isnan(normmues)], bins=np.arange(-2, 2, .1), stat='density', color = 'teal', edgecolor = None, alpha=.3, kde=True, line_kws={'linewidth':3})
    plt.xlim([-1.75, 1.75])
    plt.ylim([0, 1])
    plt.ylabel('density')
    plt.title(region + r' $\mu_e$')
    plt.axvline(-1, color = 'k', linestyle = '--')
    plt.axvline(1, color = 'k', linestyle = '--')
    plt.show()

    #histogram of normalized fit evidence standard deviations
    plt.figure()
    sns.histplot(normsiges[(normsiges<1.5)&~np.isnan(normsiges)], bins=np.arange(0, 3, .03), stat='density', color = 'teal', edgecolor = None, alpha=.3, kde=True, line_kws={'linewidth':3})
    plt.xlim([0, 1.5])
    plt.ylabel('density')
    plt.title(region + r' $\sigma_e$')
    plt.axvline(1, color = 'k', linestyle = '--')
    plt.ylim([0, 5])
    plt.show()
 
    #scatter plot of fit position means by fit evidence means, colored by normalized fit evidence standard deviation
    fig = plt.figure()
    big_ax = fig.add_axes([0.1, 0.1, 0.8, 0.55])
    #correct for side of evidence
    im = big_ax.scatter(mups, -1*mues, s=5, c = normsiges, cmap='jet', vmin=0, vmax=.5)
    plt.xlabel(r'$\mu_p$')
    plt.yticks([-15, -10, -5, 0 , 5, 10, 15])
    plt.ylabel(r'$\mu_e$')
    plt.xlim([-30, 300])
    plt.ylim([-15, 15])
    plt.axvline(0, color = 'gray', linestyle='--', linewidth=1)
    plt.axvline(200, color = 'gray', linestyle='--', linewidth=1)
    fig.colorbar(im)
    plt.title(region)
    plt.show()


    

