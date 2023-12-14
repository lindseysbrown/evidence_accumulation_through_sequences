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
from scipy.stats import f_oneway, kurtosis, skew, sem
from sklearn.preprocessing import minmax_scale
import pickle

sigthresh = .01

percentagefit = True
checkobs = True
checkvar = True
outliervar = True
plotHPC = False

def get_obs(data, evs, poses, mue, mup):
    frs = data[:, 0]
    data = data[~np.isnan(frs), :]
    frs = data[:, 0]
    pos = data[:, 1].reshape(-1, 1)
    ev = data[:, 2].reshape(-1, 1)
    frs = minmax_scale(frs)
       
    p = poses[np.argmin(np.abs(poses-mup))]
    e = evs[np.argmin(np.abs(evs-mue))]

    vp = pos[:, 0]==p
    ve = ev[:, 0]==e
    os = vp & ve
    
    if outliervar:
        if sum(os)>0:
            q1 = np.percentile(frs[os], 25)
            q3 = np.percentile(frs[os], 75)
            n_outlier = len(np.where(frs[os]>(1.5*(q3-q1)+q3))[0])
            return sum(os), np.mean(frs[os]), n_outlier
        else:
            return sum(os), np.nan, 1
        

    return sum(os), np.mean(frs[os]), sem(frs[os]) 

def plot_obs(data, evs, poses, mue, mup, i, ax):
    frs = data[:, 0]
    data = data[~np.isnan(frs), :]
    frs = data[:, 0]
    pos = data[:, 1].reshape(-1, 1)
    ev = data[:, 2].reshape(-1, 1)
    frs = minmax_scale(frs)
       
    p = poses[np.argmin(np.abs(poses-mup))]
    e = evs[np.argmin(np.abs(evs-mue))]

    vp = pos[:, 0]==p
    ve = ev[:, 0]==e
    os = vp & ve

    ax.scatter(i*np.ones(len(frs[os])), frs[os], s = 3) 
    
def plot_split(data, evs, poses, mue, mup, neurnum):
    frs = data[:, 0]
    n_trials = int(len(frs)/len(poses))
    oddts = []
    events = []
    for t in range(n_trials):
        if t%2==0:
            events = events+list(t*len(poses)+np.arange(len(poses)))
        else:
            oddts = oddts+list(t*len(poses)+np.arange(len(poses)))
    
    ip = np.argmin(np.abs(mup-poses))
    ie = np.argmin(np.abs(mue-evs))
    
    dataodd = data[oddts, :]
    dataeven = data[events, :]
    
    for k, d in enumerate([dataodd, dataeven]):
        frs = d[:, 0]
        d = d[~np.isnan(frs), :]
        frs = d[:, 0]
        pos = d[:, 1].reshape(-1, 1)
        ev = d[:, 2].reshape(-1, 1)
        frs = minmax_scale(frs)
       
        obs = np.zeros((len(evs), len(poses)))
        sems = np.zeros((len(evs), len(poses)))
        counts = np.zeros((len(evs), len(poses)))
        for i, e in enumerate(evs):
            for j, p in enumerate(poses):
                vp = pos[:, 0]==p
                ve = ev[:, 0]==e
                os = vp & ve
                counts[i][j] = sum(os)
                if sum(os)>0:
                    obs[i][j] = np.mean(frs[os])
                    sems[i][j] = sem(frs[os])
                else:
                    obs[i][j] = np.nan
                    sems[i][j] = np.nan  
                    
        plt.figure()
        plt.imshow(obs, cmap = 'Purples', interpolation = 'none', vmin = 0, vmax=.6)
        plt.title('Neuron '+str(neurnum)+' - '+['Odd Trials', 'Even Trials'][k])
        plt.scatter(ip, ie, color = 'red', s=2)


def gauss(x, mu, sig):
    return np.exp((-(x-mu)**2)/(2*sig**2))

def get_predsatmax(data, mu_p, sig_p, mu_e, sig_e):
    frs = data[:, 0]
    pos = data[:, 1].reshape(-1, 1)
    ev = data[:, 2].reshape(-1, 1)
    P = gauss(pos, mu_p, sig_p)
    E = gauss(ev, mu_e, sig_e)
    X = P*E
    lr = LinearRegression()
    lr.fit(X[~np.isnan(frs)], frs[~np.isnan(frs)])
    Enew = gauss(np.arange(-15, 15), mu_e, sig_e)
    preds = lr.predict(Enew.reshape(-1, 1))
    return preds



regions = ['ACC', 'DMS', 'HPC', 'RSC', 'V1']

for region in regions:
    evs = np.arange(-15, 16)
    
    if region in ['ACC', 'DMS']:
        poses = np.arange(-27.5, 302.5, 5)
    else:
        poses = np.arange(-30, 300, 5)    
    
    fitparams = pd.read_csv(region+'/paramfit/'+region+'allfitparams-boundedmue-NEW.csv')
    
    pvals = fitparams['Pval'].values
    
    signs = fitparams['Neuron'].values[pvals<.05]
    sigsessions = fitparams['Session'].values[pvals<.05]
    allsigneurons = [(signs[i], sigsessions[i]) for i in range(len(signs))]


    
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
    
    mue = fitparams['Mue'].values
    sige = fitparams['Sige'].values
    muenorm = np.zeros(len(mue))
    sigenorm = np.zeros(len(mue))
    for i, e in enumerate(mue):
        if e<0:
            muenorm[i] = mue[i]/np.abs(fitparams['MinE'].values[i])
            #muenorm[i] = mue[i]/np.abs(fitparams['MinERaw'].values[i])
        else:
            muenorm[i] = mue[i]/np.abs(fitparams['MaxE'].values[i])
            #muenorm[i] = mue[i]/np.abs(fitparams['MaxERaw'].values[i])
        sigenorm[i] = sige[i]/(fitparams['MaxE'].values[i]-fitparams['MinE'].values[i])
        #sigenorm[i] = sige[i]/(fitparams['MaxERaw'].values[i]-fitparams['MinERaw'].values[i])
    fitparams['NormMue'] = muenorm
    fitparams['NormSige'] = sigenorm
    

    
    
    for i, neuron in enumerate(allsigneurons):
        n = neuron[0]
        s = neuron[1]
                   
        try:
            params = fitparams[(fitparams['Neuron']==n)& (fitparams['Session']==s)].iloc[0]
            r = params['Correlation']
            if np.isnan(r):
                r = 0
        except:
            r = 0

        rs[i] = r
        mups[i] = params['Mup']
        sigps[i] = params['Sigp']
        normmues[i] = params['NormMue']
        normsiges[i] = params['NormSige']
        mues[i] = params['Mue']
        siges[i] = params['Sige']
        
        if params['Sige']<3:
            try:
                #get gaussian fit data
                ndata = np.load(region+'/firingdata/'+s+'neuron'+str(n)+'.npy')
            except:
                ndata = np.load(region+'/firingdata-split/'+s+'neuron'+str(n)+'.npy')
            
            obval, mval, semval = get_obs(ndata, evs, poses, params['Mue'], params['Mup']) 
            obs[i] = obval
            sems[i] = semval
            
            
        else:
            obs[i] = np.inf
            

    if percentagefit:                    
        q = np.percentile(rs, 20)
    else:
        q = 0
        
    if checkobs:
        o = 3
    else:
        o=0
    
    if checkvar:
        if outliervar:
            maxsig = 1
        else:
            maxsig = .2
    else:
        maxsig = np.inf
        
    
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
    
    
    if outliervar:
        newsigis = np.arange(len(allsigneurons))[(rs>q)&(obs>o)&(sems!=maxsig)]
        newsigneurons = []
        for idx in newsigis:
            newsigneurons.append(allsigneurons[idx])
        with open(region+'-nonoutliercells.p', "wb") as fp:   #Pickling
            pickle.dump(newsigneurons, fp)
    else:
        newsigis = np.arange(len(allsigneurons))[(rs>q)&(obs>o)&(sems<maxsig)]
        newsigneurons = []
        for idx in newsigis:
            newsigneurons.append(allsigneurons[idx])
        with open(region+'-lowsemcells.p', "wb") as fp:   #Pickling
            pickle.dump(newsigneurons, fp)
    
    
    
    if region=='HPC' and plotHPC:
        matchplots = 0
        fig, ax = plt.subplots()
        if outliervar:
            newsigis = np.arange(len(allsigneurons))[(rs>q)&(obs>o)&(sems!=maxsig)]
        else:
            newsigis = np.arange(len(allsigneurons))[(rs>q)&(obs>o)&(sems<maxsig)]
        newsigis = newsigis[normsiges<.2]
        newsigneurons = []
        for idx in newsigis:
            newsigneurons.append(allsigneurons[idx])
            
        if outliervar:
            with open('HPC-nonoutliercells.p', "wb") as fp:   #Pickling
                pickle.dump(newsigneurons, fp)
        else:
            if checkvar:
                with open('HPC-lowsemcells.p', "wb") as fp:   #Pickling
                    pickle.dump(newsigneurons, fp)                
        
        for i, neuron in enumerate(newsigneurons[:75]):
            n = neuron[0]
            s = neuron[1]
                   
            try:
                params = fitparams[(fitparams['Neuron']==n)& (fitparams['Session']==s)].iloc[0]
                r = params['Correlation']
                if np.isnan(r):
                    r = 0
            except:
                r = 0
    
            try:
                #get gaussian fit data
                ndata = np.load(region+'/firingdata/'+s+'neuron'+str(n)+'.npy')
            except:
                ndata = np.load(region+'/firingdata-split/'+s+'neuron'+str(n)+'.npy')    
                    
            plot_obs(ndata, evs, poses, params['Mue'], params['Mup'], i, ax)
            
            if (matchplots<0) and (normsiges[i]<.2):
                plot_split(ndata, evs, poses, params['Mue'], params['Mup'], i)
                matchplots = matchplots+1
    
    plt.figure()
    sns.histplot(mups, bins=np.arange(-50, 350, 15), stat='density', color = 'green', edgecolor = None, alpha=.3, kde=True, line_kws={'linewidth':3})
    plt.xlim([-50, 350])
    plt.ylabel('density')
    plt.title(region + r' $\mu_p$')
    if outliervar:
        plt.savefig('Figure4Plots/RestrictedGaussPlots/'+region+'mup-outlier-mean.pdf')
    else:
        plt.savefig('Figure4Plots/RestrictedGaussPlots/'+region+'mup-lowsem.pdf')
 
    '''
    plt.figure()
    sns.histplot(siges[siges<20], stat='density')
    plt.title('Raw Sig E')
    '''
 
    plt.figure()
    sns.histplot(sigps, bins=np.arange(0, 200, 10), stat='density', color = 'green', edgecolor = None, alpha=.3, kde=True, line_kws={'linewidth':3})
    plt.xlim([0, 200])
    plt.ylabel('density')
    plt.title(region + r' $\sigma_p$')
    if outliervar:
        plt.savefig('Figure4Plots/RestrictedGaussPlots/'+region+'sigp-outlier-mean.pdf')
    else:
        plt.savefig('Figure4Plots/RestrictedGaussPlots/'+region+'sigp-lowsem.pdf')        

    plt.figure()
    #correct for side of evidence
    sns.histplot(-1*normmues[(np.abs(normmues)<2)&~np.isnan(normmues)], bins=np.arange(-2, 2, .1), stat='density', color = 'teal', edgecolor = None, alpha=.3, kde=True, line_kws={'linewidth':3})
    plt.xlim([-1.75, 1.75])
    plt.ylim([0, 1])
    plt.ylabel('density')
    plt.title(region + r' $\mu_e$')
    plt.axvline(-1, color = 'k', linestyle = '--')
    plt.axvline(1, color = 'k', linestyle = '--')
    if outliervar:
        plt.savefig('Figure4Plots/RestrictedGaussPlots/'+region+'mue-outlier-mean.pdf')
    else:
        plt.savefig('Figure4Plots/RestrictedGaussPlots/'+region+'mue-lowsem.pdf') 

    plt.figure()
    sns.histplot(normsiges[(normsiges<1.5)&~np.isnan(normsiges)], bins=np.arange(0, 3, .03), stat='density', color = 'teal', edgecolor = None, alpha=.3, kde=True, line_kws={'linewidth':3})
    plt.xlim([0, 1.5])
    plt.ylabel('density')
    plt.title(region + r' $\sigma_e$')
    plt.axvline(1, color = 'k', linestyle = '--')
    plt.ylim([0, 5])
    if outliervar:
        plt.savefig('Figure4Plots/RestrictedGaussPlots/'+region+'sige-outlier-mean.pdf')
    else:
        plt.savefig('Figure4Plots/RestrictedGaussPlots/'+region+'sige-lowsem.pdf')    
 
    
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
    if outliervar:
        plt.savefig('Figure4Plots/RestrictedGaussPlots/'+region+'scatter-outlier-mean.pdf')
    else:
        plt.savefig('Figure4Plots/RestrictedGaussPlots/'+region+'scatter-lowsem.pdf')


    

