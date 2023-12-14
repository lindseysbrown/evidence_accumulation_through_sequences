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
from scipy import stats
from scipy.stats import f_oneway, kurtosis, skew, sem, rankdata
from scipy.optimize import curve_fit
import diptest
import pandas as pd
import matplotlib.gridspec as gs
from sklearn.preprocessing import minmax_scale
import pickle


sigthresh = .05

earlyexamples = {'ACC': (10, 'dFF_tetO_8_08022021_T10processedOutput'), 'DMS': (4, 'dFF_scott_d2_857_20190426processedOutput'), 'HPC': (50, 'nicFR_E39_20171103'), 'RSC': (3,'nicFR_k50_20160811_RSM_400um_114mW_zoom2p2processedFR'), 'V1':(8,'nicFR_k56_20161004')}

lateexamples = {'ACC': (153, 'dFF_tetO_8_07282021_T10processedOutput'), 'DMS':(7, 'dFF_scott_a2a_64_11072019processedOutput'), 'HPC':(25, 'nicFR_E43_20170802'), 'RSC': (44,'nicFR_k31_20160114_RSM2_350um_98mW_zoom2p2processedFR'), 'V1':(14,'nicFR_k53_20161205')}

delayexamples = {'ACC': (17, 'dFF_tetO_8_08052021_T11processedOutput'), 'DMS':(1,'dFF_scott_d1_67_20190418processedOutput'), 'HPC':(118, 'nicFR_E22_20170227'), 'RSC': (10, 'nicFR_k42_20160512_RSM_150um_65mW_zoom2p2processedFR'), 'V1':(35,'nicFR_k53_20161205')}                 



specialcolors = {0:'magenta', 1: 'cyan', 2: 'orange'}


def gauss(x, mu, sig):
    return np.exp((-(x-mu)**2)/(2*sig**2))

def logistic(x, k, x0):
    return 1/(1+np.exp(-k*(x-x0)))


def get_obs(data, evs, poses):
    frs = data[:, 0]
    data = data[~np.isnan(frs), :]
    frs = data[:, 0]
    pos = data[:, 1].reshape(-1, 1)
    ev = data[:, 2].reshape(-1, 1)
    frs = minmax_scale(frs)
   
    obs = np.zeros((len(evs), len(poses)))
    sems = np.zeros((len(evs), len(poses)))
    for i, e in enumerate(evs):
        for j, p in enumerate(poses):
            vp = pos[:, 0]==p
            ve = ev[:, 0]==e
            os = vp & ve
            if sum(os)>0:
                obs[i][j] = np.mean(frs[os])
                sems[i][j] = sem(frs[os])
            else:
                obs[i][j] = np.nan
                sems[i][j] = np.nan  
    return obs, sems


def get_preds(data, mu_p, sig_p, mu_e, sig_e, evs, poses):
    frs = data[:, 0]
    data = data[~np.isnan(frs), :]
    frs = data[:, 0]
    pos = data[:, 1].reshape(-1, 1)
    ev = data[:, 2].reshape(-1, 1)
    frs = minmax_scale(frs)
    P = gauss(pos, mu_p, sig_p)
    E = gauss(ev, mu_e, sig_e)
    X = P*E
    lr = LinearRegression()
    lr.fit(X[~np.isnan(frs)], frs[~np.isnan(frs)])
    a = lr.coef_
    b = lr.intercept_
    preds = np.zeros((len(evs), len(poses)))
    for i, e in enumerate(evs):
        for j, p in enumerate(poses):
            preds[i][j] = a*gauss(e, mu_e, sig_e)*gauss(p, mu_p, sig_p)+b
    return preds

def get_preds_logistic(data, mu_p, sig_p, k, x0, evs, poses):
    frs = data[:, 0]
    data = data[~np.isnan(frs), :]
    frs = data[:, 0]
    pos = data[:, 1].reshape(-1, 1)
    ev = data[:, 2].reshape(-1, 1)
    frs = minmax_scale(frs)
    P = gauss(pos, mu_p, sig_p)
    E = logistic(ev, k, x0)
    X = P*E
    lr = LinearRegression()
    lr.fit(X[~np.isnan(frs)], frs[~np.isnan(frs)])
    a = lr.coef_
    b = lr.intercept_
    preds = np.zeros((len(evs), len(poses)))
    for i, e in enumerate(evs):
        for j, p in enumerate(poses):
            preds[i][j] = a*gauss(p, mu_p, sig_p)*logistic(e, k, x0)+b
    return preds   

regions = ['ACC', 'DMS', 'HPC', 'RSC', 'V1']

individual_plots = False

for region in regions[:-1]:
    with open(region+"-nonoutliercells.p", "rb") as fp:   #Pickling
        nstoplot = pickle.load(fp)
    
    evs = np.arange(-15, 16)
    
    if region in ['ACC', 'DMS']:
        poses = np.arange(-27.5, 302.5, 5)
    else:
        poses = np.arange(-30, 300, 5)
    
    fitparams = pd.read_csv(region+'/paramfit/'+region+'allfitparams-evonly-Cueonly.csv')
    
    keep = np.zeros(len(fitparams))
    for i in range(len(fitparams)):
        n = (fitparams['Neuron'].values[i], fitparams['Session'].values[i])
        if n in nstoplot:
            keep[i] = 1
            
    fitparams['Keep'] = keep
    
    fitparams = fitparams[fitparams['Keep']>0]
    
    fitparamsgauss = pd.read_csv(region+'/paramfit/'+region+'allfitparams-boundedmue-NEW.csv')

    keep = np.zeros(len(fitparamsgauss))
    for i in range(len(fitparamsgauss)):
        n = (fitparamsgauss['Neuron'].values[i], fitparamsgauss['Session'].values[i])
        if n in nstoplot:
            keep[i] = 1
            
    fitparamsgauss['Keep'] = keep
    
    fitparamsgauss = fitparamsgauss[fitparamsgauss['Keep']>0]
    
    
    fitlog = fitparams['rlog'].values
    fitgauss = fitparams['rgauss'].values
    
    siglog = fitparams['siglog'].values<.05
    siggauss = fitparams['siggauss'].values<.05
    
    mups = [float(m.replace('[','').replace(']', '')) for m in fitparams['Mup']]
        

    mues = fitparams['Mue'].values
    siges = fitparams['Sige'].values
    
    ns = fitparams['Neuron'].values
    ss = fitparams['Session'].values

    
    x0s = fitparams['x0'].values
    ks = fitparams['k'].values

    Mes = fitparams['MaxEMean'].values
    mes = fitparams['MinEMean'].values
            
    #replace nonsignifcant fits   
    fitlogcomp = fitlog.copy()
    fitlogcomp[siglog<1] = 0
        
    fitgausscomp = fitgauss.copy()
    fitgausscomp[siggauss<1] = 0
    

    
    #bestfits = np.argsort(fitparamsgauss['Correlation'].values)[::-1]
    bestfits = np.argsort(np.nanmax(np.vstack((fitlogcomp, fitgausscomp)), axis=0))[::-1]
    
        
    totalplots = 0
    
    numcurves = {'ACC':15, 'DMS':15, 'HPC':15, 'RSC':15, 'V1':15}
        
    toplotx = []
    toploty = []
    toplotmues = []
    toplotcolor = []
    widths = []
    
    bests = []
    
    plt.figure()
    for i in bestfits:
        if (totalplots < numcurves[region]) and (mups[i]>200):
            fits = np.array([fitlogcomp[i], fitgausscomp[i]])
            best = np.argmax(fits)
            bests.append(best)
            
            Me = Mes[i]
            me = mes[i]
            mue = mues[i]
            
            
            if best==0:
                k = ks[i]
                x0 = x0s[i]
                
                if Me-me>15:
                    evrange = np.arange(me,Me+1)
                    if individual_plots:
                        plt.figure()
                        plt.plot(-1*evrange, minmax_scale(logistic(evrange, k, x0)))
                    totalplots = totalplots+1
                    
                    if mue<0:
                        toplotx.append(-1*evrange)
                        toploty.append(minmax_scale(logistic(evrange, k, x0)))
                        toplotcolor.append('r')
                        toplotmues.append(mue)
                    else:
                        toplotx.append(-1*evrange)
                        toploty.append(minmax_scale(logistic(evrange, k, x0)))
                        toplotcolor.append('b')
                        toplotmues.append(mue)
                    
                
            if best ==1:
                sige = siges[i]
                if Me-me>15:
                    evrange = np.arange(me,Me+1)
                    if individual_plots:
                        plt.figure()
                        plt.plot(-1*evrange, minmax_scale(gauss(evrange, mue, sige)))
                    totalplots = totalplots+1
                    
                    if mue<0:
                        toplotx.append(-1*evrange)
                        toploty.append(minmax_scale(gauss(evrange, mue, sige)))
                        toplotcolor.append('r')
                        toplotmues.append(mue)
                    else:
                        toplotx.append(-1*evrange)
                        toploty.append(minmax_scale(gauss(evrange, mue, sige)))
                        toplotcolor.append('b')
                        toplotmues.append(mue) 
                        
    exampleplotneuron = delayexamples[region]
    params = fitparams[(fitparams.Neuron==exampleplotneuron[0]) & (fitparams.Session==exampleplotneuron[1])].iloc[0]
    Me = params['MaxEMean']
    me = params['MinEMean']
    mue = params['Mue']
    evrange = np.arange(me,Me+1)
    examplex = -1*evrange
    if params['rgauss']>params['rlog']:
        exampley = minmax_scale(gauss(evrange, params['Mue'], params['Sige']))
    else:
        exampley = minmax_scale(logistic(evrange, params['k'], params['x0']))
    if mue<0:
        examplecolor = 'r'
    else:
        examplecolor = 'b'
    
    
    ranks = np.array(toplotmues)
    starts = [np.nanargmax(y) for y in toploty]
    ranks[np.array(toplotcolor)=='b'] = rankdata(np.array(toplotmues)[np.array(toplotcolor)=='b'])/sum(np.array(toplotcolor)=='b')
    ranks[np.array(toplotcolor)=='r'] = rankdata(-1*np.array(toplotmues)[np.array(toplotcolor)=='r'])/sum(np.array(toplotcolor)=='r')    
    
    plt.figure()
    for x,y,c,r in zip(toplotx, toploty, toplotcolor, ranks):
        if c == 'r' or c=='b':
            plt.plot(x,y, color = c, alpha = r)
        else:
            plt.plot(x,y, color = c, linewidth = 3)    
    plt.plot(examplex, exampley, color = 'k', linewidth = 5)
    plt.title(region+ ' Delay')
    plt.xlim([-10, 10])
    plt.xlabel('Evidence')
    plt.ylabel('Activity')
    plt.savefig('Figure4Plots/TuningCurves/'+region+'delay-fit.pdf')

#reset for late cue
    totalplots = 0
    
    numcurves = {'ACC':15, 'DMS':15, 'HPC':15, 'RSC':15, 'V1':15}
        
    toplotx = []
    toploty = []
    toplotmues = []
    toplotcolor = []
    widths = []
    
    bests = []
    
    plt.figure()
    for i in bestfits:
        if (totalplots < numcurves[region]) and (mups[i]>100) and (mups[i]<200):
            fits = np.array([fitlogcomp[i], fitgausscomp[i]])
            best = np.argmax(fits)
            bests.append(best)
            
            Me = Mes[i]
            me = mes[i]
            mue = mues[i]
            
            
            if best==0:
                k = ks[i]
                x0 = x0s[i]
                
                if Me-me>0:
                    evrange = np.arange(me,Me+1)
                    if individual_plots:
                        plt.figure()
                        plt.plot(-1*evrange, minmax_scale(logistic(evrange, k, x0)))
                    totalplots = totalplots+1
                    
                    if mue<0:
                        toplotx.append(-1*evrange)
                        toploty.append(minmax_scale(logistic(evrange, k, x0)))
                        toplotcolor.append('r')
                        toplotmues.append(mue)
                    else:
                        toplotx.append(-1*evrange)
                        toploty.append(minmax_scale(logistic(evrange, k, x0)))
                        toplotcolor.append('b')
                        toplotmues.append(mue)
                    
                
            if best ==1:
                sige = siges[i]
                if Me-me>0:
                    evrange = np.arange(me,Me+1)
                    if individual_plots:
                        plt.figure()
                        plt.plot(-1*evrange, minmax_scale(gauss(evrange, mue, sige)))
                    totalplots = totalplots+1
                    
                    if mue<0:
                        toplotx.append(-1*evrange)
                        toploty.append(minmax_scale(gauss(evrange, mue, sige)))
                        toplotcolor.append('r')
                        toplotmues.append(mue)
                    else:
                        toplotx.append(-1*evrange)
                        toploty.append(minmax_scale(gauss(evrange, mue, sige)))
                        toplotcolor.append('b')
                        toplotmues.append(mue)                
           

    exampleplotneuron = lateexamples[region]
    params = fitparams[(fitparams.Neuron==exampleplotneuron[0]) & (fitparams.Session==exampleplotneuron[1])].iloc[0]
    Me = params['MaxEMean']
    me = params['MinEMean']
    mue = params['Mue']
    print(mue)
    evrange = np.arange(me,Me+1)
    examplex = -1*evrange
    if params['rgauss']>params['rlog']:
        exampley = minmax_scale(gauss(evrange, params['Mue'], params['Sige']))
    else:
        exampley = minmax_scale(logistic(evrange, params['k'], params['x0']))
    if mue<0:
        examplecolor = 'r'
    else:
        examplecolor = 'b'
            
    
    
    ranks = np.array(toplotmues)
    starts = [np.nanargmax(y) for y in toploty]
    ranks[np.array(toplotcolor)=='b'] = rankdata(np.array(toplotmues)[np.array(toplotcolor)=='b'])/sum(np.array(toplotcolor)=='b')
    ranks[np.array(toplotcolor)=='r'] = rankdata(-1*np.array(toplotmues)[np.array(toplotcolor)=='r'])/sum(np.array(toplotcolor)=='r')    
    
    plt.figure()
    for x,y,c,r in zip(toplotx, toploty, toplotcolor, ranks):
        if c == 'r' or c=='b':
            plt.plot(x,y, color = c, alpha = r)
        else:
            plt.plot(x,y, color = c, linewidth = 3)    
    plt.plot(examplex, exampley, color = 'k', linewidth = 5)        
    plt.title(region+ ' Late Cue')
    plt.xlim([-10, 10])
    plt.xlabel('Evidence')
    plt.ylabel('Activity')
    plt.savefig('Figure4Plots/TuningCurves/'+region+'late-fit.pdf')

#reset for early cue
    totalplots = 0
    
    numcurves = {'ACC':15, 'DMS':15, 'HPC':15, 'RSC':15, 'V1':15}
        
    toplotx = []
    toploty = []
    toplotmues = []
    toplotcolor = []
    widths = []
    
    bests = []
    
    plt.figure()
    for i in bestfits:
        if (totalplots < numcurves[region]) and (mups[i]>0) and (mups[i]<100):
            fits = np.array([fitlogcomp[i], fitgausscomp[i]])
            best = np.argmax(fits)
            bests.append(best)
            
            Me = Mes[i]
            me = mes[i]
            mue = mues[i]
            
            
            if best==0:
                k = ks[i]
                x0 = x0s[i]
                
                if Me-me>0:
                    evrange = np.arange(me,Me+1)
                    if individual_plots:
                        plt.figure()
                        plt.plot(-1*evrange, minmax_scale(logistic(evrange, k, x0)))
                    totalplots = totalplots+1
                    
                    if mue<0:
                        toplotx.append(-1*evrange)
                        toploty.append(minmax_scale(logistic(evrange, k, x0)))
                        toplotcolor.append('r')
                        toplotmues.append(mue)
                    else:
                        toplotx.append(-1*evrange)
                        toploty.append(minmax_scale(logistic(evrange, k, x0)))
                        toplotcolor.append('b')
                        toplotmues.append(mue)
                    
                
            if best ==1:
                sige = siges[i]
                if Me-me>0:
                    evrange = np.arange(me,Me+1)
                    if individual_plots:
                        plt.figure()
                        plt.plot(-1*evrange, minmax_scale(gauss(evrange, mue, sige)))
                    totalplots = totalplots+1
                    
                    if mue<0:
                        toplotx.append(-1*evrange)
                        toploty.append(minmax_scale(gauss(evrange, mue, sige)))
                        toplotcolor.append('r')
                        toplotmues.append(mue)
                    else:
                        toplotx.append(-1*evrange)
                        toploty.append(minmax_scale(gauss(evrange, mue, sige)))
                        toplotcolor.append('b')
                        toplotmues.append(mue)    
                        

    exampleplotneuron = earlyexamples[region]
    params = fitparams[(fitparams.Neuron==exampleplotneuron[0]) & (fitparams.Session==exampleplotneuron[1])].iloc[0]
    Me = params['MaxEMean']
    me = params['MinEMean']
    mue = params['Mue']
    evrange = np.arange(me,Me+1)
    examplex = -1*evrange
    if params['rgauss']>params['rlog']:
        exampley = minmax_scale(gauss(evrange, params['Mue'], params['Sige']))
    else:
        exampley = minmax_scale(logistic(evrange, params['k'], params['x0']))
    if mue<0:
        examplecolor = 'r'
    else:
        examplecolor = 'b'                    

    
    ranks = np.array(toplotmues)
    starts = [np.nanargmax(y) for y in toploty]
    ranks[np.array(toplotcolor)=='b'] = rankdata(np.array(toplotmues)[np.array(toplotcolor)=='b'])/sum(np.array(toplotcolor)=='b')
    ranks[np.array(toplotcolor)=='r'] = rankdata(-1*np.array(toplotmues)[np.array(toplotcolor)=='r'])/sum(np.array(toplotcolor)=='r')    
    
    plt.figure()
    for x,y,c,r in zip(toplotx, toploty, toplotcolor, ranks):
        if c == 'r' or c=='b':
            plt.plot(x,y, color = c, alpha = r)
        else:
            plt.plot(x,y, color = c, linewidth = 3)    
    plt.plot(examplex, exampley, color = 'k', linewidth = 5)        
    plt.title(region+ ' Early Cue')
    plt.xlim([-6, 6])
    plt.xlabel('Evidence')
    plt.ylabel('Activity')
    plt.savefig('Figure4Plots/TuningCurves/'+region+'early-fit-updatedrange.pdf')
