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
from scipy.signal import savgol_filter
from scipy.interpolate import splrep, splev
import pickle

#demo
demo = True #True to run with example datasets, False to run for full data in structured folders

sigthresh = .05

earlyexamples = {'ACC': (10, 'dFF_tetO_8_08022021_T10processedOutput'), 'DMS': (4, 'dFF_scott_d2_857_20190426processedOutput'), 'HPC': (50, 'nicFR_E39_20171103'), 'RSC': (8,'nicFR_k46_20160719_RSM_175um_83mW_zoom2p2processedFR'), 'V1':(8,'nicFR_k56_20161004')}

lateexamples = {'ACC': (153, 'dFF_tetO_8_07282021_T10processedOutput'), 'DMS':(7, 'dFF_scott_a2a_64_11072019processedOutput'), 'HPC':(25, 'nicFR_E43_20170802'), 'RSC': (44,'nicFR_k31_20160114_RSM2_350um_98mW_zoom2p2processedFR'), 'V1':(14,'nicFR_k53_20161205')}

delayexamples = {'ACC': (17, 'dFF_tetO_8_08052021_T11processedOutput'), 'DMS':(1,'dFF_scott_d1_67_20190418processedOutput'), 'HPC':(118, 'nicFR_E22_20170227'), 'RSC': (10, 'nicFR_k42_20160512_RSM_150um_65mW_zoom2p2processedFR'), 'V1':(35,'nicFR_k53_20161205')}                     


specialcolors = {0:'magenta', 1: 'cyan', 2: 'orange'}


def gauss(x, mu, sig):
    return np.exp((-(x-mu)**2)/(2*sig**2))

def logistic(x, k, x0):
    return 1/(1+np.exp(-k*(x-x0)))




def get_crosssection(data, mup, poses, sigp):
    '''
    === INPUTS ===
    data: array of firing rates (1st column), positions (2nd column), and cumulative evidences (3rd column)
    mup: fit mean position of the data
    poses: array of positions
    sigp: fit position standard deviation of the data

    === OUTPUTS ===
    evrange: observed evidence values at maximum position
    crosssection: average observed firing rate at each observed evidence level
    '''

    #parse data
    frs = data[:, 0]
    data = data[~np.isnan(frs), :]
    frs = data[:, 0]
    pos = data[:, 1].reshape(-1, 1)
    ev = data[:, 2].reshape(-1, 1)
    frs = minmax_scale(frs)
    
    validps = np.where((np.abs(pos-mup)<.5*sigp) & (pos>0))[0] #find valid positions to calculate crosssection based on active position    
    frs = frs[validps]
    evs = ev[validps]

    uniqpos = set(pos[validps].flatten())

    evrange = np.sort(list(set(evs.flatten())))
    evrange = evrange[np.abs(evrange)<11]
    crosssection = np.zeros(len(evrange))
    for i, e in enumerate(evrange):
        if sum(evs.flatten()==e) > len(uniqpos):    
            crosssection[i] = np.mean(frs[evs.flatten()==e]) #calculate mean firing at each evidence level
        else:
            crosssection[i] = np.nan
    evrange = evrange[~np.isnan(crosssection)]
    crosssection = crosssection[~np.isnan(crosssection)]
    
    return evrange, crosssection  


   
if not demo:
    regions = ['ACC', 'DMS', 'HPC', 'RSC']
else:
    regions = ['ACC']


individual_plots = False

for region in regions:
    with open(region+"-nonoutliercells.p", "rb") as fp:   #Pickling
        nstoplot = pickle.load(fp)
    
    
    evs = np.arange(-15, 16)
    
    if region in ['ACC', 'DMS']:
        poses = np.arange(-27.5, 302.5, 5)
    else:
        poses = np.arange(-30, 300, 5)
    
    #load parameters from joint gaussian fit
    if not demo:
        fitparamsgauss = pd.read_csv(region+'/paramfit/'+region+'allfitparams.csv')
    else:
        fitparamsgauss = pd.read_csv('ExampleData/ACCparamfitexample.csv')

    #load parameters from fit at the maximum position to individual datapoints
    if not demo:
        fitparams = pd.read_csv(region+'/paramfit/'+region+'allfitparams-evonly-Cueonly.csv')
    else:
        fitparams = pd.read_csv('ExampleData/ACCparamfitexample-evonly.csv')

    #determine which cells had significant joint gaussian fit
    keep = np.zeros(len(fitparams))
    for i in range(len(fitparams)):
        n = (fitparams['Neuron'].values[i], fitparams['Session'].values[i])
        try:
            MSEsig = fitparamsgauss[(fitparamsgauss.Neuron==fitparams['Neuron'].values[i]) & (fitparamsgauss.Session==fitparams['Session'].values[i])]['PvalMSE'].iloc[0]
        except:
            MSEsig = 1
        if (n in nstoplot) and (MSEsig<.05):
            keep[i] = 1
            
    fitparams['Keep'] = keep
    
    fitparams = fitparams[fitparams['Keep']>0]
        
    keep = np.zeros(len(fitparamsgauss))
    for i in range(len(fitparamsgauss)):
        n = (fitparamsgauss['Neuron'].values[i], fitparamsgauss['Session'].values[i])
        if n in nstoplot:
            keep[i] = 1
            
    fitparamsgauss['Keep'] = keep
    
    fitparamsgauss = fitparamsgauss[fitparamsgauss['Keep']>0]
    
    
    fitlog = fitparams['mselog'].values
    fitgauss = fitparams['msegauss'].values
        
    fitlogr = fitparams['rlog'].values
    fitgaussr = fitparams['rgauss'].values
    
    siglog = fitparams['siglog'].values<.05
    siggauss = fitparams['siggauss'].values<.05
    
    mups = [float(m.replace('[','').replace(']', '')) for m in fitparams['Mup']]
    sigps = fitparams['Sigp'].values

    mues = fitparams['Mue'].values
    siges = fitparams['Sige'].values
    
    ns = fitparams['Neuron'].values
    ss = fitparams['Session'].values

    
    x0s = fitparams['x0'].values
    ks = fitparams['k'].values

    Mes = fitparams['MaxEMean'].values
    mes = fitparams['MinEMean'].values
           
    if region == 'HPC':
        sigparams = pd.read_csv(region+'/paramfit/'+region+'allfitparams-evonly-Cueonly-mse.csv')
        for i, (n, s) in enumerate(zip(ns, ss)):
            try:
                ps = sigparams[(sigparams.Neuron==n)&(sigparams.Session==s)].iloc[0]
                siglog[i] = ps['pmselog']<.05
                siggauss[i] = ps['pmsegauss']<.05
            except:
                siglog[i] = siglog[i]
                siggauss[i] = siggauss[i]
    
    
    #replace nonsignifcant fits   
    fitlogcomp = fitlog.copy()
    fitlogcomp[siglog<1] = 100
        
    fitgausscomp = fitgauss.copy()
    fitgausscomp[siggauss<1] = 100  
    
    fitlogcompr = fitlogr.copy()
    fitlogcompr[siglog<1] = 0
        
    fitgausscompr = fitgaussr.copy()
    fitgausscompr[siggauss<1] = 0  


    
    #select same set of cells as in Fig. 5 for plotting
    bestfits = np.argsort(np.nanmax(np.vstack((fitlogcompr, fitgausscompr)), axis=0))[::-1]

    if demo:
        exampleplotneuron = lateexamples[region]
        params = fitparams[(fitparams.Neuron==exampleplotneuron[0]) & (fitparams.Session==exampleplotneuron[1])].iloc[0]
        n = exampleplotneuron[0]
        s = exampleplotneuron[1]    
        ndata = np.load('ExampleData/exampleneuron.npy') #example neuron is the late delay example
    
        evrange, crosssection = get_crosssection(ndata, float(params['Mup']), poses, params['Sigp'])
        mue = params['Mue']
        examplex = -1*evrange
        exampley = minmax_scale(crosssection)
        if mue<0:
            examplecolor = 'r'
        else:
            examplecolor = 'b' 

        plt.figure() 
        plt.plot(examplex, exampley, color = examplecolor, linewidth = 5)    
        plt.title(region+' Late Example Raw Tuning')
        plt.xlim([-10, 10])
        plt.xlabel('Evidence')
        plt.ylabel('Activity')
        plt.show()           

    else:    
        #delay
        print('delay')
        totalplots = 0
        
        numcurves = {'ACC':15, 'DMS':15, 'HPC':15, 'RSC':15, 'V1':15}
            
        toplotx = []
        toploty = []
        toplotmues = []
        toplotcolor = []
        
        bests = []
        
        plt.figure()
        for i in bestfits:
            if (totalplots < numcurves[region]) and (mups[i]>200):
                n = fitparams['Neuron'].iloc[i]
                s = fitparams['Session'].iloc[i]
                
                mup = mups[i]
                sigp = sigps[i]
                mue = mues[i]
                
                try:
                    ndata = np.load(region+'/firingdata/'+s+'neuron'+str(n)+'.npy')
                except:
                    ndata = np.load(region+'/firingdata-split/'+s+'neuron'+str(n)+'.npy')
                    
                evrange, crosssection = get_crosssection(ndata, mup, poses, sigp)
                
                if len(evrange)>2:
                    if mue<0:
                        toplotx.append(-1*evrange)
                        toploty.append(minmax_scale(crosssection))
                        toplotcolor.append('r')
                        toplotmues.append(mue)
                    else:
                        toplotx.append(-1*evrange)
                        toploty.append(minmax_scale(crosssection))
                        toplotcolor.append('b')
                        toplotmues.append(mue)  
                    
                    totalplots = totalplots+1

                        

        exampleplotneuron = delayexamples[region]
        n = exampleplotneuron[0]
        s = exampleplotneuron[1]
        params = fitparams[(fitparams.Neuron==exampleplotneuron[0]) & (fitparams.Session==exampleplotneuron[1])].iloc[0]
        try:
            ndata = np.load(region+'/firingdata/'+s+'neuron'+str(n)+'.npy')
        except:
            ndata = np.load(region+'/firingdata-split/'+s+'neuron'+str(n)+'.npy')
    
        evrange, crosssection = get_crosssection(ndata, float(params['Mup']), poses, params['Sigp'])
        mue = params['Mue']
        examplex = -1*evrange
        exampley = minmax_scale(crosssection)
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
                plt.plot(x,y, color = c, alpha = r, linewidth = 1)
            else:
                plt.plot(x,y, color = c, linewidth = 3)    
        plt.plot(examplex, exampley, color = 'k', linewidth = 5)    
        plt.title(region+' Delay')
        plt.xlim([-10, 10])
        plt.xlabel('Evidence')
        plt.ylabel('Activity')
        plt.show()   
        
        
        #late cue
        print('late')
        totalplots = 0
        
        numcurves = {'ACC':15, 'DMS':15, 'HPC':15, 'RSC':15, 'V1':15}
            
        toplotx = []
        toploty = []
        toplotmues = []
        toplotcolor = []
        
        bests = []
        
        plt.figure()
        for i in bestfits:
            if (totalplots < numcurves[region]) and (mups[i]>100) and (mups[i]<200):
                n = fitparams['Neuron'].iloc[i]
                s = fitparams['Session'].iloc[i]
                
                mup = mups[i]
                sigp = sigps[i]
                mue = mues[i]
                
                try:
                    ndata = np.load(region+'/firingdata/'+s+'neuron'+str(n)+'.npy')
                except:
                    ndata = np.load(region+'/firingdata-split/'+s+'neuron'+str(n)+'.npy')
                    
                evrange, crosssection = get_crosssection(ndata, mup, poses, sigp)
                
                if len(evrange)>2:
                    if mue<0:
                        toplotx.append(-1*evrange)
                        toploty.append(minmax_scale(crosssection))
                        toplotcolor.append('r')
                        toplotmues.append(mue)
                    else:
                        toplotx.append(-1*evrange)
                        toploty.append(minmax_scale(crosssection))
                        toplotcolor.append('b')
                        toplotmues.append(mue)  
                    
                    totalplots = totalplots+1

                        

        
        exampleplotneuron = lateexamples[region]
        params = fitparams[(fitparams.Neuron==exampleplotneuron[0]) & (fitparams.Session==exampleplotneuron[1])].iloc[0]
        n = exampleplotneuron[0]
        s = exampleplotneuron[1]    
        try:
            ndata = np.load(region+'/firingdata/'+s+'neuron'+str(n)+'.npy')
        except:
            ndata = np.load(region+'/firingdata-split/'+s+'neuron'+str(n)+'.npy')
    
        evrange, crosssection = get_crosssection(ndata, float(params['Mup']), poses, params['Sigp'])
        mue = params['Mue']
        examplex = -1*evrange
        exampley = minmax_scale(crosssection)
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
                plt.plot(x,y, color = c, alpha = r, linewidth = 1)
            else:
                plt.plot(x,y, color = c, linewidth = 3)    
        plt.plot(examplex, exampley, color = 'k', linewidth = 5)    
        plt.title(region+' Late Cue')
        plt.xlim([-10, 10])
        plt.xlabel('Evidence')
        plt.ylabel('Activity')
        plt.show()        
        
        
        
        #early cue
        print('early')
        totalplots = 0
        
        numcurves = {'ACC':15, 'DMS':15, 'HPC':15, 'RSC':15, 'V1':15}
            
        toplotx = []
        toploty = []
        toplotmues = []
        toplotcolor = []
        
        bests = []
        
        plt.figure()
        for i in bestfits:
            if (totalplots < numcurves[region]) and (mups[i]>0) and (mups[i]<100):
                n = fitparams['Neuron'].iloc[i]
                s = fitparams['Session'].iloc[i]
                
                mup = mups[i]
                sigp = sigps[i]
                mue = mues[i]
                
                try:
                    ndata = np.load(region+'/firingdata/'+s+'neuron'+str(n)+'.npy')
                except:
                    ndata = np.load(region+'/firingdata-split/'+s+'neuron'+str(n)+'.npy')
                    
                evrange, crosssection = get_crosssection(ndata, mup, poses, sigp)
                
                if len(evrange)>2:
                    if mue<0:
                        toplotx.append(-1*evrange)
                        toploty.append(minmax_scale(crosssection))
                        toplotcolor.append('r')
                        toplotmues.append(mue)
                    else:
                        toplotx.append(-1*evrange)
                        toploty.append(minmax_scale(crosssection))
                        toplotcolor.append('b')
                        toplotmues.append(mue)  
                    
                    totalplots = totalplots+1

                        
        
        exampleplotneuron = earlyexamples[region]
        n = exampleplotneuron[0]
        s = exampleplotneuron[1]
        params = fitparams[(fitparams.Neuron==exampleplotneuron[0]) & (fitparams.Session==exampleplotneuron[1])].iloc[0]
        try:
            ndata = np.load(region+'/firingdata/'+s+'neuron'+str(n)+'.npy')
        except:
            ndata = np.load(region+'/firingdata-split/'+s+'neuron'+str(n)+'.npy')
    
        evrange, crosssection = get_crosssection(ndata, float(params['Mup']), poses, params['Sigp'])
        mue = params['Mue']
        examplex = -1*evrange
        exampley = minmax_scale(crosssection)
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
                plt.plot(x,y, color = c, alpha = r, linewidth = 1)
            else:
                plt.plot(x,y, color = c, linewidth = 3)    
        plt.plot(examplex, exampley, color = 'k', linewidth = 5)    
        plt.title(region+' Early Cue')
        plt.xlim([-6, 6])
        plt.xlabel('Evidence')
        plt.ylabel('Activity')
        plt.show()
