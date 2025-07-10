# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 10:19:35 2023

@author: lindseyb
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
from scipy.stats import sem, rankdata
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.gridspec as gs
from sklearn.preprocessing import minmax_scale
import pickle


#demo
demo = True #True to run with example datasets, False to run for full data in structured folders

sigthresh = .05

#examples used in Fig. 5A,E,I and Extended Data Fig. 5A to be plotted in bold for reference
earlyexamples = {'ACC': (10, 'dFF_tetO_8_08022021_T10processedOutput'), 'DMS': (4, 'dFF_scott_d2_857_20190426processedOutput'), 'HPC': (50, 'nicFR_E39_20171103'), 'RSC': (8,'nicFR_k46_20160719_RSM_175um_83mW_zoom2p2processedFR')}

lateexamples = {'ACC': (153, 'dFF_tetO_8_07282021_T10processedOutput'), 'DMS':(7, 'dFF_scott_a2a_64_11072019processedOutput'), 'HPC':(25, 'nicFR_E43_20170802'), 'RSC': (44,'nicFR_k31_20160114_RSM2_350um_98mW_zoom2p2processedFR')}

delayexamples = {'ACC': (17, 'dFF_tetO_8_08052021_T11processedOutput'), 'DMS':(1,'dFF_scott_d1_67_20190418processedOutput'), 'HPC':(118, 'nicFR_E22_20170227'), 'RSC': (10, 'nicFR_k42_20160512_RSM_150um_65mW_zoom2p2processedFR')}                 

#gaussian function
def gauss(x, mu, sig):
    return np.exp((-(x-mu)**2)/(2*sig**2))

#gaussian function with an intercept
def gaussintercept(x, mu, sig, a, b):
    return a*np.exp((-(x-mu)**2)/(2*sig**2))+b

#logistic function
def logistic(x, k, x0):
    return 1/(1+np.exp(-k*(x-x0)))

#logistic function with intercept
def logintercept(x, k, x0, a, b):
    return a/(1+np.exp(-k*(x-x0)))+b 


def fitcrosssection(evrange, crosssection):
    '''
    ===INPUTS===
    evrange: observed evidence values at the position of the cross section
    crosssection: average firing rate at each evidence value

    ===OUTPUTS===
    predicted firing rates from the better fitting of a gaussian or logistic function to the crosssection

    '''
    crosssection = minmax_scale(crosssection)
    #fit a gaussian
    try:
        gaussps, gausscov = curve_fit(gaussintercept, evrange, crosssection, p0 = [evrange[np.argmax(crosssection)], 1, .5, .01], bounds = ([-15, 0, 0, 0], [15, 10, 10, 1]))
        mu, sig, a, b = gaussps
        gausspred = gaussintercept(evrange, mu, sig, a, b)
        gausserror = np.mean((crosssection-gausspred)**2)
    except:
        gausserror = np.inf
    
    #fit a logistic function
    try:
        logps, logcov = curve_fit(logintercept, evrange, crosssection, p0 = [0, 0, .5, .01], bounds = ([-15, -1, 0, 0], [15, 1, 10, 1]))
        x0, k, a, b = logps
        logpred = logintercept(evrange, x0, k, a, b)
        logerror = np.mean((crosssection-logpred)**2)
    except:
        logerror = np.inf
    
    #compare mse of predictions from gaussian and predictions from logistic
    if logerror<gausserror:
        return minmax_scale(logpred)
    else:
        return minmax_scale(gausspred)

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
    evs: all evidence levels at active position
    frs: all firing rates at the active position with corresponding evidence in evs
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
    
    return evrange, crosssection, evs, frs



if not demo:
    regions = ['ACC', 'DMS', 'HPC', 'RSC']
else:
    regions = ['ACC']

individual_plots = False

for region in regions:
    print(region)
    with open(region+"-nonoutliercells.p", "rb") as fp:   
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
    
    #to select cells, use fit to all data as a proxy for best fits
    fitlog = fitparams['mselog'].values
    fitgauss = fitparams['msegauss'].values
        
    fitlogr = fitparams['rlog'].values
    fitgaussr = fitparams['rgauss'].values
    
    siglog = fitparams['siglog'].values<.05
    siggauss = fitparams['siggauss'].values<.05
    
    try:
        mups = [float(m.replace('[','').replace(']', '')) for m in fitparams['Mup']]
    except:
        mups = fitparams['Mup'].values
        

    mues = fitparams['Mue'].values
    siges = fitparams['Sige'].values
    
    mups = [float(m.replace('[','').replace(']', '')) for m in fitparams['Mup']]
    sigps = fitparams['Sigp'].values
    
    ns = fitparams['Neuron'].values
    ss = fitparams['Session'].values

    
    x0s = fitparams['x0'].values
    ks = fitparams['k'].values
    alogs = fitparams['alog'].values
    
    agauss = fitparams['agauss'].values

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

    
    #choose best fitting cells to plot curves
    bestfits = np.argsort(np.nanmax(np.vstack((fitlogcompr, fitgausscompr)), axis=0))[::-1]
    
    if demo:
        #plot single example
        exampleplotneuron = lateexamples[region]
        
        ndata = np.load('ExampleData/exampleneuron.npy') #this is the late example in ACC

        n = exampleplotneuron[0]
        s = exampleplotneuron[1]

        mup = fitparamsgauss[(fitparamsgauss.Neuron==n) & (fitparamsgauss.Session==s)]['Mup'].iloc[0]
        sigp = fitparamsgauss[(fitparamsgauss.Neuron==n) & (fitparamsgauss.Session==s)]['Sigp'].iloc[0]
        mue = fitparamsgauss[(fitparamsgauss.Neuron==n) & (fitparamsgauss.Session==s)]['Mue'].iloc[0]

        evrange, crosssection, evpts, frpts = get_crosssection(ndata, mup, poses, sigp)
        
        examplex = -1*evrange
        exampley = minmax_scale(fitcrosssection(evrange, crosssection))
        if mue<0:
            examplecolor = 'r'
        else:
            examplecolor = 'b'

        plt.figure()   
        plt.plot(examplex, exampley, color = examplecolor, linewidth = 5)
        plt.title(region+ ' Late Example')
        plt.xlim([-10, 10])
        plt.xlabel('Evidence')
        plt.ylabel('Activity')
        plt.show()
        

        

    else:
        #plot example cells from delay region    
        totalplots = 0
        
        numcurves = {'ACC':15, 'DMS':15, 'HPC':15, 'RSC':15, 'V1':15}
            
        toplotx = []
        toploty = []
        toplotmues = []
        toplotcolor = []
        widths = []
        
        bests = []
        
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
                    
                evrange, crosssection, evpts, frpts = get_crosssection(ndata, mup, poses, sigp)
                
                
                fits = np.array([fitlogcomp[i], fitgausscomp[i]])

                print(np.array([alogs[i], agauss[i]]))
                best = np.argmin(fits)
                bests.append(best)
                
                Me = Mes[i]
                me = mes[i]
                mue = mues[i]
                
                if len(evrange)>2:
                    toplotx.append(-1*evrange)
                    toploty.append(fitcrosssection(evrange, crosssection))
                    totalplots= totalplots+1
                    toplotmues.append(mue)
                    if mue<0:
                        toplotcolor.append('r')
                    else:
                        toplotcolor.append('b')
                

                            
        exampleplotneuron = delayexamples[region]
        
        n = exampleplotneuron[0]
        s = exampleplotneuron[1]
        try:
            ndata = np.load(region+'/firingdata/'+s+'neuron'+str(n)+'.npy')
        except:
            ndata = np.load(region+'/firingdata-split/'+s+'neuron'+str(n)+'.npy')
            
        evrange, crosssection, evpts, frpts = get_crosssection(ndata, mup, poses, sigp)
        

        examplex = -1*evrange
        exampley = minmax_scale(fitcrosssection(evrange, crosssection))

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
                n = fitparams['Neuron'].iloc[i]
                s = fitparams['Session'].iloc[i]
                
                mup = mups[i]
                sigp = sigps[i]
                mue = mues[i]
                
                try:
                    ndata = np.load(region+'/firingdata/'+s+'neuron'+str(n)+'.npy')
                except:
                    ndata = np.load(region+'/firingdata-split/'+s+'neuron'+str(n)+'.npy')
                    
                evrange, crosssection, evpts, frpts = get_crosssection(ndata, mup, poses, sigp)
                
                
                fits = np.array([fitlogcomp[i], fitgausscomp[i]])

                print(np.array([alogs[i], agauss[i]]))
                best = np.argmin(fits)
                bests.append(best)
                
                Me = Mes[i]
                me = mes[i]
                mue = mues[i]
                
                if len(evrange)>2:
                    toplotx.append(-1*evrange)
                    toploty.append(fitcrosssection(evrange, crosssection))
                    totalplots= totalplots+1
                    toplotmues.append(mue)
                    if mue<0:
                        toplotcolor.append('r')
                    else:
                        toplotcolor.append('b')
                
            

        exampleplotneuron = lateexamples[region]
        
        n = exampleplotneuron[0]
        s = exampleplotneuron[1]
        try:
            ndata = np.load(region+'/firingdata/'+s+'neuron'+str(n)+'.npy')
        except:
            ndata = np.load(region+'/firingdata-split/'+s+'neuron'+str(n)+'.npy')
            
        evrange, crosssection, evpts, frpts = get_crosssection(ndata, mup, poses, sigp)
        
        examplex = -1*evrange
        exampley = minmax_scale(fitcrosssection(evrange, crosssection))

                
        
        
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
        plt.show()

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
                n = fitparams['Neuron'].iloc[i]
                s = fitparams['Session'].iloc[i]
                
                mup = mups[i]
                sigp = sigps[i]
                mue = mues[i]
                
                try:
                    ndata = np.load(region+'/firingdata/'+s+'neuron'+str(n)+'.npy')
                except:
                    ndata = np.load(region+'/firingdata-split/'+s+'neuron'+str(n)+'.npy')
                    
                evrange, crosssection, evpts, frpts = get_crosssection(ndata, mup, poses, sigp)
                
                
                fits = np.array([fitlogcomp[i], fitgausscomp[i]])

                print(np.array([alogs[i], agauss[i]]))
                best = np.argmin(fits)
                bests.append(best)
                
                Me = Mes[i]
                me = mes[i]
                mue = mues[i]
                
                if len(evrange)>2:
                    toplotx.append(-1*evrange)
                    toploty.append(fitcrosssection(evrange, crosssection))
                    totalplots= totalplots+1
                    toplotmues.append(mue)
                    if mue<0:
                        toplotcolor.append('r')
                    else:
                        toplotcolor.append('b')
                fits = np.array([fitlogcomp[i], fitgausscomp[i]])
                best = np.argmin(fits)
                bests.append(best)
                
                Me = Mes[i]
                me = mes[i]
                mue = mues[i]
                
                            

        exampleplotneuron = earlyexamples[region]
        
        n = exampleplotneuron[0]
        s = exampleplotneuron[1]
        try:
            ndata = np.load(region+'/firingdata/'+s+'neuron'+str(n)+'.npy')
        except:
            ndata = np.load(region+'/firingdata-split/'+s+'neuron'+str(n)+'.npy')
            
        evrange, crosssection, evpts, frpts = get_crosssection(ndata, mup, poses, sigp)
        
        examplex = -1*evrange
        exampley = minmax_scale(fitcrosssection(evrange, crosssection))              

        
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
        plt.show()
