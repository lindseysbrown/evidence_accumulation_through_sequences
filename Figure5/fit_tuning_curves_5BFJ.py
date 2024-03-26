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
import pandas as pd
from sklearn.preprocessing import minmax_scale
import pickle
from scipy.stats import rankdata

#demo
demo = True #True to run with example datasets, False to run for full data in structured folders

#dictionary of examples used in each maze region of Figure 5
earlyexamples = {'ACC': (10, 'dFF_tetO_8_08022021_T10processedOutput'), 'DMS': (4, 'dFF_scott_d2_857_20190426processedOutput'), 'HPC': (50, 'nicFR_E39_20171103'), 'RSC': (3,'nicFR_k50_20160811_RSM_400um_114mW_zoom2p2processedFR'), 'V1':(8,'nicFR_k56_20161004')}
lateexamples = {'ACC': (153, 'dFF_tetO_8_07282021_T10processedOutput'), 'DMS':(7, 'dFF_scott_a2a_64_11072019processedOutput'), 'HPC':(25, 'nicFR_E43_20170802'), 'RSC': (44,'nicFR_k31_20160114_RSM2_350um_98mW_zoom2p2processedFR'), 'V1':(14,'nicFR_k53_20161205')}
delayexamples = {'ACC': (17, 'dFF_tetO_8_08052021_T11processedOutput'), 'DMS':(1,'dFF_scott_d1_67_20190418processedOutput'), 'HPC':(118, 'nicFR_E22_20170227'), 'RSC': (10, 'nicFR_k42_20160512_RSM_150um_65mW_zoom2p2processedFR'), 'V1':(35,'nicFR_k53_20161205')}                 


def gauss(x, mu, sig):
    '''
    Gaussian function with mean mu and standard deviation sig
    '''
    return np.exp((-(x-mu)**2)/(2*sig**2))

def logistic(x, k, x0):
    '''
    logistic function with growth rate k and midpoint x0
    '''
    return 1/(1+np.exp(-k*(x-x0)))

if not demo:
    regions = ['ACC', 'DMS', 'HPC', 'RSC']

else:
    regions = ['ACC']

for region in regions:
    #open a list of cells that have been verified to meet non-outlier criteria (e.g. for cells with evidence SD<3, there is not exactly one outlier in the bin containing the mean evidence and mean position)
    if not demo:
        with open(region+"-nonoutliercells.p", "rb") as fp:
            nstoplot = pickle.load(fp)
    else:
        with open('ExampleData/'+region+"-nonoutliercells.p", "rb") as fp:
            nstoplot = pickle.load(fp)

    #evidence bins
    evs = np.arange(-15, 16)
    
    #get position bins corresponding to region
    if region in ['ACC', 'DMS']:
        poses = np.arange(-27.5, 302.5, 5)
    else:
        poses = np.arange(-30, 300, 5)
    
    #load results of fitting logistic and gaussian to just the region around the peak
    if not demo:
        fitparams = pd.read_csv(region+'/paramfit/'+region+'allfitparams-evonly.csv')
    else:
        fitparams = pd.read_csv('ExampleData/ACCparamfitexample-evonly.csv')
    
    #only keep cells that meet non-outlier criteria
    keep = np.zeros(len(fitparams))
    for i in range(len(fitparams)):
        n = (fitparams['Neuron'].values[i], fitparams['Session'].values[i])
        if n in nstoplot:
            keep[i] = 1    
    fitparams['Keep'] = keep
    fitparams = fitparams[fitparams['Keep']>0]
    
    #load parameters from the joint fitting
    if not demo:
        fitparamsgauss = pd.read_csv(region+'/paramfit/'+region+'allfitparams.csv')
    else:
        fitparamsgauss = pd.read_csv('ExampleData/ACCparamfitexample.csv')

    #only keep cells that meet non-outlier criteria
    keep = np.zeros(len(fitparamsgauss))
    for i in range(len(fitparamsgauss)):
        n = (fitparamsgauss['Neuron'].values[i], fitparamsgauss['Session'].values[i])
        if n in nstoplot:
            keep[i] = 1
    fitparamsgauss['Keep'] = keep
    fitparamsgauss = fitparamsgauss[fitparamsgauss['Keep']>0]
    

    fitlog = fitparams['rlog'].values #correlations of local logistic fit
    fitgauss = fitparams['rgauss'].values #correlations of local gausian fit
    
    siglog = fitparams['siglog'].values<.05 #whether local logistic fit was significant compared to pseudosession
    siggauss = fitparams['siggauss'].values<.05 #whether local gaussian fit was significant compared to pseudosession
    
    mups = [float(m.replace('[','').replace(']', '')) for m in fitparams['Mup']] #update so mean position is a float and not a string
        
    #parameters for gaussian and logistic fits
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
    
    bestfits = np.argsort(np.nanmax(np.vstack((fitlogcomp, fitgausscomp)), axis=0))[::-1] #sort in descending order of best fit of gauss or logistic
    
    totalplots = 0
    
    numcurves = {'ACC':15, 'DMS':15, 'HPC':15, 'RSC':15} #number of curves to plot for each brain region

    #initialize lists of xvalues, yvalues, mean evidence levels, and color indicating left vs right preference    
    toplotx = []
    toploty = []
    toplotmues = []
    toplotcolor = []
    widths = []
    
    bests = []
    
    #delay region of the maze
    plt.figure()
    for i in bestfits: #iterate over ordered list of fits
        if (totalplots < numcurves[region]) and (mups[i]>200): #only plot if the cell in delay region for maximum of numcurves
            
            #determine if gaussian or logistic was better fit
            fits = np.array([fitlogcomp[i], fitgausscomp[i]])
            best = np.argmax(fits)
            bests.append(best)
            
            #parameters of evidence
            Me = Mes[i]
            me = mes[i]
            mue = mues[i]
            
            
            if best==0: #logistic fit better
                #get logistic fit parameters
                k = ks[i]
                x0 = x0s[i]
                
                if Me-me>15: #only consider if reasonable observed evidence range
                    evrange = np.arange(me,Me+1) #get observed range of evidence levels
                    totalplots = totalplots+1 #increment number of lines being plotted
                    
                    if mue<0: #test if right or left preferring
                        toplotx.append(-1*evrange) #correct left vs. right evidence definition
                        toploty.append(minmax_scale(logistic(evrange, k, x0))) #plot fit tuning curve from fit parameters
                        toplotcolor.append('r') #right preferring, red
                        toplotmues.append(mue)
                    else:
                        toplotx.append(-1*evrange)
                        toploty.append(minmax_scale(logistic(evrange, k, x0)))
                        toplotcolor.append('b') #left preferring, blue
                        toplotmues.append(mue)
                    
                
            if best ==1: #gaussian fit better
                #get gaussian fit parameters
                sige = siges[i]
                if Me-me>15: #only consider if reasonable observed evidence range
                    evrange = np.arange(me,Me+1)
                    totalplots = totalplots+1
                    
                    if mue<0:#test if right or left preferring
                        toplotx.append(-1*evrange) #correct left vs. right evidence definition
                        toploty.append(minmax_scale(gauss(evrange, mue, sige))) #plot fit tuning curve from fit parameters
                        toplotcolor.append('r') #right preferring, red
                        toplotmues.append(mue)
                    else:
                        toplotx.append(-1*evrange)
                        toploty.append(minmax_scale(gauss(evrange, mue, sige)))
                        toplotcolor.append('b') #left preferring, blue
                        toplotmues.append(mue) 
    #get curve for the corresponding example in that region                    
    exampleplotneuron = delayexamples[region]
    params = fitparams[(fitparams.Neuron==exampleplotneuron[0]) & (fitparams.Session==exampleplotneuron[1])].iloc[0] #parameters for specific neuron
    Me = params['MaxEMean']
    me = params['MinEMean']
    mue = params['Mue']
    evrange = np.arange(me,Me+1)
    examplex = -1*evrange
    if params['rgauss']>params['rlog']: #get plot for better fit
        exampley = minmax_scale(gauss(evrange, params['Mue'], params['Sige']))
    else:
        exampley = minmax_scale(logistic(evrange, params['k'], params['x0']))
    if mue<0: #color according to side preference
        examplecolor = 'r'
    else:
        examplecolor = 'b'
    
    #get varied opacity of lines based on magnitude of left or right evidence tuning
    ranks = np.array(toplotmues)
    starts = [np.nanargmax(y) for y in toploty]
    ranks[np.array(toplotcolor)=='b'] = rankdata(np.array(toplotmues)[np.array(toplotcolor)=='b'])/sum(np.array(toplotcolor)=='b')
    ranks[np.array(toplotcolor)=='r'] = rankdata(-1*np.array(toplotmues)[np.array(toplotcolor)=='r'])/sum(np.array(toplotcolor)=='r')    
    
    #plot figure with example tuing curves
    plt.figure()
    for x,y,c,r in zip(toplotx, toploty, toplotcolor, ranks): #plot each fit curve with correct color and opacity
        if c == 'r' or c=='b':
            plt.plot(x,y, color = c, alpha = r) 
    plt.plot(examplex, exampley, color = 'k', linewidth = 5) #plot example curve thicker
    plt.title(region+ ' Delay')
    plt.xlim([-10, 10])
    plt.xlabel('Evidence')
    plt.ylabel('Activity')
    plt.show()

    #repeat for late cue
    totalplots = 0
    
    numcurves = {'ACC':15, 'DMS':15, 'HPC':15, 'RSC':15}
        
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
    plt.plot(examplex, exampley, color = 'k', linewidth = 5)        
    plt.title(region+ ' Late Cue')
    plt.xlim([-10, 10])
    plt.xlabel('Evidence')
    plt.ylabel('Activity')
    plt.show()

    #repeat for early cue
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
    plt.plot(examplex, exampley, color = 'k', linewidth = 5)        
    plt.title(region+ ' Early Cue')
    plt.xlim([-6, 6])
    plt.xlabel('Evidence')
    plt.ylabel('Activity')
    plt.show()
