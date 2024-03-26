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
from scipy import stats
from scipy.stats import sem, rankdata
import pandas as pd
from sklearn.preprocessing import minmax_scale
import pickle

#demo
demo = True #True to run with example datasets, False to run for full data in structured folders

#dictionary of examples used in each maze region of Figure 5
earlyexamples = {'ACC': (10, 'dFF_tetO_8_08022021_T10processedOutput'), 'DMS': (4, 'dFF_scott_d2_857_20190426processedOutput'), 'HPC': (50, 'nicFR_E39_20171103'), 'RSC': (3,'nicFR_k50_20160811_RSM_400um_114mW_zoom2p2processedFR'), 'V1':(8,'nicFR_k56_20161004')}
lateexamples = {'ACC': (153, 'dFF_tetO_8_07282021_T10processedOutput'), 'DMS':(7, 'dFF_scott_a2a_64_11072019processedOutput'), 'HPC':(25, 'nicFR_E43_20170802'), 'RSC': (44,'nicFR_k31_20160114_RSM2_350um_98mW_zoom2p2processedFR'), 'V1':(14,'nicFR_k53_20161205')}
delayexamples = {'ACC': (17, 'dFF_tetO_8_08052021_T11processedOutput'), 'DMS':(1,'dFF_scott_d1_67_20190418processedOutput'), 'HPC':(118, 'nicFR_E22_20170227'), 'RSC': (10, 'nicFR_k42_20160512_RSM_150um_65mW_zoom2p2processedFR'), 'V1':(35,'nicFR_k53_20161205')}                     

def get_crosssection(data, mup, poses, sigp):
    '''
    function to return raw averaged firing rate at each evidence level during a cell's active position

    ===inputs===
    data: array of data with firing rates (1st column), positions (2nd column), and cumulative evidence (3rd column)
    mup: fit position mean of the neuron
    poses: array of position bins
    sigp: fit position standard deviation of the neurons

    ===outputs===
    evrange: range of observed evidence levels at the active position
    crosssection: average firing rate at each evidence level in evrange
    '''
    #read in neural data
    frs = data[:, 0] #firing rates, 1st column
    data = data[~np.isnan(frs), :] #remove nan firing rates
    frs = data[:, 0]
    pos = data[:, 1].reshape(-1, 1) #positions, 2nd column
    ev = data[:, 2].reshape(-1, 1) #cumulative evidence, 3rd column

    validps = np.where((np.abs(pos-mup)<.5*sigp) & (pos>0))[0] #identify active positions, defined as positions within .5 fit position standard deviations of the fit position means, only consider firing during the cue period

    #restrict to the active position
    frs = frs[validps]
    evs = ev[validps]
    
    #set of valid positions
    uniqpos = set(pos[validps].flatten())

    evrange = np.sort(list(set(evs.flatten()))) #get unique evidence levels
    evrange = evrange[np.abs(evrange)<11] #only consider evidence in range [-10, 10]
    
    #find the average firing rate at each evidence level in evrange
    crosssection = np.zeros(len(evrange)) 
    for i, e in enumerate(evrange):
        if sum(evs.flatten()==e) > len(uniqpos):  #require that the evidence level is sampled on more than one trial  
            crosssection[i] = np.mean(frs[evs.flatten()==e]) #take average at each evdiecne level
        else:
            crosssection[i] = np.nan

    #remove any undersampled evidence values
    evrange = evrange[~np.isnan(crosssection)]
    crosssection = crosssection[~np.isnan(crosssection)]

    return evrange, crosssection   
 
if not demo:
    regions = ['ACC', 'DMS', 'HPC', 'RSC']
else:
    regions = ['ACC']

for region in regions:
    #open a list of cells that have been verified to meet non-outlier criteria (e.g. for cells with evidence SD<3, there is not exactly one outlier in the bin containing the mean evidence and mean position)
    with open(region+"-nonoutliercells.p", "rb") as fp:   
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
    fitgauss = fitparams['rgauss'].values #correlations of local gaussian fit
    
    siglog = fitparams['siglog'].values<.05 #whether local logistic fit was significant compared to pseudosession
    siggauss = fitparams['siggauss'].values<.05 #whether local gaussian fit was significant compared to pseudosession
    
    mups = [float(m.replace('[','').replace(']', '')) for m in fitparams['Mup']] #update so mean position is a float and not a string
    sigps = fitparams['Sigp'].values
        
    #get parameters for joint fit
    mues = fitparams['Mue'].values
    siges = fitparams['Sige'].values
    
    ns = fitparams['Neuron'].values
    ss = fitparams['Session'].values

    Mes = fitparams['MaxEMean'].values
    mes = fitparams['MinEMean'].values
            
    #replace nonsignifcant fits   
    fitlogcomp = fitlog.copy()
    fitlogcomp[siglog<1] = 0
        
    fitgausscomp = fitgauss.copy()
    fitgausscomp[siggauss<1] = 0
    
    bestfits = np.argsort(np.nanmax(np.vstack((fitlogcomp, fitgausscomp)), axis=0))[::-1] #sort in descending order of best fit of gauss or logistic, to get same cells as Fig. 5
    
    numcurves = {'ACC':15, 'DMS':15, 'HPC':15, 'RSC':15}  #number of curves to plot for each brain region

    if not demo:
        #delay region of the maze
        totalplots = 0
        toplotx = []
        toploty = []
        toplotmues = []
        toplotcolor = []
            
        plt.figure()
        for i in bestfits: #iterate over ordered list of fits, same as in Figure 5
            if (totalplots < numcurves[region]) and (mups[i]>200): #only plot if the cell in delay region for maximum of numcurves
                n = fitparams['Neuron'].iloc[i]
                s = fitparams['Session'].iloc[i]
                
                mup = mups[i]
                sigp = sigps[i]
                mue = mues[i]
                
                ndata = np.load(region+'/firingdata/'+s+'neuron'+str(n)+'.npy') #load corresponding neural data
                evrange, crosssection = get_crosssection(ndata, mup, poses, sigp) #get average evidence tuning
                
                if len(evrange)>2:
                    if mue<0: #test if right or left preferring
                        toplotx.append(-1*evrange)
                        toploty.append(minmax_scale(crosssection)) #put all cross sections on the same scale
                        toplotcolor.append('r') #right preferring, red
                        toplotmues.append(mue)
                    else:
                        toplotx.append(-1*evrange)
                        toploty.append(minmax_scale(crosssection))
                        toplotcolor.append('b') #left preferring, blue
                        toplotmues.append(mue)  
                    
                    totalplots = totalplots+1
            #get curve for the corresponding example in that region 
            exampleplotneuron = delayexamples[region]
            n = exampleplotneuron[0]
            s = exampleplotneuron[1]
            params = fitparams[(fitparams.Neuron==exampleplotneuron[0]) & (fitparams.Session==exampleplotneuron[1])].iloc[0] #parameters for specific neuron
            ndata = np.load(region+'/firingdata/'+s+'neuron'+str(n)+'.npy')
            evrange, crosssection = get_crosssection(ndata, float(params['Mup']), poses, params['Sigp']) #get average evidence tuning curve at active posiiton
            mue = params['Mue']
            examplex = -1*evrange
            exampley = minmax_scale(crosssection)
            if mue<0: #color according to choice preference
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
                    plt.plot(x,y, color = c, alpha = r, linewidth = 1)
                else:
                    plt.plot(x,y, color = c, linewidth = 3)    
            plt.plot(examplex, exampley, color = 'k', linewidth = 5)  #plot example curve thicker  
            plt.title(region+' Delay')
            plt.xlim([-10, 10])
            plt.xlabel('Evidence')
            plt.ylabel('Activity')
            plt.show()
            
            
            #repeat for late cue
            totalplots = 0
            toplotx = []
            toploty = []
            toplotmues = []
            toplotcolor = []
                
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
            ndata = np.load(region+'/firingdata/'+s+'neuron'+str(n)+'.npy')
        
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
            
            
            
            #repeat for early cue
            totalplots = 0        
            toplotx = []
            toploty = []
            toplotmues = []
            toplotcolor = []
            
            plt.figure()
            for i in bestfits:
                if (totalplots < numcurves[region]) and (mups[i]>0) and (mups[i]<100):
                    n = fitparams['Neuron'].iloc[i]
                    s = fitparams['Session'].iloc[i]
                    
                    mup = mups[i]
                    sigp = sigps[i]
                    mue = mues[i]
                    
                    ndata = np.load(region+'/firingdata/'+s+'neuron'+str(n)+'.npy')
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
            ndata = np.load(region+'/firingdata/'+s+'neuron'+str(n)+'.npy')
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
    else:
        exampleplotneuron = lateexamples[region]
        params = fitparams[(fitparams.Neuron==exampleplotneuron[0]) & (fitparams.Session==exampleplotneuron[1])].iloc[0]
        n = exampleplotneuron[0]
        s = exampleplotneuron[1]    
        ndata = np.load('ExampleData/exampleneuron.npy')
    
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
            plt.title(region+' Late Cue')
            plt.xlim([-6, 6])
            plt.xlabel('Evidence')
            plt.ylabel('Activity')
            plt.show()
        
