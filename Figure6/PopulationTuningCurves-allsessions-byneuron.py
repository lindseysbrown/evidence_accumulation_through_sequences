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
from scipy.stats import f_oneway, kurtosis, skew, zscore
import os
from scipy.optimize import curve_fit
import diptest
import pandas as pd


sigthresh = .05
pthresh = .05
region_threshold = .25 #.5 Ryan
region_width = 4
base_thresh = 0 #3 Ryan

current_cmap = matplotlib.cm.get_cmap('YlOrRd')
current_cmap.set_bad(color='black', alpha=.3) 

def logistic(x, L, k, x0, b):
    return L/(1+np.exp(-k*(x-x0)))+b

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

def get_single_region(data, M, threshold, width, base_thresh):
    upreg = np.where(data>(threshold*M))[0]
    baseline = np.nanmean(data[~upreg])
    regions = []
    if len(upreg)==0:
        return regions
    
    last = np.argmax(data[upreg])
    i=1
    currentreg = [last]
    while (last+i) in upreg:
        currentreg.append(last+i)
        i = i+1
    i = 1
    while (last-i) in upreg:
        currentreg.append(last-i)
        i = i+1
    if len(currentreg)>width:
        if np.nanmean(data[currentreg])>base_thresh*baseline:
            regions.append(currentreg)
    return regions

def divide_LR(alldata, leftchoices, rightchoices, pthresh, region_threshold, region_width, basethresh, single_region = False):
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
        if single_region:
            upregions = get_single_region(avgdata[i, :], m, region_threshold, region_width, base_thresh)
        else:
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




def get_pos_out(data, position, trial, Lcuepos, Rcuepos, mazeID, corrects, choices):
    n_neurons = np.shape(data)[1]
    trials = list(set([x[0] for x in trial]))
    n_trials = len(trials)
    n_pos = 66
    base = 5
    maintrials = []
    correcttrials = []
    leftchoices = []
    rightchoices = []
    lefts = []
    rights = []
    neuraldata = np.zeros((n_trials, n_neurons, n_pos))
    trialmap = np.zeros((n_trials,))
    for it, t in enumerate(trials):
        trialmap[it] = t
        inds = np.where(trial==t)[0]
        maze = mazeID[inds[0]]
        if maze >9:
            maintrials.append(it)
            #correct trials only includes maintrials
            if corrects[inds[0]]:
                correcttrials.append(it)
        if choices[inds[0]]==0:
            leftchoices.append(it)
        else:
            rightchoices.append(it)
        rights.append(Rcuepos[inds[0]][0][0])
        lefts.append(Lcuepos[inds[0]][0][0])
        trialdata = data[inds]
        pos = position[inds]
        posbinned = base*np.round(pos/base)
        avgbypos = np.zeros((n_pos, n_neurons))
        for ip, p in enumerate(range(-30, 300, 5)):
            pis = np.where(posbinned == p)[0]
            if len(pis)>0:
                avgbypos[ip, :] = np.nanmean(trialdata[pis, :], axis=0)
            else:
                avgbypos[ip, :] = avgbypos[ip-1, :]
        neuraldata[it, :, :] = avgbypos.T
    return np.nan_to_num(neuraldata), trialmap, maintrials, correcttrials, lefts, rights, leftchoices, rightchoices

def get_tuningcurve(data, evidence, tuningdict):
    evs = np.arange(-15, 16)
    obs = np.zeros((len(evs), 66))
    for p in range(66):
        for i, e in enumerate(evs):
            validis = np.where(evidence[:, p]==e)[0]
            if len(validis)>2: #1: #2 or more trials
                if (i, p) in tuningdict:
                    tuningdict[(i, p)] = np.concatenate((tuningdict[(i, p)], np.nanmean(data[validis, :, p], axis=0)))
                else:
                    tuningdict[(i, p)] = np.mean(data[validis, :, p], axis=0)
    return tuningdict
    
def dict_to_tuning(tuningdict):
    evs = np.arange(-15, 16)
    obs = np.zeros((len(evs), 66))
    for p in range(66):
        for i, e in enumerate(evs):
            if (i, p) in tuningdict:
                obs[i, p] = np.mean(tuningdict[(i,p)])
            else:
                obs[i, p] = np.nan
    return obs

def jointdict_to_tuning(tuningdictright, tuningdictleft):
    evs = np.arange(-15, 16)
    obs = np.zeros((len(evs), 66))
    for p in range(66):
        for i, e in enumerate(evs):
            ri = np.where(evs==-1*e)[0][0]
            if (i, p) in tuningdictleft:
                obsL = tuningdictleft[(i,p)]
            else:
                obsL = []
            if (ri, p) in tuningdictright:
                obsR = tuningdictright[(ri,p)]
            else:
                obsR = [] 
            if len(obsL)+len(obsR)>0:
                obs[i, p] = np.mean(np.concatenate((obsL, obsR)))
            else:
                obs[i, p] = np.nan
    return obs

def logistic_slope(tuningdict, p):
    evs = np.arange(-15, 16)
    keys = tuningdict.keys()
    evals = []
    frvals = []
    for k in keys:
        if k[1] == p:
            e = evs[k[0]]
            for v in tuningdict[k]:
                if np.abs(e)<11:
                    evals.append(e)
                    frvals.append(v)
    if max(evals) == min(evals):
        return np.nan
    L, k, x0, b = curve_fit(logistic, evals, frvals, p0 = [1, 1, 0, 0], bounds=([-3, .1, -1, 0], [3, 5, 1, 5]) )[0] 

    return k

def jointlogistic_slope(tuningdictL, tuningdictR, p):
    evs = np.arange(-15, 16)
    keysL = tuningdictL.keys()
    keysR = tuningdictR.keys()
    evals = []
    frvals = []
    for k in keysL:
        if k[1] == p:
            e = evs[k[0]]
            for v in tuningdictL[k]:
                if np.abs(e)<11:
                    evals.append(e)
                    frvals.append(v)
    for k in keysR:
        if k[1] == p:
            e = -1*evs[k[0]]
            for v in tuningdictR[k]:
                if np.abs(e)<11:
                    evals.append(e)
                    frvals.append(v)
                    
    if max(evals) == min(evals):
        return np.nan
    L, k, x0, b = curve_fit(logistic, evals, frvals, p0 = [1, 1, 0, 0], bounds=([-3, .1, -1, 0], [3, 5, 1, 5]) )[0] 

    return k

regions = ['ACC', 'DMS', 'RSC', 'HPC', 'V1']

files = os.listdir('./DMS')
DMSmatfiles = [f for f in files if f.startswith('dFF_scott')]

files = os.listdir('./ACC')
ACCmatfiles = [f for f in files if f.startswith('dFF_tet')]

files = os.listdir('./V1')
V1matfiles = [f for f in files if f.startswith('nic')]

files = os.listdir('./RSC')
RSCmatfiles = [f for f in files if f.startswith('nic')]

files = os.listdir('./HPC')
HPCmatfiles = [f for f in files if f.startswith('nic')]

filelist = [ACCmatfiles, DMSmatfiles,  RSCmatfiles,  HPCmatfiles, V1matfiles]

fitvsort = pd.DataFrame()

allLCellLChoice = np.zeros((1, 66))
allLCellRChoice = np.zeros((1, 66))

allRCellLChoice = np.zeros((1, 66))
allRCellRChoice = np.zeros((1, 66))

allLeftmups = np.array([])
allRightmups = np.array([])

vm = {'ACC':1.8, 'DMS': 1.9, 'HPC': 1.95, 'RSC': 1.55, 'V1':1.3}
vM = {'ACC':2.3, 'DMS': 2.35, 'HPC':2.15, 'RSC':2.0, 'V1': 1.7}

colors = {16:'darkslateblue', 26:'dodgerblue', 36:'aqua', 46:'purple', 56:'fuchsia'}

for region, matfiles in zip(regions[1:3], filelist[1:3]):
    regiontuningdictL = {}
    regiontuningdictR = {}
    finaldiff1to4 = np.zeros((1, 60))
    finaldiff5to9 = np.zeros((1, 60))
    finaldiffgreater9 = np.zeros((1, 60))
    negfinaldiff1to4 = np.zeros((1, 60))
    negfinaldiff5to9 = np.zeros((1, 60))
    negfinaldiffgreater9 = np.zeros((1, 60))
    
    
    sortedcells = pd.DataFrame()
    LCellLChoice = np.zeros((1, 66))
    LCellRChoice = np.zeros((1, 66))
    
    RCellLChoice = np.zeros((1, 66))
    RCellRChoice = np.zeros((1, 66))
    
    SCellLChoice = np.zeros((1, 66))
    SCellRChoice = np.zeros((1, 66))
       
    Leftmups = np.array([])
    Rightmups = np.array([])
    
    #fitparamsCS = pd.read_csv(region+'/paramfit/'+region+'allfitparams.csv')
    #fitparamssplit = pd.read_csv(region+'/paramfit/'+region+'allfitparams-split.csv')
    
    #fitparams = pd.concat([fitparamsCS, fitparamssplit], ignore_index=True)
    fitparams = pd.read_csv(region+'/paramfit/'+region+'allfitparams-boundedmue-NEW.csv')
    
    for file in matfiles:
        data = loadmat(region+'/'+file)
        
        if region in ['DMS', 'ACC']:
            bfile = file.split('dFF_')[1]
            bfile = bfile.split('processedOutput.mat')[0]+'.mat'
            cuedata = loadmat(region+'/Behavior/'+bfile)
            lefts = cuedata['logSumm']['cuePos_L'][0][0][0]
            rights = cuedata['logSumm']['cuePos_R'][0][0][0]
            
            #create 3d array (trials x neurons x timepoints)
            n_neurons = np.shape(data['out']['FR2_pos'][0][0])[1]
            [n_trials, n_pos] = np.shape(data['out']['FR2_pos'][0][0][0][0]) 
            
            alldata = np.zeros((n_trials, n_neurons, n_pos))
            
            maintrials = data['out']['Trial_Main_Maze'][0][0][0]-1 #subtract one for difference in matlab indexing
            correcttrials = data['out']['correct'][0][0][0]-1
            
            #get data normalized by position
            for i in range(n_neurons):
                alldata[:, i, :] = data['out']['FR2_pos'][0][0][0][i]
            
            #alldata[np.isnan(alldata)]=0 #replace nan values with 0
            pos = data['out']['Yposition'][0][0][0]
            
            #get different data subsets for left and right choices
            leftchoices_correct = data['out']['correct_left'][0][0][0]-1
            leftchoices_incorrect = data['out']['incorrect_left'][0][0][0]-1
            rightchoices_correct = data['out']['correct_right'][0][0][0]-1
            rightchoices_incorrect = data['out']['incorrect_right'][0][0][0]-1 
            
            lchoices = np.concatenate((leftchoices_correct, leftchoices_incorrect))
            rchoices = np.concatenate((rightchoices_correct, rightchoices_incorrect)) 
            
            
            avgdata = np.nanmean(alldata[:, :, :], axis=0) #neurons x position
        else:
            Fdata = data['nic_output']['ROIactivities'][0][0]
            ndata = data['nic_output']['firingrate2'][0][0]
            ndata[np.isnan(Fdata)] = np.nan #replace FR data where there were Nans in F trace
            position = data['nic_output']['Position'][0][0]
            time = data['nic_output']['Time'][0][0]
            Lcuepos = data['nic_output']['Lcuepos'][0][0]
            Rcuepos = data['nic_output']['Rcuepos'][0][0]
            mazeID = data['nic_output']['MazeID'][0][0]
            corrects = data['nic_output']['ChoiceCorrect'][0][0]
            choices = data['nic_output']['Choice'][0][0]
            
            trial = data['nic_output']['Trial'][0][0]
               
            alldata, tmap, maintrials, correcttrials, lefts, rights, lchoices, rchoices = get_pos_out(ndata, position, trial, Lcuepos, Rcuepos, mazeID, corrects, choices)
            pos = np.arange(-30, 300, 5)
            
            n_trials, n_neurons, n_pos = np.shape(alldata)
            
            #alldata[np.isnan(alldata)]=0
        
        session = file.split('.')[0]
        fileparams = fitparams[fitparams['Session']==session]
        pvals = fileparams['Pval'].values
        mues = fileparams['Mue'].values
        mups = fileparams['Mup'].values
        siges = fileparams['Sige'].values
        corrs = fileparams['Correlation'].values
        
        vp = pvals<sigthresh
        
        #get only evidence selective parameters
        evselectiveneurons = fileparams['Neuron'].values[vp]
        mues = mues[vp]
        mups = mups[vp]
        alldata = alldata[:, evselectiveneurons, :]
        
        evidence = np.zeros((n_trials, n_pos))
        for i in range(n_trials):
            evidence[i] = get_cum_evidence(lefts[i], rights[i], pos)
        evidence = evidence[correcttrials, :]

        #restrict to correct trials
        choices = np.zeros(n_trials)
        choices[lchoices] = 1
        choices = choices[correcttrials]
        alldata = alldata[correcttrials, :, :]
        lchoices = np.where(choices==1)[0]
        rchoices = np.where(choices==0)[0]
        
        leftis = np.where(mues>0)[0]
        rightis = np.where(mues<0)[0]        
        leftmups = mups[leftis]
        rightmups = mups[rightis]
        
        if len(leftis)>0:        
            leftactivity = alldata[:, leftis, :]
            lefttotal = leftactivity#np.nanmean(leftactivity, axis = 1)
            regiontuningdictL = get_tuningcurve(lefttotal, evidence, regiontuningdictL) 
        else:
            lefttotal = np.zeros((len(correcttrials), n_pos))
            
        if len(rightis)>0:
            rightactivity = alldata[:, rightis, :] 
            righttotal = rightactivity#np.nanmean(rightactivity, axis = 1)
            regiontuningdictR = get_tuningcurve(righttotal, evidence, regiontuningdictR)
        else:
            righttotal = np.zeros((len(correcttrials), n_pos))
            
        
    
    evs = np.arange(-15, 16)
    for p in range(10, 30):
        plt.figure()
        for i, e in enumerate(evs):
            if (i, p) in regiontuningdictL:
                obs = regiontuningdictL[(i, p)]
                plt.scatter(e*np.ones(len(obs)), obs)
        plt.xlabel('evidence')
        plt.xlim([-10, 10])
        plt.title(region+'Position'+str(pos[p])+'Left')
        plt.ylabel('Individual Neuron Average Activity')
        
    evs = np.arange(-15, 16)
    for p in range(10, 30):
        plt.figure()
        for i, e in enumerate(evs):
            if (i, p) in regiontuningdictR:
                obs = regiontuningdictR[(i, p)]
                plt.scatter(e*np.ones(len(obs)), obs)
        plt.xlabel('evidence')
        plt.xlim([-10, 10])
        plt.title(region+'Position'+str(pos[p])+'Right')
        plt.ylabel('Individual Neuron Average Activity')    
    
    
    
    
    
    poptuningleft = dict_to_tuning(regiontuningdictL)
    poptuningright = dict_to_tuning(regiontuningdictR)
    poptuningcombined = jointdict_to_tuning(regiontuningdictL, regiontuningdictR)
    
    kRs = np.zeros(66)
    kLs = np.zeros(66)
    kJs = np.zeros(66)
    for i in range(66):
        try:
            kLs[i] = logistic_slope(regiontuningdictL, i)
        except:
            kLs[i] = np.nan
        try:
            kRs[i] = logistic_slope(regiontuningdictR, i)
        except:
            kRs[i] = np.nan
        try:
            kJs[i] = jointlogistic_slope(regiontuningdictL, regiontuningdictR, i)
        except:
            kJs[i] = np.nan
    
    poses = np.arange(-30, 300, 5)
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(poses, np.abs(kLs))
    ax[0].set_title('Left Cells')
    ax[0].set_xticks([])
    ax[0].set_xlim([-30, 300])
    ax[0].set_ylabel('|k|')
    ax[0].set_ylim([0, 5.2])
    ax[1].plot(poses, np.abs(kRs))
    ax[1].set_xticks([-30, 0, 100, 200, 250, 300])
    ax[1].set_xlim([-30, 300])
    ax[1].set_xticklabels(['-30', '0', 'cues', '200', 'delay', '300'])
    ax[1].set_title('Right Cells')
    ax[1].set_ylabel('|k|')
    ax[1].set_ylim([0, 5.2])
    plt.suptitle(region)
    
    plt.figure()
    plt.plot(poses, np.abs(kJs))
    plt.xticks([-30, 0, 100, 200, 250, 300], labels=['-30', '0', 'cues', '200', 'delay', '300'])
    plt.ylabel('|k|')
    plt.title(region+' Pref. - Non. Pref')
    
    
    fig, ax = plt.subplots(2, 1)
    ax[0].imshow(poptuningleft, cmap = 'YlOrRd')
    ax[0].set_xticks([])
    ax[0].set_yticks([0, 15, 30])
    ax[0].set_yticklabels(['-15', '0', '15'])
    ax[0].set_title('Left Cells')
    ax[1].imshow(poptuningright, cmap = 'YlOrRd')
    ax[1].set_xticks([0, 5.5, 25.5, 45.5, 55.25, 65])
    ax[1].set_xticklabels(['-30', '0', 'cues', '200', 'delay', '300'])
    ax[1].set_yticks([0, 15, 30])
    ax[1].set_yticklabels(['-15', '0', '15'])
    ax[1].set_title('Right Cells')
    plt.suptitle(region)
    
    plt.figure()
    plt.imshow(poptuningcombined, cmap='YlOrRd', vmin=vm[region], vmax = vM[region], aspect='auto', interpolation='none')
    plt.xticks([0, 5.5, 25.5, 45.5, 55.25, 65], labels=['-30', '0', 'cues', '200', 'delay', '300'])
    #plt.yticks([0, 15, 30], ['-15', '0', '15'])
    plt.yticks([5, 15, 25], ['-10', '0', '10'])
    for c, i in enumerate([16, 26, 36, 46, 56]):
        plt.axvline(i, color=colors[i], linewidth=2.5, linestyle = '--')
    plt.title(region+' Preferred - Non Pref Population Tuning')
    plt.colorbar()
    plt.ylim([25,5])
    plt.savefig('Figure4Plots/PopulationTuningCurves/'+region+'heatmap-byneuron.pdf')
    
    evs = np.arange(-15, 16)
    plt.figure()
    for c, i in enumerate([16, 26, 36, 46, 56][::-1]):
        plt.plot(evs, poptuningcombined[:, i][::-1], color=colors[i], linewidth = 3)
    plt.title(region+' Preferred - Non Pref')
    plt.xlim([-10, 10])
    plt.ylim([vm[region], vM[region]])
    plt.savefig('Figure4Plots/PopulationTuningCurves/'+region+'cross-byneuron.pdf')
    
    
    
    
    
    evs = np.arange(-15, 16)
    fig,ax = plt.subplots(2)
    for i in range(0, 66, 8):
        ax[0].plot(evs, poptuningleft[:, i], color='green', alpha=(i+1)/66)
        ax[1].plot(evs, poptuningright[:, i], color='green', alpha=(i+1)/66)
    ax[0].set_title('Left Cells')
    ax[1].set_title('Right Cells')
    plt.suptitle(region)
        

        

        
