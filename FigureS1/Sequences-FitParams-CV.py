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
from scipy.stats import f_oneway, kurtosis, skew
import os
from scipy.optimize import curve_fit
import diptest
import pandas as pd


sigthresh = .05
pthresh = .05
region_threshold = .25 #.5 Ryan
region_width = 4
base_thresh = 0 #3 Ryan

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

regions = ['ACC', 'DMS', 'HPC', 'RSC']#, 'V1']

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

filelist = [ACCmatfiles, DMSmatfiles,  HPCmatfiles,  RSCmatfiles]#, V1matfiles]

fitvsort = pd.DataFrame()

allLCellLChoice = np.zeros((1, 66))
allLCellRChoice = np.zeros((1, 66))

allRCellLChoice = np.zeros((1, 66))
allRCellRChoice = np.zeros((1, 66))

allLeftmups = np.array([])
allRightmups = np.array([])

for region, matfiles in zip(regions, filelist):
    print(region)
    sortedcells = pd.DataFrame()
    trainLCellLChoice = np.zeros((1, 66))
    trainLCellRChoice = np.zeros((1, 66))
    
    trainRCellLChoice = np.zeros((1, 66))
    trainRCellRChoice = np.zeros((1, 66))
    
    testLCellLChoice = np.zeros((1, 66))
    testLCellRChoice = np.zeros((1, 66))
    
    testRCellLChoice = np.zeros((1, 66))
    testRCellRChoice = np.zeros((1, 66))    
    
    SCellLChoice = np.zeros((1, 66))
    SCellRChoice = np.zeros((1, 66))
       
    Leftmups = np.array([])
    Rightmups = np.array([])
    
    fitparamsCS = pd.read_csv(region+'/paramfit/'+region+'allfitparams.csv')
    fitparamssplit = pd.read_csv(region+'/paramfit/'+region+'allfitparams-split.csv')
    
    fitparams = pd.concat([fitparamsCS, fitparamssplit], ignore_index=True)
    for file in matfiles:
        data = loadmat(region+'/'+file)
        
        if region in ['DMS', 'ACC']:
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
        vc = corrs>.2
        vmL0 = mues>.5
        
        #get only evidence selective parameters
        evselectiveneurons = fileparams['Neuron'].values[vp]
        mues = mues[vp]
        mups = mups[vp]
        alldata = alldata[:, evselectiveneurons, :]

        print(np.nanmax(alldata))

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
        
        Leftmups = np.concatenate((Leftmups, leftmups))
        Rightmups = np.concatenate((Rightmups, rightmups))
        
        
        #work only on the odd trials
        trainalldata = alldata[::2, :, :]
        testalldata = alldata[1::2, :, :]
    
        trainlchoices = [int(l/2) for l in lchoices if l%2==0]
        testlchoices = [int(l/2) for l in lchoices if l%2==1]
    
        trainrchoices = [int(r/2) for r in rchoices if r%2==0]
        testrchoices = [int(r/2) for r in rchoices if r%2==1]
        
        trainleftdata = trainalldata[trainlchoices, :, :]
        trainrightdata = trainalldata[trainrchoices, :, :]
        testleftdata = testalldata[testlchoices, :, :]
        testrightdata = testalldata[testrchoices, :, :]
          
        if len(leftis)>0:
            trainleftcellsleftchoice = np.nanmean(trainleftdata[:, leftis, :], axis=0)
            trainleftcellsrightchoice = np.nanmean(trainrightdata[:, leftis, :], axis=0)
            trainLCellLChoice = np.vstack((trainLCellLChoice, trainleftcellsleftchoice))
            trainLCellRChoice = np.vstack((trainLCellRChoice, trainleftcellsrightchoice))  
            testleftcellsleftchoice = np.nanmean(testleftdata[:, leftis, :], axis=0)
            testleftcellsrightchoice = np.nanmean(testrightdata[:, leftis, :], axis=0)
            testLCellLChoice = np.vstack((testLCellLChoice, testleftcellsleftchoice))
            testLCellRChoice = np.vstack((testLCellRChoice, testleftcellsrightchoice))              
        
        if len(rightis)>0:
            trainrightcellsleftchoice = np.nanmean(trainleftdata[:, rightis, :], axis=0)
            trainrightcellsrightchoice = np.nanmean(trainrightdata[:, rightis, :], axis=0)
            trainRCellLChoice = np.vstack((trainRCellLChoice, trainrightcellsleftchoice))
            trainRCellRChoice = np.vstack((trainRCellRChoice, trainrightcellsrightchoice))
            testrightcellsleftchoice = np.nanmean(testleftdata[:, rightis, :], axis=0)
            testrightcellsrightchoice = np.nanmean(testrightdata[:, rightis, :], axis=0)
            testRCellLChoice = np.vstack((testRCellLChoice, testrightcellsleftchoice))
            testRCellRChoice = np.vstack((testRCellRChoice, testrightcellsrightchoice))            

    
    allLeftmups = np.concatenate((allLeftmups, Leftmups))
    allRightmups = np.concatenate((allRightmups, Rightmups))
    
    trainLCellLChoice = trainLCellLChoice[1:, :]
    trainLCellRChoice = trainLCellRChoice[1:, :]
    testLCellLChoice = testLCellLChoice[1:, :]
    testLCellRChoice = testLCellRChoice[1:, :]
    
    #allLCellLChoice = np.vstack((allLCellLChoice, LCellLChoice))
    #allLCellRChoice = np.vstack((allLCellRChoice, LCellRChoice))
    
    #newLis = np.argsort(np.argmax((trainLCellLChoice+trainLCellRChoice)/2, axis=1))
    newLis = np.argsort(Leftmups)
    
    #resort data so always in sequence order
    trainLCellLChoice = trainLCellLChoice[newLis, :]
    trainLCellRChoice = trainLCellRChoice[newLis, :]
    testLCellLChoice = testLCellLChoice[newLis, :]
    testLCellRChoice = testLCellRChoice[newLis, :]
    
    trainRCellLChoice = trainRCellLChoice[1:, :]
    trainRCellRChoice = trainRCellRChoice[1:, :]
    testRCellLChoice = testRCellLChoice[1:, :]
    testRCellRChoice = testRCellRChoice[1:, :]
    
    #allRCellLChoice = np.vstack((allRCellLChoice, RCellLChoice))
    #allRCellRChoice = np.vstack((allRCellRChoice, RCellRChoice))
       
    #newRis = np.argsort(np.argmax((trainRCellRChoice+trainRCellLChoice)/2, axis=1))
    newRis = np.argsort(Rightmups)
    
    #resort data so always in sequence order
    trainRCellLChoice = trainRCellLChoice[newRis, :]
    trainRCellRChoice = trainRCellRChoice[newRis, :]
    testRCellLChoice = testRCellLChoice[newRis, :]
    testRCellRChoice = testRCellRChoice[newRis, :]

    print(region+':'+str(len(newLis)+len(newRis))+', Left:' +str(len(newLis)))


    #normalized plots
    for i in range(len(newLis)):
        M = np.nanmax([np.nanmax(testLCellLChoice[i, :]), np.nanmax(testLCellRChoice[i, :])])
        m = np.nanmin([np.nanmin(testLCellLChoice[i, :]), np.nanmin(testLCellRChoice[i, :])])
        trainLCellLChoice[i, :] = (trainLCellLChoice[i, :]-m)/(M-m)
        trainLCellRChoice[i, :] = (trainLCellRChoice[i, :]-m)/(M-m)
        #M = np.nanmax([np.nanmax(testLCellLChoice[i, :]), np.nanmax(testLCellRChoice[i, :])])
        #m = np.nanmin([np.nanmin(testLCellLChoice[i, :]), np.nanmin(testLCellRChoice[i, :])])        
        testLCellLChoice[i, :] = (testLCellLChoice[i, :]-m)/(M-m)
        testLCellRChoice[i, :] = (testLCellRChoice[i, :]-m)/(M-m)
        
    for i in range(len(newRis)):
        M = np.nanmax([np.nanmax(testRCellLChoice[i, :]), np.nanmax(testRCellRChoice[i, :])])
        m = np.nanmin([np.nanmin(testRCellLChoice[i, :]), np.nanmin(testRCellRChoice[i, :])])
        trainRCellLChoice[i, :] = (trainRCellLChoice[i, :]-m)/(M-m)
        trainRCellRChoice[i, :] = (trainRCellRChoice[i, :]-m)/(M-m)
        #M = np.nanmax([np.nanmax(testRCellLChoice[i, :]), np.nanmax(testRCellRChoice[i, :])])
        #m = np.nanmin([np.nanmin(testRCellLChoice[i, :]), np.nanmin(testRCellRChoice[i, :])])
        testRCellLChoice[i, :] = (testRCellLChoice[i, :]-m)/(M-m)
        testRCellRChoice[i, :] = (testRCellRChoice[i, :]-m)/(M-m)

    
    fig, ax = plt.subplots(2, 2, gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [len(newLis), len(newRis)]})
    
    vm = 1
    vl = 0
    
    ax[0, 0].imshow(trainLCellLChoice, cmap = 'Greys', aspect='auto', vmin=vl, vmax=vm)
    ax[0, 0].set_title('left choice trials')
    ax[0, 0].set_ylabel('left pref.')
    ax[0,0].set_xticks([])
    ax[0,0].set_yticks([])
    
    ax[0, 1].imshow(trainLCellRChoice, cmap = 'Greys', aspect='auto', vmin=vl, vmax=vm)
    ax[0, 1].set_title('right choice trials')
    ax[0,1].set_xticks([])
    ax[0,1].set_yticks([])
    
    ax[1, 0].imshow(trainRCellLChoice, cmap = 'Greys', aspect='auto', vmin=vl, vmax=vm)
    ax[1, 0].set_ylabel('right pref.')
    ax[1,0].set_xticks([])
    ax[1,0].set_yticks([])
    
    ax[1, 1].imshow(trainRCellRChoice, cmap = 'Greys', aspect='auto', vmin=vl, vmax=vm)
    ax[1,1].set_xticks([])
    ax[1,1].set_yticks([])

    plt.suptitle(region)
    plt.savefig('FitSequences/'+region+'-eventrials.pdf')
    
    fig, ax = plt.subplots(2, 2, gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [len(newLis), len(newRis)]})
    
    vm = 1
    vl = 0
    
    ax[0, 0].imshow(testLCellLChoice, cmap = 'Greys', aspect='auto', vmin=vl, vmax=vm)
    ax[0, 0].set_title('left choice trials')
    ax[0, 0].set_ylabel('left pref.')
    ax[0,0].set_xticks([])
    ax[0,0].set_yticks([])
    
    ax[0, 1].imshow(testLCellRChoice, cmap = 'Greys', aspect='auto', vmin=vl, vmax=vm)
    ax[0, 1].set_title('right choice trials')
    ax[0,1].set_xticks([])
    ax[0,1].set_yticks([])
    
    ax[1, 0].imshow(testRCellLChoice, cmap = 'Greys', aspect='auto', vmin=vl, vmax=vm)
    ax[1, 0].set_ylabel('right pref.')
    ax[1,0].set_xticks([])
    ax[1,0].set_yticks([])
    
    ax[1, 1].imshow(testRCellRChoice, cmap = 'Greys', aspect='auto', vmin=vl, vmax=vm)
    ax[1,1].set_xticks([])
    ax[1,1].set_yticks([])

    plt.suptitle(region)
    plt.savefig('FitSequences/'+region+'-oddtrials.pdf')   

'''    
allnewLis = np.argsort(allLeftmups)
allnewRis = np.argsort(allRightmups)

allLCellLChoice = allLCellLChoice[1:, :]
allLCellRChoice = allLCellRChoice[1:, :]    

allRCellLChoice = allRCellLChoice[1:, :]
allRCellRChoice = allRCellRChoice[1:, :]      
    
allLCellLChoice = allLCellLChoice[allnewLis, :]
allLCellRChoice = allLCellRChoice[allnewLis, :]
allRCellLChoice = allRCellLChoice[allnewRis, :]
allRCellRChoice = allRCellRChoice[allnewRis, :]

#normalized plots
for i in range(len(allnewLis)):
    M = np.nanmax([np.nanmax(allLCellLChoice[i, :]), np.nanmax(allLCellRChoice[i, :])])
    m = np.nanmin([np.nanmin(allLCellLChoice[i, :]), np.nanmin(allLCellRChoice[i, :])])
    allLCellLChoice[i, :] = (allLCellLChoice[i, :]-m)/(M-m)
    allLCellRChoice[i, :] = (allLCellRChoice[i, :]-m)/(M-m)
    
for i in range(len(allnewRis)):
    M = np.nanmax([np.nanmax(allRCellLChoice[i, :]), np.nanmax(allRCellRChoice[i, :])])
    m = np.nanmin([np.nanmin(allRCellLChoice[i, :]), np.nanmin(allRCellRChoice[i, :])])
    allRCellLChoice[i, :] = (allRCellLChoice[i, :]-m)/(M-m)
    allRCellRChoice[i, :] = (allRCellRChoice[i, :]-m)/(M-m)

fig, ax = plt.subplots(2, 2, gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [len(allnewLis), len(allnewRis)]})

vm = 1
vl = 0

ax[0, 0].imshow(allLCellLChoice, cmap = 'Greys', aspect='auto', vmin=vl, vmax=vm)
ax[0, 0].set_title('left choice trials')
ax[0, 0].set_ylabel('left pref.')
ax[0,0].set_xticks([])
ax[0,0].set_yticks([])

ax[0, 1].imshow(allLCellRChoice, cmap = 'Greys', aspect='auto', vmin=vl, vmax=vm)
ax[0, 1].set_title('right choice trials')
ax[0,1].set_xticks([])
ax[0,1].set_yticks([])

ax[1, 0].imshow(allRCellLChoice, cmap = 'Greys', aspect='auto', vmin=vl, vmax=vm)
ax[1, 0].set_ylabel('right pref.')
ax[1,0].set_xticks([])
ax[1,0].set_yticks([])

ax[1, 1].imshow(allRCellRChoice, cmap = 'Greys', aspect='auto', vmin=vl, vmax=vm)
ax[1,1].set_xticks([])
ax[1,1].set_yticks([])

plt.suptitle('All Regions')
plt.savefig('FitSequences/ALLregions-V1removed.pdf')
'''
