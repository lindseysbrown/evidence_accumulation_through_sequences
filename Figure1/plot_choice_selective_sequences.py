# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 10:19:35 2023

@author: lindseyb
"""

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
from scipy.stats import f_oneway, kurtosis, skew
import os
from scipy.optimize import curve_fit
import diptest
import pandas as pd

#set plot labels larger
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Arial']})


#set parameters
sigthresh = .05 #threshold for significance for evidence selectivity


def get_pos_out(data, position, trial, Lcuepos, Rcuepos, mazeID, corrects, choices):
    '''
    function to get same output format as ACC and DMS data for the RSC and HPC data

    ===inputs===
    data: array of neural firing data
    position: positions corresponding to neural data
    trial: trial numbers correspondig to each data point
    Lcuepos: position of left cues
    Rcuepos: position of right cues
    mazeID: level of the maze in the shaping protocol
    corrects: mapping of whether each trial was correct
    choices: mapping of which choice the animal made on each trial

    ===outputs===
    neuraldata: array of neural data of shape (trials, neurons, timepoints)
    trialmap: mapping between trial index and trial number
    maintrials: trials for which accumulation of evidence was required
    correcttrials: trials for which the animal was correct
    lefts: left cues for each trial
    rights: right cues for each trial
    leftchoices: trials for which the animal made a left choice
    rightchoices: trials for which the animal made a right choice
    '''

    #get relevant data shapes
    n_neurons = np.shape(data)[1]
    trials = list(set([x[0] for x in trial]))
    n_trials = len(trials)
    n_pos = 66
    base = 5

    #setup output arrays
    maintrials = []
    correcttrials = []
    leftchoices = []
    rightchoices = []
    lefts = []
    rights = []
    neuraldata = np.zeros((n_trials, n_neurons, n_pos))
    trialmap = np.zeros((n_trials,))

    #for each trial in the list
    for it, t in enumerate(trials):
        trialmap[it] = t
        inds = np.where(trial==t)[0]
        maze = mazeID[inds[0]]
        if maze >9: #check that trial is an accumulation of evidence maze
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
        posbinned = base*np.round(pos/base) #assign position bin to each trial
        avgbypos = np.zeros((n_pos, n_neurons))
        for ip, p in enumerate(range(-30, 300, 5)):
            pis = np.where(posbinned == p)[0]
            if len(pis)>0:
                avgbypos[ip, :] = np.nanmean(trialdata[pis, :], axis=0) #take the average of all data within the position bin
            else:
                avgbypos[ip, :] = avgbypos[ip-1, :]
        neuraldata[it, :, :] = avgbypos.T
    return np.nan_to_num(neuraldata), trialmap, maintrials, correcttrials, lefts, rights, leftchoices, rightchoices

regions = ['ACC', 'DMS', 'HPC', 'RSC']

#identify corresponding matlab files with relevant trial and behavior data
files = os.listdir('./DMS')
DMSmatfiles = [f for f in files if f.startswith('dFF_scott')]

files = os.listdir('./ACC')
ACCmatfiles = [f for f in files if f.startswith('dFF_tet')]

files = os.listdir('./RSC')
RSCmatfiles = [f for f in files if f.startswith('nic')]

files = os.listdir('./HPC')
HPCmatfiles = [f for f in files if f.startswith('nic')]

filelist = [ACCmatfiles, DMSmatfiles,  HPCmatfiles,  RSCmatfiles]


#set up arrays for average activity data with shape 1 x number of positions
allLCellLChoice = np.zeros((1, 66))
allLCellRChoice = np.zeros((1, 66))

allRCellLChoice = np.zeros((1, 66))
allRCellRChoice = np.zeros((1, 66))

#save position means to be able to sort data
allLeftmups = np.array([])
allRightmups = np.array([])

#for each region, get average activity of left and right preferring cells, using fit mean position and evidence
for region, matfiles in zip(regions, filelist):
    print(region)
    #set up arrays for average activity data for single region
    sortedcells = pd.DataFrame()
    LCellLChoice = np.zeros((1, 66))
    LCellRChoice = np.zeros((1, 66))
    
    RCellLChoice = np.zeros((1, 66))
    RCellRChoice = np.zeros((1, 66))
    
    SCellLChoice = np.zeros((1, 66))
    SCellRChoice = np.zeros((1, 66))
       
    Leftmups = np.array([])
    Rightmups = np.array([])

    #load data from joint gaussian fits 
    fitparamsCS = pd.read_csv(region+'/paramfit/'+region+'allfitparams.csv')
    fitparamssplit = pd.read_csv(region+'/paramfit/'+region+'allfitparams-split.csv')
    fitparams = pd.concat([fitparamsCS, fitparamssplit], ignore_index=True)

    #iterate over each session
    for file in matfiles:
        data = loadmat(region+'/'+file)
        
        #load behavioral trial data from preprocessing for ACC and DMS regions
        if region in ['DMS', 'ACC']:
            #create 3d array (trials x neurons x timepoints)
            n_neurons = np.shape(data['out']['FR2_pos'][0][0])[1]
            [n_trials, n_pos] = np.shape(data['out']['FR2_pos'][0][0][0][0]) 
            
            alldata = np.zeros((n_trials, n_neurons, n_pos))
            
            maintrials = data['out']['Trial_Main_Maze'][0][0][0]-1 #subtract one for difference in matlab indexing
            correcttrials = data['out']['correct'][0][0][0]-1
            
            #load data for each position
            for i in range(n_neurons):
                alldata[:, i, :] = data['out']['FR2_pos'][0][0][0][i]
            
            #load vector of position locations
            pos = data['out']['Yposition'][0][0][0]
            
            #get different data subsets for left and right choices
            leftchoices_correct = data['out']['correct_left'][0][0][0]-1
            leftchoices_incorrect = data['out']['incorrect_left'][0][0][0]-1
            rightchoices_correct = data['out']['correct_right'][0][0][0]-1
            rightchoices_incorrect = data['out']['incorrect_right'][0][0][0]-1 
            
            lchoices = np.concatenate((leftchoices_correct, leftchoices_incorrect))
            rchoices = np.concatenate((rightchoices_correct, rightchoices_incorrect)) 
        #load behavioral trial data from preprocessing for HPC and RSC    
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

            #get data in the same format as other dataset   
            alldata, tmap, maintrials, correcttrials, lefts, rights, lchoices, rchoices = get_pos_out(ndata, position, trial, Lcuepos, Rcuepos, mazeID, corrects, choices)
            pos = np.arange(-30, 300, 5)
            
            n_trials, n_neurons, n_pos = np.shape(alldata)
        
        #load fit parameters from data
        session = file.split('.')[0]
        fileparams = fitparams[fitparams['Session']==session]
        pvals = fileparams['Pval'].values
        mues = fileparams['Mue'].values
        mups = fileparams['Mup'].values
        siges = fileparams['Sige'].values
        corrs = fileparams['Correlation'].values
        
        #only consider cells with significant evidence tuning
        vp = pvals<sigthresh
        
        #get only evidence selective parameters
        evselectiveneurons = fileparams['Neuron'].values[vp]
        mues = mues[vp]
        mups = mups[vp]
        alldata = alldata[:, evselectiveneurons, :]

        #restrict to correct trials
        choices = np.zeros(n_trials)
        choices[lchoices] = 1
        choices = choices[correcttrials]
        alldata = alldata[correcttrials, :, :]
        lchoices = np.where(choices==1)[0]
        rchoices = np.where(choices==0)[0]

        #define left (leftis) and right (rightis) preferring cells based on the sign of the fit evidence mean (mue)        
        leftis = np.where(mues>0)[0]
        rightis = np.where(mues<0)[0]        

        #find position means for sorting by peak activity
        leftmups = mups[leftis]
        rightmups = mups[rightis]   
        Leftmups = np.concatenate((Leftmups, leftmups))
        Rightmups = np.concatenate((Rightmups, rightmups))
        
        #split data by whether the animal made a left or right choice
        leftdata = alldata[lchoices, :, :]
        rightdata = alldata[rchoices, :, :]

        #append average activity to the appropriate array for left and right preferring cells
        if len(leftis)>0:
            leftcellsleftchoice = np.nanmean(leftdata[:, leftis, :], axis=0)
            leftcellsrightchoice = np.nanmean(rightdata[:, leftis, :], axis=0)
            LCellLChoice = np.vstack((LCellLChoice, leftcellsleftchoice))
            LCellRChoice = np.vstack((LCellRChoice, leftcellsrightchoice))    
        if len(rightis)>0:
            rightcellsleftchoice = np.nanmean(leftdata[:, rightis, :], axis=0)
            rightcellsrightchoice = np.nanmean(rightdata[:, rightis, :], axis=0)
            RCellLChoice = np.vstack((RCellLChoice, rightcellsleftchoice))
            RCellRChoice = np.vstack((RCellRChoice, rightcellsrightchoice))
    
    allLeftmups = np.concatenate((allLeftmups, Leftmups))
    allRightmups = np.concatenate((allRightmups, Rightmups))
    
    #remove initialized zero array
    LCellLChoice = LCellLChoice[1:, :]
    LCellRChoice = LCellRChoice[1:, :]
    
    allLCellLChoice = np.vstack((allLCellLChoice, LCellLChoice))
    allLCellRChoice = np.vstack((allLCellRChoice, LCellRChoice))
    
    #resort leftpreferring cell data so always in sequence order
    newLis = np.argsort(Leftmups)
    LCellLChoice = LCellLChoice[newLis, :]
    LCellRChoice = LCellRChoice[newLis, :]
    
    #remove initialized zero array
    RCellLChoice = RCellLChoice[1:, :]
    RCellRChoice = RCellRChoice[1:, :]
    
    allRCellLChoice = np.vstack((allRCellLChoice, RCellLChoice))
    allRCellRChoice = np.vstack((allRCellRChoice, RCellRChoice))
       
    #resort data so always in sequence order
    newRis = np.argsort(Rightmups)
    RCellLChoice = RCellLChoice[newRis, :]
    RCellRChoice = RCellRChoice[newRis, :]

    #normalize data to [0,1] range
    for i in range(len(newLis)):
        M = np.nanmax([np.nanmax(LCellLChoice[i, :]), np.nanmax(LCellRChoice[i, :])])
        m = np.nanmin([np.nanmin(LCellLChoice[i, :]), np.nanmin(LCellRChoice[i, :])])
        LCellLChoice[i, :] = (LCellLChoice[i, :]-m)/(M-m)
        LCellRChoice[i, :] = (LCellRChoice[i, :]-m)/(M-m)
        
    for i in range(len(newRis)):
        M = np.nanmax([np.nanmax(RCellLChoice[i, :]), np.nanmax(RCellRChoice[i, :])])
        m = np.nanmin([np.nanmin(RCellLChoice[i, :]), np.nanmin(RCellRChoice[i, :])])
        RCellLChoice[i, :] = (RCellLChoice[i, :]-m)/(M-m)
        RCellRChoice[i, :] = (RCellRChoice[i, :]-m)/(M-m)

    #generate sequence plots    
    fig, ax = plt.subplots(2, 2, gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [len(newLis), len(newRis)]})
    
    vm = 1
    vl = 0
    
    ax[0, 0].imshow(LCellLChoice, cmap = 'Greys', aspect='auto', vmin=vl, vmax=vm)
    ax[0, 0].set_title('left choice trials')
    ax[0, 0].set_ylabel('left pref.')
    ax[0,0].set_xticks([])
    ax[0,0].set_yticks([])
    
    ax[0, 1].imshow(LCellRChoice, cmap = 'Greys', aspect='auto', vmin=vl, vmax=vm)
    ax[0, 1].set_title('right choice trials')
    ax[0,1].set_xticks([])
    ax[0,1].set_yticks([])
    
    ax[1, 0].imshow(RCellLChoice, cmap = 'Greys', aspect='auto', vmin=vl, vmax=vm)
    ax[1, 0].set_ylabel('right pref.')
    ax[1,0].set_xticks([])
    ax[1,0].set_yticks([])
    
    ax[1, 1].imshow(RCellRChoice, cmap = 'Greys', aspect='auto', vmin=vl, vmax=vm)
    ax[1,1].set_xticks([])
    ax[1,1].set_yticks([])

    plt.suptitle(region)

#repeat plotting routine for cells of all regions combined    
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
