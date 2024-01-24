# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 10:19:35 2023

@author: lindseyb
"""

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
import os
import pandas as pd

#set threshold on cells for significant evidence tuning
sigthresh = .05

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

files = os.listdir('./DMS')
DMSmatfiles = [f for f in files if f.startswith('dFF_scott')]

files = os.listdir('./ACC')
ACCmatfiles = [f for f in files if f.startswith('dFF_tet')]

files = os.listdir('./RSC')
RSCmatfiles = [f for f in files if f.startswith('nic')]

files = os.listdir('./HPC')
HPCmatfiles = [f for f in files if f.startswith('nic')]

filelist = [ACCmatfiles, DMSmatfiles,  HPCmatfiles,  RSCmatfiles]

allLCellLChoice = np.zeros((1, 66))
allLCellRChoice = np.zeros((1, 66))

allRCellLChoice = np.zeros((1, 66))
allRCellRChoice = np.zeros((1, 66))

allLeftmups = np.array([])
allRightmups = np.array([])

for region, matfiles in zip(regions, filelist): #iterate over brain regions
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
    
    fitparams = pd.read_csv(region+'/paramfit/'+region+'allfitparams.csv') #load parameters from joint gaussian fits
    
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
        
        #get parameters for the session
        session = file.split('.')[0]
        fileparams = fitparams[fitparams['Session']==session]
        pvals = fileparams['Pval'].values
        mues = fileparams['Mue'].values
        mups = fileparams['Mup'].values
        siges = fileparams['Sige'].values
        corrs = fileparams['Correlation'].values
        
        #get only evidence selective parameters
        vp = pvals<sigthresh
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
        
        #determine cell selectivity based on fit evidence mean
        leftis = np.where(mues>0)[0]
        rightis = np.where(mues<0)[0]        
        leftmups = mups[leftis]
        rightmups = mups[rightis]   
        
        Leftmups = np.concatenate((Leftmups, leftmups))
        Rightmups = np.concatenate((Rightmups, rightmups))
        
        
        #work only on the odd trials
        trainalldata = alldata[::2, :, :] #even trials
        testalldata = alldata[1::2, :, :] #odd trials
    
        #get corresponing choices for even/odd trials
        trainlchoices = [int(l/2) for l in lchoices if l%2==0]
        testlchoices = [int(l/2) for l in lchoices if l%2==1]
    
        trainrchoices = [int(r/2) for r in rchoices if r%2==0]
        testrchoices = [int(r/2) for r in rchoices if r%2==1]
        
        #divide neural data by choice
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
    
    #sort by fit mean position
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
    
    #sort by fit mean position
    newRis = np.argsort(Rightmups)
    
    #resort data so always in sequence order
    trainRCellLChoice = trainRCellLChoice[newRis, :]
    trainRCellRChoice = trainRCellRChoice[newRis, :]
    testRCellLChoice = testRCellLChoice[newRis, :]
    testRCellRChoice = testRCellRChoice[newRis, :]

    #normalized plots based on odd trials
    for i in range(len(newLis)):
        M = np.nanmax([np.nanmax(testLCellLChoice[i, :]), np.nanmax(testLCellRChoice[i, :])])
        m = np.nanmin([np.nanmin(testLCellLChoice[i, :]), np.nanmin(testLCellRChoice[i, :])])
        trainLCellLChoice[i, :] = (trainLCellLChoice[i, :]-m)/(M-m)
        trainLCellRChoice[i, :] = (trainLCellRChoice[i, :]-m)/(M-m)     
        testLCellLChoice[i, :] = (testLCellLChoice[i, :]-m)/(M-m)
        testLCellRChoice[i, :] = (testLCellRChoice[i, :]-m)/(M-m)
        
    for i in range(len(newRis)):
        M = np.nanmax([np.nanmax(testRCellLChoice[i, :]), np.nanmax(testRCellRChoice[i, :])])
        m = np.nanmin([np.nanmin(testRCellLChoice[i, :]), np.nanmin(testRCellRChoice[i, :])])
        trainRCellLChoice[i, :] = (trainRCellLChoice[i, :]-m)/(M-m)
        trainRCellRChoice[i, :] = (trainRCellRChoice[i, :]-m)/(M-m)
        testRCellLChoice[i, :] = (testRCellLChoice[i, :]-m)/(M-m)
        testRCellRChoice[i, :] = (testRCellRChoice[i, :]-m)/(M-m)

    #plot even trials
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

    plt.suptitle(region+'even trials')
    plt.show()
    
    #plot odd trials
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

    plt.suptitle(region+' odd trials')
    plt.show() 