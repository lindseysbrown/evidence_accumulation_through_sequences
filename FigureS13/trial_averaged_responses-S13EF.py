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
from scipy import stats
from scipy.stats import sem, f
import pickle
import os
import pandas as pd

demo = True

def get_cum_evidence(lefts, rights, pos):
    '''
    calculate cumulative evidence at each position in the maze
    === inputs ===
    lefts: list of positions of left towers
    rights: list of positions of right towers
    pos: list of maze positions
    '''
    cum_ev = np.zeros(len(pos))
    for i, p in enumerate(pos):
        cum_ev[i] = np.sum(lefts<p)-np.sum(rights<p)
    return -1*cum_ev

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

if not demo:
    regions = ['ACC', 'RSC']

    #get session files for each region
    files = os.listdir('./ACC')
    ACCmatfiles = [f for f in files if f.startswith('dFF_tet')]

    files = os.listdir('./RSC')
    RSCmatfiles = [f for f in files if f.startswith('nic')]

    filelist = [ACCmatfiles, RSCmatfiles]

else:
    regions = ['ACC']
    matfiles = ['ExampleData/ACCsessionexample.mat']
    filelist = [matfiles]

with open('ExampleData/numtotaltrials.pkl', 'rb') as handle:
    numcorrecttrials = pickle.load(handle)

for region, matfiles in zip(regions[:], filelist[:]):
    if not demo:
        fitparamsencoding = np.load(region+'/paramfit/'+region+'allfitparams-linearencoding.npy') #load saved parameters from running encoding model
        fitneurondata = pd.read_csv(region+'/paramfit/'+region+'allfitparams-linearencoding-neuroninfo.csv') #load neuron, session information corresponding to parameters

    else:
        fitparamsencoding = np.load('ExampleData/ACCsingleneuronencoding.npy')
        fitneurondata = pd.read_csv('ExampleData/ACCsingleneuronencoding-neuroninfo.csv')
    
    sigF = np.zeros(len(fitneurondata))
    
    #determine which neurons have significant evidence coefficient
    for j in range(len(sigF)):
        session = fitneurondata['Session'].values[j]
        ncorrect = numcorrecttrials[session]
        sigF[j] = f.ppf(.99, 1, ncorrect-3)
    
    #set up arrays to hold trials for different final evidence levels
    npo = 66 #number of positions
    finaldiff1to4 = np.zeros((1, npo))
    finaldiff5to9 = np.zeros((1, npo))
    finaldiffgreater9 = np.zeros((1, npo))
    negfinaldiff1to4 = np.zeros((1, npo))
    negfinaldiff5to9 = np.zeros((1, npo))
    negfinaldiffgreater9 = np.zeros((1, npo))
    
    #load parameters of gaussians fit to half Gaussian filtered data
    if not demo:
        fitparams = pd.read_csv(region+'/paramfit/'+region+'allfitparams-HG.csv')
    else:
        fitparams = pd.read_csv('ExampleData/ACCfitparams-halfgaussian.csv')
    for file in matfiles:
        #load firing data
        if not demo:
            data = loadmat(region+'/HalfGaussianFiringRate/'+file)
        else:
            data = loadmat('ExampleData/ACCsessionexample.mat')
        
        if region in ['DMS', 'ACC']:
            if not demo:
                bfile = file.split('dFF_')[1]
                bfile = bfile.split('processedOutput-NEW.mat')[0]+'.mat'
                cuedata = loadmat(region+'/Behavior/'+bfile)
            else:
                cuedata = loadmat('ExampleData/ACCsessionexample-behavior.mat')
            lefts = cuedata['logSumm']['cuePos_L'][0][0][0]
            rights = cuedata['logSumm']['cuePos_R'][0][0][0]
            
            #create 3d array (trials x neurons x timepoints)
            n_neurons = np.shape(data['out']['FRHG_pos'][0][0])[1]
            [n_trials, n_pos] = np.shape(data['out']['FRHG_pos'][0][0][0][0]) 
            
            alldata = np.zeros((n_trials, n_neurons, n_pos))
            
            maintrials = data['out']['Trial_Main_Maze'][0][0][0]-1 #subtract one for difference in matlab indexing
            correcttrials = data['out']['correct'][0][0][0]-1
            
            #get data normalized by position
            for i in range(n_neurons):
                alldata[:, i, :] = data['out']['FRHG_pos'][0][0][0][i]
            
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
            ndata = data['nic_output']['firingratehalfgauss'][0][0]
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
            
        if not demo:
            session = file.split('-NEW.')[0]
        else:
            session = 'dFF_tetO_8_07282021_T10processedOutput'
        evselectiveneurons = []
        choiceselectiveneurons = []
        
        #identify which neurons had a significant evidence coefficient in the single neuron encoding model
        for n in range(n_neurons): 
            try:
                index =  np.where((fitneurondata['Session']==session) & (fitneurondata['Index']==n))[0][0]
                
                sigE = sum(fitparamsencoding[:, 3, index]>sigF[index])>0
                sigC = sum(fitparamsencoding[:, 4, index]>sigF[index])>0
                if sigE:
                    evselectiveneurons.append(n)
                else:
                    if sigC:
                        choiceselectiveneurons.append(n)
            except:
                'index did not exist'

        
        
        fileparams = fitparams[fitparams['Session']==session]
        pvals = fileparams['PvalMSE'].values
        mues = fileparams['Mue'].values
        mups = fileparams['Mup'].values
        sigps = fileparams['Sigp'].values
        siges = fileparams['Sige'].values
        corrs = fileparams['Correlation'].values
        nids = fileparams['Neuron'].values
        

        vp = []
        for eid in evselectiveneurons:
            try:
                vp.append(np.where(nids==eid)[0][0])
            except:
                print('missing idx in gaussian fit')
            
        #get only evidence selective parameters
        mues = mues[vp]
        mups = mups[vp]
        sigps = sigps[vp]
        nids = nids[vp]
        alldata = alldata[:, nids, :]
        
        #get cumulative evidence at each position on each trial
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
        leftsigps = sigps[leftis]
        rightsigps = sigps[rightis]
        
        #get activity of position active left preferring neurons
        if len(leftis)>0:
            leftactivity = alldata[:, leftis, :]
            lefttotal = np.nanmean(leftactivity, axis=1)
            for i, p in enumerate(pos):
                validns = np.where(np.abs(leftmups-p)<leftsigps)[0] #only active cells
                lefttotal[:, i] = np.nanmean(leftactivity[:, validns, i], axis=1)           
        else:
            lefttotal = np.nan*np.zeros((len(correcttrials), n_pos))

        #get activity of position active right preferring neurons  
        if len(rightis)>0:
            rightactivity = alldata[:, rightis, :] 
            righttotal = np.nanmean(rightactivity, axis = 1)
            for i, p in enumerate(pos):
                validns = np.where(np.abs(rightmups-p)<rightsigps)[0]
                righttotal[:, i] = np.nanmean(rightactivity[:, validns, i], axis=1)            
        else:
            righttotal = np.nan*np.zeros((len(correcttrials), n_pos))
        
        
        lefttotal = lefttotal[:, :]-np.mean(lefttotal[:, :], axis=0)
        righttotal = righttotal[:, :]-np.mean(righttotal[:, :], axis=0)
        diff = righttotal-lefttotal #difference in activity

        #add each trial to appropriate array based on final evidence
        for i, e in enumerate(evidence[:, -1]):
            if (e>0) and (e<4): #low right edvidence
                finaldiff1to4 = np.vstack((finaldiff1to4, diff[i, :]))

            if (e>3) and (e<7): #medium right evidence
                finaldiff5to9 = np.vstack((finaldiff5to9, diff[i, :]))

            if e>6: #high right evidence
                finaldiffgreater9 = np.vstack((finaldiffgreater9, diff[i, :]))

            if (e<0) and (e>-4): #low left evidence
                negfinaldiff1to4 = np.vstack((negfinaldiff1to4, diff[i, :]))
 
            if (e<-3) and (e>-7): #medium left evidence
                negfinaldiff5to9 = np.vstack((negfinaldiff5to9, diff[i, :]))

            if e<-6: #high left evidence
                negfinaldiffgreater9 = np.vstack((negfinaldiffgreater9, diff[i, :]))

                
    #remove initialized zero array
    finaldiff1to4 = finaldiff1to4[1:]
    finaldiff5to9 = finaldiff5to9[1:]
    finaldiffgrater9 = finaldiffgreater9[1:]
    negfinaldiff1to4 = negfinaldiff1to4[1:]
    negfinaldiff5to9 = negfinaldiff5to9[1:]
    negfinaldiffgrater9 = negfinaldiffgreater9[1:]

    #plot binned trial averages
    plt.figure()
    plt.errorbar(pos, np.nanmean(finaldiff1to4, axis=0), yerr = sem(finaldiff1to4, axis=0, nan_policy = 'omit'), color = 'r', alpha = .33)
    plt.errorbar(pos, np.nanmean(finaldiff5to9, axis=0), yerr = sem(finaldiff5to9, axis=0, nan_policy = 'omit'), color = 'r', alpha = .67)
    plt.errorbar(pos, np.nanmean(finaldiffgreater9, axis=0), yerr = sem(finaldiffgreater9, axis=0, nan_policy = 'omit'), color = 'r', alpha = 1)
    plt.errorbar(pos, np.nanmean(negfinaldiff1to4, axis=0), yerr = sem(negfinaldiff1to4, axis=0, nan_policy = 'omit'), color = 'b', alpha = .33)
    plt.errorbar(pos, np.nanmean(negfinaldiff5to9, axis=0), yerr = sem(negfinaldiff5to9, axis=0, nan_policy = 'omit'), color = 'b', alpha = .67)
    plt.errorbar(pos, np.nanmean(negfinaldiffgreater9, axis=0), yerr = sem(negfinaldiffgreater9, axis=0, nan_policy = 'omit'), color = 'b', alpha = 1)
    plt.title(region)
    plt.ylabel('Activity (R-L)')
    plt.show()

 
