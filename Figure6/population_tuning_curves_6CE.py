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
import os
import pandas as pd

#demo
demo = True #True to run with example datasets, False to run for full data in structured folders

#set threshold on cells for significant evidence tuning
sigthresh = .05

#set nan color to gray
current_cmap = matplotlib.cm.get_cmap('YlOrRd')
current_cmap.set_bad(color='black', alpha=.3) 


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
    return cum_ev

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

def get_tuningcurve(data, evidence, tuningdict):
    '''
    build dictionary of neural firing rates in each position x evidence bin

    === inputs ===
    data: neural firing data at each position on each trial
    evidence: cumulative evidence at each position on each trial
    tuningdict: current dictionary of neural firing rates with keys tuples of evidence and position indices and values an array of observed firing rates

    === outputs ===
    tuningdict: updated dictionary of neural firing rates

    '''
    evs = np.arange(-15, 16) #get evidence bins
    #loop over position and evidence bins
    for p in range(66):
        for i, e in enumerate(evs):
            validis = np.where(evidence[:, p]==e)[0] #find trials for which cumulative evidence in that position x evidecne bin
            if len(validis)>2: #at least three trials
                if (i, p) in tuningdict: #check if already in the dictionary
                    tuningdict[(i, p)] = np.concatenate((tuningdict[(i, p)], np.nanmean(data[validis, :, p], axis=0))) #append average activity of each neuron in that position and evidence bin
                else:
                    tuningdict[(i, p)] = np.nanmean(data[validis, :, p], axis=0)
    return tuningdict

def jointdict_to_tuning(tuningdictright, tuningdictleft):
    '''
    take tuning dictionaries and convert to population averages

    ===inputs===
    tuningdictright: dictionary with keys tuples of evidence and position and values lists of mean firing rate in that bin of each right preferring neuron
    tuningdictleft: dictionary with keys tuples of evidence and position and values lists of mean firing rate in that bin of each left preferring neuron

    ===outputs===
    array of average activity in preferred evidence by position bins
    '''
    evs = np.arange(-15, 16) #define evidence bins
    obs = np.zeros((len(evs), 66))
    for p in range(66):
        for i, e in enumerate(evs): #loop over all position x evidence bins
            ri = np.where(evs==-1*e)[0][0] #reverse right preferring cells to be in preferred evidence space
            if (i, p) in tuningdictleft:
                obsL = tuningdictleft[(i,p)] #list of average activity of each left preferring cell in bin
            else:
                obsL = []
            if (ri, p) in tuningdictright:
                obsR = tuningdictright[(ri,p)] #list of average activity of each right preferring cell in preferred evidence bin
            else:
                obsR = [] 
            if len(obsL)+len(obsR)>0: #at least some neurons there
                obs[i, p] = np.mean(np.concatenate((obsL, obsR))) #take average across all neurons
            else:
                obs[i, p] = np.nan
    return obs

if not demo:
    regions = ['ACC', 'DMS', 'RSC', 'HPC']

    #get session files for each region
    files = os.listdir('./DMS')
    DMSmatfiles = [f for f in files if f.startswith('dFF_scott')]

    files = os.listdir('./ACC')
    ACCmatfiles = [f for f in files if f.startswith('dFF_tet')]

    files = os.listdir('./RSC')
    RSCmatfiles = [f for f in files if f.startswith('nic')]

    files = os.listdir('./HPC')
    HPCmatfiles = [f for f in files if f.startswith('nic')]

    filelist = [ACCmatfiles, DMSmatfiles,  RSCmatfiles,  HPCmatfiles]

else:
    regions = ['ACC']
    matfiles = ['ExampleData/ACCsessionexample.mat']
    filelist = [matfiles]

#define range of mean firing rate plots
vm = {'ACC':1.8, 'DMS': 1.9, 'HPC': 1.95, 'RSC': 1.55, 'V1':1.3}
vM = {'ACC':2.3, 'DMS': 2.35, 'HPC':2.15, 'RSC':2.0, 'V1': 1.7}

#define colors for each position
colors = {16:'darkslateblue', 26:'dodgerblue', 36:'aqua', 46:'purple', 56:'fuchsia'}

for region, matfiles in zip(regions, filelist):
    regiontuningdictL = {}
    regiontuningdictR = {}
          
    Leftmups = np.array([])
    Rightmups = np.array([])
    
    if not demo:
        fitparams = pd.read_csv(region+'/paramfit/'+region+'allfitparams.csv') #load parameters from joint gaussian fit
    else:
        fitparams = pd.read_csv('ExampleData/ACCparamfitexample.csv')
    
    for file in matfiles:
        if not demo:
            data = loadmat(region+'/'+file) #read in datafile
        else:
            data = loadmat(file)
        
        if region in ['DMS', 'ACC']: #use correct method to get data in same output format
            if not demo:
                bfile = file.split('dFF_')[1]
                bfile = bfile.split('processedOutput.mat')[0]+'.mat'
                cuedata = loadmat(region+'/Behavior/'+bfile)
            else:
                cuedata = loadmat('ExampleData/ACCsessionexample-behavior.mat')
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
            
        else: #get session data in correct output format
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
        
        if not demo:
            session = file.split('.')[0]
        else:
            session = 'dFF_tetO_8_07282021_T10processedOutput'
        fileparams = fitparams[fitparams['Session']==session]
        pvals = fileparams['Pval'].values
        mues = fileparams['Mue'].values
        mups = fileparams['Mup'].values
        siges = fileparams['Sige'].values
        corrs = fileparams['Correlation'].values
        
        #find neurons with significant evidence tuning
        vp = pvals<sigthresh
        
        #get only evidence selective parameters
        evselectiveneurons = fileparams['Neuron'].values[vp]
        mues = mues[vp]
        mups = mups[vp]
        alldata = alldata[:, evselectiveneurons, :]
        
        #calculate cumulative evidence at each position
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
        
        #divide cells into left and right by sign of fit evidence mean
        leftis = np.where(mues>0)[0]
        rightis = np.where(mues<0)[0]        
        leftmups = mups[leftis]
        rightmups = mups[rightis]
        
        #get tuning for left preferring cells
        if len(leftis)>0:        
            leftactivity = alldata[:, leftis, :]
            lefttotal = leftactivity
            regiontuningdictL = get_tuningcurve(lefttotal, evidence, regiontuningdictL) 
        else:
            lefttotal = np.zeros((len(correcttrials), n_pos))
        
        #get tuning for right preferring cells
        if len(rightis)>0:
            rightactivity = alldata[:, rightis, :] 
            righttotal = rightactivity
            regiontuningdictR = get_tuningcurve(righttotal, evidence, regiontuningdictR)
        else:
            righttotal = np.zeros((len(correcttrials), n_pos))
             
    #convertion from tuning dictionaries to arrays
    poptuningcombined = jointdict_to_tuning(regiontuningdictL, regiontuningdictR)

    #plot 2D population tuning curve    
    plt.figure()
    plt.imshow(poptuningcombined, cmap='YlOrRd', vmin=vm[region], vmax = vM[region], aspect='auto', interpolation='none')
    plt.xticks([0, 5.5, 25.5, 45.5, 55.25, 65], labels=['-30', '0', 'cues', '200', 'delay', '300'])
    plt.yticks([5, 15, 25], ['-10', '0', '10'])
    for c, i in enumerate([16, 26, 36, 46, 56]):
        plt.axvline(i, color=colors[i], linewidth=2.5, linestyle = '--')
    plt.title(region+' Preferred - Non Pref Population Tuning')
    plt.colorbar()
    plt.ylim([25,5])
    plt.show()
    
    #plot crosssections
    evs = np.arange(-15, 16)
    plt.figure()
    for c, i in enumerate([16, 26, 36, 46, 56][::-1]):
        plt.plot(evs, poptuningcombined[:, i][::-1], color=colors[i], linewidth = 3)
    plt.title(region+' Preferred - Non Pref')
    plt.xlim([-10, 10])
    plt.ylim([vm[region], vM[region]])
    plt.show()