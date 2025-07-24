# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 11:14:15 2024

@author: lindseyb
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
import patsy
from scipy.signal import convolve
import os
import pandas as pd
import pickle
from scipy.io import loadmat
from sklearn.linear_model import Ridge
from scipy.stats import sem

demo = True

sigthresh = .05

colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']

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

#get cubic spline basis set
x = np.linspace(0., 300, 60)
y = patsy.bs(x, df=7, degree=3, include_intercept=True)

#second spline basis set for higher evidence values
y2 = patsy.bs(np.linspace(0, 200, 40), df=7, degree=3, include_intercept=True)


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

alphalist = [1]

for region, matfiles in zip(regions, filelist):
    #load parameters from joint gaussian fit
    if not demo:
        fitparams = pd.read_csv(region+'/paramfit/'+region+'allfitparams.csv')
    else:
        fitparams = pd.read_csv('ExampleData/ACCparamfitexample.csv')
    fits = fitparams['MSE'].values[fitparams['PvalMSE'].values < .05]
    lefttoLcuekernelsearly = {}
    lefttoRcuekernelsearly = {}
    lefttoLcuekernelsmed = {}
    lefttoRcuekernelsmed = {}
    righttoLcuekernelsearly = {}
    righttoRcuekernelsearly = {}
    lefttoLcuekernelslate = {}
    lefttoRcuekernelslate = {}
    righttoLcuekernelslate = {}
    righttoRcuekernelslate = {}
    performance = {}
    
    for a in alphalist:
        lefttoLcuekernelsearly[a] = np.zeros((1, 60))
        lefttoRcuekernelsearly[a] = np.zeros((1, 60))
        righttoLcuekernelsearly[a] = np.zeros((1, 60))
        righttoRcuekernelsearly[a] = np.zeros((1, 60))
        lefttoLcuekernelsmed[a] = np.zeros((1, 40))
        lefttoRcuekernelsmed[a] = np.zeros((1,40))
        lefttoLcuekernelslate[a] = np.zeros((1, 40))
        lefttoRcuekernelslate[a] = np.zeros((1, 40))
        righttoLcuekernelslate[a] = np.zeros((1, 40))
        righttoRcuekernelslate[a] = np.zeros((1, 40)) 
        performance[a] = []
    
    for f, file in enumerate(matfiles):
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
        fileparams = fitparams[fitparams['Session']==session]
        pvals = fileparams['PvalMSE'].values
        mues = fileparams['Mue'].values
        mups = fileparams['Mup'].values
        sigps = fileparams['Sigp'].values
        siges = fileparams['Sige'].values
        corrs = fileparams['Correlation'].values
        fits = fileparams['MSE'].values
        
        vp = pvals<sigthresh
        
        #get only evidence selective parameters
        evselectiveneurons = fileparams['Neuron'].values[vp]
        mues = mues[vp]
        mups = mups[vp]
        sigps = sigps[vp]
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
        leftsigps = sigps[leftis]
        rightsigps = sigps[rightis]
        

        
        #create feature matrix
        X = np.zeros((len(correcttrials)*(n_pos-5), 44))
        
        for idx, t in enumerate(correcttrials):
            ps = pos[5:]
            es = np.sign(evidence[idx, 5:])
            ps = ps*es
            if region in ['ACC', 'DMS']:
                lcues = lefts[t][0]
                rcues = rights[t][0]
            else:
                lcues = lefts[t]
                rcues = rights[t]
            Lcuelocsearly = np.zeros(n_pos,)
            Rcuelocsearly = np.zeros(n_pos,)
            Lcuelocsmed = np.zeros(n_pos,)
            Rcuelocsmed = np.zeros(n_pos,)
            Lcuelocslate =np.zeros(n_pos,)
            Rcuelocslate = np.zeros(n_pos,)
            feats = np.concatenate((0*(ps).reshape(-1, 1), 0*es.reshape(-1, 1)), axis = 1)
            for l in lcues:
                cuebin = np.where((pos-l)>0)[0][0]-1 #bin where cue appears
                eatcue = evidence[idx, cuebin-2]
                if np.abs(eatcue)<2: #low evidence
                    Lcuelocsearly[cuebin-2] = 1 #locked to cue appearance
                elif np.abs(eatcue)<5: #medium evidence
                   Lcuelocsmed[cuebin-2] = 1 
                else: #high evidence
                    Lcuelocslate[cuebin-2] = 1
            for r in rcues:
                cuebin = np.where((pos-r)>0)[0][0]-1
                eatcue = evidence[idx, cuebin-2]
                if np.abs(eatcue)<2: #low evidence
                    Rcuelocsearly[cuebin-2] = 1 #locked to cue appearance
                elif np.abs(eatcue)<5: #medium evidence
                    Rcuelocsmed[cuebin-2] = 1
                else: #high evidence
                    Rcuelocslate[cuebin-2] = 1
            for spl in range(7): #convolve cue times with basis set
                zLearly = convolve(Lcuelocsearly[5:], y[:, spl], mode = 'full')[:(n_pos-5)]
                zRearly = convolve(Rcuelocsearly[5:], y[:, spl], mode = 'full')[:(n_pos-5)]
                zLmed = convolve(Lcuelocsmed[5:], y2[:, spl], mode = 'full')[:(n_pos-5)]
                zRmed = convolve(Rcuelocsmed[5:], y2[:, spl], mode = 'full')[:(n_pos-5)]
                zLlate = convolve(Lcuelocslate[5:], y2[:, spl], mode = 'full')[:(n_pos-5)]
                zRlate = convolve(Rcuelocslate[5:], y2[:, spl], mode = 'full')[:(n_pos-5)]
                feats = np.concatenate((feats, zLearly.reshape(-1, 1), zRearly.reshape(-1, 1), zLmed.reshape(-1,1), zRmed.reshape(-1,1), zLlate.reshape(-1, 1), zRlate.reshape(-1, 1)), axis = 1)
            X[idx*(n_pos-5):(idx+1)*(n_pos-5), :] = feats
        
        if len(leftis)>0 and len(rightis)>0:
            leftactivity = alldata[:, leftis, :]
            lefttotal = np.nanmean(leftactivity, axis=1)
            for i, p in enumerate(pos):
                validns = np.where(np.abs(leftmups-p)<leftsigps)[0]
                lefttotal[:, i] = np.nanmean(leftactivity[:, validns, i], axis=1)
            lefttotal = lefttotal[:, 5:] #predict only from start of cue region
            
            rightactivity = alldata[:, rightis, :] 
            righttotal = np.nanmean(rightactivity, axis = 1)
            for i, p in enumerate(pos):
                validns = np.where(np.abs(rightmups-p)<rightsigps)[0]
                righttotal[:, i] = np.nanmean(rightactivity[:, validns, i], axis=1) 

            righttotal = righttotal[:, 5:]
            
            #predict difference in activity
            difftotal = (lefttotal - np.nanmean(lefttotal, axis=0))-(righttotal-np.nanmean(righttotal, axis=0))
            
            yd = difftotal.flatten()            

            X = X[~np.isnan(yd)]
            yd = yd[~np.isnan(yd)]
            
            #fit ridge regression model for different regularization strengths
            for a in alphalist:
                model = Ridge(alpha=a)
                model.fit(X, yd)
                coefs = model.coef_
                
                ypred = model.predict(X)
                   
                #calculate kernels from spliens and fit coefficients
                Lcuekernelearly = y[:, :7]@coefs[2:44:6] 
                Rcuekernelearly = y[:, :7]@coefs[3:44:6]
                Lcuekernelmed = y2[:, :7]@coefs[4:44:6]
                Rcuekernelmed = y2[:, :7]@coefs[5:44:6]
                Lcuekernellate = y2[:, :7]@coefs[6:44:6]
                Rcuekernellate = y2[:, :7]@coefs[7:44:6]                
        
                lefttoLcuekernelsearly[a] = np.vstack((lefttoLcuekernelsearly[a], Lcuekernelearly))
                lefttoRcuekernelsearly[a] = np.vstack((lefttoRcuekernelsearly[a], Rcuekernelearly))
                lefttoLcuekernelsmed[a] = np.vstack((lefttoLcuekernelsmed[a], Lcuekernelmed))
                lefttoRcuekernelsmed[a] = np.vstack((lefttoRcuekernelsmed[a], Rcuekernelmed))                
                lefttoLcuekernelslate[a] = np.vstack((lefttoLcuekernelslate[a], Lcuekernellate))
                lefttoRcuekernelslate[a] = np.vstack((lefttoRcuekernelslate[a], Rcuekernellate)) 
                
                performance[a].append(model.score(X, yd)) #get r^2
            

              
    for a in alphalist:          
        lefttoLcuekernelsearly[a] = lefttoLcuekernelsearly[a][1:]
        lefttoRcuekernelsearly[a] = lefttoRcuekernelsearly[a][1:]
        lefttoLcuekernelsmed[a] = lefttoLcuekernelsmed[a][1:]
        lefttoRcuekernelsmed[a] = lefttoRcuekernelsmed[a][1:]
        lefttoLcuekernelslate[a] = lefttoLcuekernelslate[a][1:]
        lefttoRcuekernelslate[a] = lefttoRcuekernelslate[a][1:]

        
        perf = np.mean(performance[a])

        if not demo:        
            plt.figure()
            plt.errorbar(np.arange(0, 300, 5), np.nanmean(lefttoLcuekernelsearly[a], axis=0), yerr = sem(lefttoLcuekernelsearly[a], axis=0, nan_policy='omit'), color = 'red', alpha = .3)
            plt.errorbar(np.arange(0, 300, 5), np.nanmean(lefttoRcuekernelsearly[a], axis=0), yerr = sem(lefttoRcuekernelsearly[a], axis=0, nan_policy='omit'), color = 'blue', alpha = .3)
            plt.errorbar(np.arange(0, 200, 5), np.nanmean(lefttoLcuekernelsmed[a], axis=0), yerr = sem(lefttoLcuekernelsmed[a], axis=0, nan_policy='omit'), color = 'red', alpha = .6)
            plt.errorbar(np.arange(0, 200, 5), np.nanmean(lefttoRcuekernelsmed[a], axis=0), yerr = sem(lefttoRcuekernelsmed[a], axis=0, nan_policy='omit'), color = 'blue', alpha = .6)
            plt.errorbar(np.arange(0, 200, 5), np.nanmean(lefttoLcuekernelslate[a], axis=0), yerr = sem(lefttoLcuekernelslate[a], axis=0, nan_policy='omit'), color = 'red')
            plt.errorbar(np.arange(0, 200, 5), np.nanmean(lefttoRcuekernelslate[a], axis=0), yerr = sem(lefttoRcuekernelslate[a], axis=0, nan_policy='omit'), color = 'blue')
            plt.xlabel('distance from cue appearance (cm)')
            plt.ylabel('kernel amplitude')
            plt.xlim([0, 200])
            plt.axhline(0, color = 'k', linestyle = '--')
            plt.title(region+', alpha = '+str(a)+', r2 ='+str(perf))
            plt.show()
        else:
            plt.figure()
            plt.plot(np.arange(0, 300, 5), np.nanmean(lefttoLcuekernelsearly[a], axis=0), color = 'red', alpha = .3)
            plt.plot(np.arange(0, 300, 5), np.nanmean(lefttoRcuekernelsearly[a], axis=0), color = 'blue', alpha = .3)
            plt.plot(np.arange(0, 200, 5), np.nanmean(lefttoLcuekernelsmed[a], axis=0), color = 'red', alpha = .6)
            plt.plot(np.arange(0, 200, 5), np.nanmean(lefttoRcuekernelsmed[a], axis=0),  color = 'blue', alpha = .6)
            plt.plot(np.arange(0, 200, 5), np.nanmean(lefttoLcuekernelslate[a], axis=0),  color = 'red')
            plt.plot(np.arange(0, 200, 5), np.nanmean(lefttoRcuekernelslate[a], axis=0),  color = 'blue')
            plt.xlabel('distance from cue appearance (cm)')
            plt.ylabel('kernel amplitude')
            plt.xlim([0, 200])
            plt.axhline(0, color = 'k', linestyle = '--')
            plt.title(region+', alpha = '+str(a)+', r2 ='+str(perf))
            plt.show()                                   