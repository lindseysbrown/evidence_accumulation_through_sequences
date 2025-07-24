# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 10:19:35 2023

@author: lindseyb
"""

import numpy as np
from scipy.io import loadmat
from sklearn.linear_model import RidgeCV
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
import os
from scipy.stats import sem, pearsonr
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import cross_validate, cross_val_predict
import sklearn

#demo
demo = True #True to run with example datasets, False to run for full data in structured folders

alpha_list = [.0001, .001, .01, .1, 1, 10, 100, 1000]

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

#pearson correlation coefficient as metric to evaluate decoder
def pearsonronly(X, y):
    return pearsonr(X, y)[0]
corrmetric = sklearn.metrics.make_scorer(pearsonronly)    
    
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

for region, matfiles in zip(regions, filelist):
    #load parameters from joint gaussian fit
    if not demo:
        fitparams = pd.read_csv(region+'/paramfit/'+region+'allfitparams.csv')
    else:
        fitparams = pd.read_csv('ExampleData/ACCparamfitexample.csv')

    shufflereps = 5
    
    decodingleftr2 = np.zeros((len(matfiles), 66))*np.nan
    decodingleftr = np.zeros((len(matfiles), 66))*np.nan
    decodingleftcorr = np.zeros((len(matfiles), 66))*np.nan
    bimodalleft = np.zeros((len(matfiles), 66))*np.nan
    bimodallefttrue = np.zeros((len(matfiles), 66))*np.nan
    decodingleftcorrS = np.zeros((shufflereps*len(matfiles), 66))*np.nan
    decodingleftr2S = np.zeros((shufflereps*len(matfiles), 66))*np.nan
    decodingleftrS = np.zeros((shufflereps*len(matfiles), 66))*np.nan
    
    decodingrightr2 = np.zeros((len(matfiles), 66))*np.nan
    decodingrightr = np.zeros((len(matfiles), 66))*np.nan
    decodingrightcorr = np.zeros((len(matfiles), 66))*np.nan
    bimodalright = np.zeros((len(matfiles), 66))*np.nan
    bimodalrighttrue = np.zeros((len(matfiles), 66))*np.nan
    decodingrightcorrS = np.zeros((shufflereps*len(matfiles), 66))*np.nan
    decodingrightr2S = np.zeros((shufflereps*len(matfiles), 66))*np.nan
    decodingrightrS = np.zeros((shufflereps*len(matfiles), 66))*np.nan
    
    for fi, file in enumerate(matfiles):
        if not demo:
            session = file.split('.')[0]
        else:
            session = 'dFF_tetO_8_07282021_T10processedOutput'

        if not demo:
            data = loadmat(region+'/'+file)
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
            n_neurons = np.shape(data['out']['FR2_pos'][0][0])[1]
            [n_trials, n_pos] = np.shape(data['out']['FR2_pos'][0][0][0][0]) 
            
            alldata = np.zeros((n_trials, n_neurons, n_pos))
            
            maintrials = data['out']['Trial_Main_Maze'][0][0][0]-1 #subtract one for difference in matlab indexing
            correcttrials = data['out']['correct'][0][0][0]-1
            
            #get data normalized by position
            for i in range(n_neurons):
                alldata[:, i, :] = data['out']['FR2_pos'][0][0][0][i]
            
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
                            
        evidence = np.zeros((n_trials, n_pos))
        for i in range(n_trials):
            evidence[i] = get_cum_evidence(lefts[i], rights[i], pos)
        evidence = evidence[correcttrials, :]
        
        mups = np.zeros(n_neurons)
        sigps = np.zeros(n_neurons)
        evsel = np.zeros(n_neurons)
        for n in range(n_neurons):
            try:
                mups[n] = fitparams[(fitparams.Neuron==n) & (fitparams.Session==session)]['Mup'].iloc[0]
                sigps[n] = fitparams[(fitparams.Neuron==n) & (fitparams.Session==session)]['Sigp'].iloc[0]
                evsel[n] = fitparams[(fitparams.Neuron==n) & (fitparams.Session==session)]['PvalMSE'].iloc[0]<.05
            except:
                mups[n] = np.nan
                sigps[n] = np.nan
                evsel[n] = False
        
        #restrict to correct trials
        if len(correcttrials)>10:
            choices = np.zeros(n_trials)
            choices[lchoices] = 1
            choices = choices[correcttrials]
            alldata = alldata[correcttrials, :, :]
            lchoices = np.where(choices==1)[0]
            rchoices = np.where(choices==0)[0]
            
            for p in range(6, n_pos):
                validnsbyp = (np.abs(mups-p)<sigps)*1
                validns = validnsbyp
                validns = np.where(validns)[0]

                
                evatp = evidence[:, p]
                
                if len(validns)>0: #define left and right splits but with zero excluded
                    XL = alldata[:, validns, p]
                    XL = XL[evatp>0, :]
                    yL = evidence[evatp>0, p]
                    
                    
                    XR = alldata[:, validns, p]
                    XR = XR[evatp<0, :]
                    yR = evidence[evatp<0, p]
                    
                    #remove large evidence values
                    XL = XL[np.abs(yL)<20, :]
                    yL = yL[np.abs(yL)<20]
                    XR = XR[np.abs(yR)<20, :]
                    yR = yR[np.abs(yR)<20]
                
                    imputer = KNNImputer() #impute if missing values


                    #nested cross-validation, but ensure with no zeros that there are enough trials for splits
                    if len(yL)>10:
                        XL = imputer.fit_transform(XL)
                        Lridge = RidgeCV(alphas = alpha_list, cv=5)
                        left_results = cross_validate(Lridge, XL, yL, scoring = corrmetric)
                        predyL = cross_val_predict(Lridge, XL, yL)
                        decodingleftcorr[fi, p] = np.nanmean(left_results['test_score']) 
                

                    
                    if len(yR)>10:
                        XR = imputer.fit_transform(XR)                                
                        Rridge = RidgeCV(alphas = alpha_list, cv=5)
                        right_results = cross_validate(Rridge, XR, yR, scoring = corrmetric)
                        predyR = cross_val_predict(Rridge, XR, yR)
                        decodingrightcorr[fi, p] = np.nanmean(right_results['test_score'])
                                


                    #get performance on shuffle    
                    if len(yL)>10:
                        ySL = yL.copy()
    
                        for j in range(shufflereps):
                            np.random.shuffle(ySL)                      
                            
                            LSridge = RidgeCV(alphas = alpha_list, cv=5)
                            leftshuffle = cross_validate(LSridge, XL, ySL, scoring = corrmetric)
                            decodingleftcorrS[fi*shufflereps+j, p] = np.nanmean(leftshuffle['test_score'])
                            

                    if len(yR)>10:
                        ySR = yR.copy()
    
                        for j in range(shufflereps):
                    
                            np.random.shuffle(ySR)
                            
                            RSridge = RidgeCV(alphas = alpha_list, cv=5)
                            rightshuffle = cross_validate(RSridge, XR, ySR, scoring = corrmetric)
                            decodingrightcorrS[fi*shufflereps+j,p] = np.nanmean(rightshuffle['test_score'])

                else:
                    decodingleftr[fi, p] = np.nan
                    decodingleftr2[fi,p] = np.nan
                    decodingrightr[fi, p] = np.nan
                    decodingrightr2[fi, p] = np.nan
                    
                    for j in range(shufflereps):
                        decodingleftrS[fi*shufflereps+j, p] = np.nan
                        decodingleftr2S[fi*shufflereps+j, p] = np.nan
                        decodingrightrS[fi*shufflereps+j, p] = np.nan
                        decodingrightr2S[fi*shufflereps+j, p] = np.nan
                        decodingrightcorrS[fi*shufflereps+j, p] = np.nan                    
               
                    
        else:
            decodingleftr[fi, :] = np.nan
            decodingleftr2[fi,:] = np.nan
            decodingrightr[fi, :] = np.nan
            decodingrightr2[fi, :] = np.nan
            
            for j in range(shufflereps):
                decodingleftrS[fi*shufflereps+j, :] = np.nan
                decodingleftr2S[fi*shufflereps+j, :] = np.nan
                decodingrightrS[fi*shufflereps+j, :] = np.nan
                decodingrightr2S[fi*shufflereps+j, :] = np.nan
                decodingrightcorrS[fi*shufflereps+j, :] = np.nan
        

    plt.figure()
    plt.plot(pos, np.nanmean(decodingleftcorr, axis=0), color='b')
    plt.fill_between(pos, np.nanmean(decodingleftcorr, axis=0)-sem(decodingleftcorr, axis=0, nan_policy='omit'), np.nanmean(decodingleftcorr, axis=0)+sem(decodingleftcorr, axis=0, nan_policy='omit'), alpha = .5, color = 'b')
    plt.plot(pos, np.nanmean(decodingrightcorr, axis=0), color='r')
    plt.fill_between(pos, np.nanmean(decodingrightcorr, axis=0)-sem(decodingrightcorr, axis=0, nan_policy='omit'), np.nanmean(decodingrightcorr, axis=0)+sem(decodingrightcorr, axis=0, nan_policy='omit'), alpha = .5, color = 'r')
    plt.plot(pos, np.nanmean(decodingrightcorrS, axis=0), color='grey', linestyle=':')
    plt.fill_between(pos, np.nanmean(decodingrightcorrS, axis=0)-sem(decodingrightcorrS, axis=0, nan_policy='omit'), np.nanmean(decodingrightcorrS, axis=0)+sem(decodingrightcorrS, axis=0, nan_policy='omit'), alpha = .5, color = 'grey')
    plt.plot(pos, np.nanmean(decodingleftcorrS, axis=0), color='grey', linestyle='--')
    plt.fill_between(pos, np.nanmean(decodingleftcorrS, axis=0)-sem(decodingleftcorrS, axis=0, nan_policy='omit'), np.nanmean(decodingleftcorrS, axis=0)+sem(decodingleftcorrS, axis=0, nan_policy='omit'), alpha = .5, color = 'grey')    
    plt.ylabel('Pearson Correlation (r)')
    plt.xlabel('Position') 
    plt.xlim([0, 300])
    plt.title(region)
    plt.ylim([-.25, .45])
    plt.axhline(0, color = 'k', linestyle = '--')
    plt.show()  

       
