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
from scipy.stats import f_oneway, kurtosis, skew, zscore, sem, f
import pickle
import os
from scipy.optimize import curve_fit
import diptest
import pandas as pd


sigthresh = .05
pthresh = .05
region_threshold = .25 #.5 Ryan
region_width = 4
base_thresh = 0 #3 Ryan

def get_cum_evidence(lefts, rights, pos):
    cum_ev = np.zeros(len(pos))
    for i, p in enumerate(pos):
        cum_ev[i] = np.sum(lefts<p)-np.sum(rights<p)
    return -1*cum_ev

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

regions = ['ACC', 'RSC']

files = os.listdir('./DMS')
DMSmatfiles = [f for f in files if f.startswith('dFF_scott')]

files = os.listdir('./ACC/HalfGaussianFiringRate')
ACCmatfiles = [f for f in files if f.startswith('dFF_tet')]

files = os.listdir('./V1')
V1matfiles = [f for f in files if f.startswith('nic')]

files = os.listdir('./RSC/HalfGaussianFiringRate')
RSCmatfiles = [f for f in files if f.startswith('nic')]

files = os.listdir('./HPC')
HPCmatfiles = [f for f in files if f.startswith('nic')]

filelist = [ACCmatfiles, RSCmatfiles]

fitvsort = pd.DataFrame()

allLCellLChoice = np.zeros((1, 66))
allLCellRChoice = np.zeros((1, 66))

allRCellLChoice = np.zeros((1, 66))
allRCellRChoice = np.zeros((1, 66))

allLeftmups = np.array([])
allRightmups = np.array([])

with open('numtotaltrials.pkl', 'rb') as handle:
    numcorrecttrials = pickle.load(handle)

for region, matfiles in zip(regions[:], filelist[:]):
    fitparamsencoding = np.load(region+'/paramfit/'+region+'allfitparams-linearencoding-correctandincorrect.npy')
    fitneurondata = pd.read_csv(region+'/paramfit/'+region+'allfitparams-linearencoding-neuroninfo-correctandincorrect.csv')
    
    sigF = np.zeros(len(fitneurondata))
    
    for j in range(len(sigF)):
        session = fitneurondata['Session'].values[j]
        ncorrect = numcorrecttrials[session]
        sigF[j] = f.ppf(.99, 1, ncorrect-3)
    
    npo = 66
    finaldiff1to4 = np.zeros((1, npo))
    finaldiff5to9 = np.zeros((1, npo))
    finaldiffgreater9 = np.zeros((1, npo))
    negfinaldiff1to4 = np.zeros((1, npo))
    negfinaldiff5to9 = np.zeros((1, npo))
    negfinaldiffgreater9 = np.zeros((1, npo))

    absfinaldiff1to4 = np.zeros((1, npo))
    absfinaldiff5to9 = np.zeros((1, npo))
    absfinaldiffgreater9 = np.zeros((1, npo))
    
    leftfinaldiff1to4 = np.zeros((1, npo))
    leftfinaldiff5to9 = np.zeros((1, npo))
    leftfinaldiffgreater9 = np.zeros((1, npo))
    leftnegfinaldiff1to4 = np.zeros((1, npo))
    leftnegfinaldiff5to9 = np.zeros((1, npo))
    leftnegfinaldiffgreater9 = np.zeros((1, npo))
    
    rightfinaldiff1to4 = np.zeros((1, npo))
    rightfinaldiff5to9 = np.zeros((1, npo))
    rightfinaldiffgreater9 = np.zeros((1, npo))
    rightnegfinaldiff1to4 = np.zeros((1, npo))
    rightnegfinaldiff5to9 = np.zeros((1, npo))
    rightnegfinaldiffgreater9 = np.zeros((1, npo))
    
    
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
    fitparams = pd.read_csv(region+'/paramfit/'+region+'allfitparams-HG.csv')
    for file in matfiles:
        data = loadmat(region+'/HalfGaussianFiringRate/'+file)
        
        if region in ['DMS', 'ACC']:
            bfile = file.split('dFF_')[1]
            bfile = bfile.split('processedOutput-NEW.mat')[0]+'.mat'
            cuedata = loadmat(region+'/Behavior/'+bfile)
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
            
            #alldata[np.isnan(alldata)]=0
        
        session = file.split('-NEW.')[0]
        evselectiveneurons = []
        choiceselectiveneurons = []
        for n in range(n_neurons): 
            try:
                index =  np.where((fitneurondata['Session']==session) & (fitneurondata['Index']==n))[0][0]
                #pidx = np.argmin(np.abs(poses-mu))
                
                sigE = sum(fitparamsencoding[:, 3, index]>sigF[index])>0
                sigC = sum(fitparamsencoding[:, 4, index]>sigF[index])>0
                if sigE:
                    evselectiveneurons.append(n)
                else:
                    if sigC:
                        choiceselectiveneurons.append(n)
            except:
                'index did not exist'
        
        
        #evselectiveneurons = choiceselectiveneurons
        
        
        fileparams = fitparams[fitparams['Session']==session]
        pvals = fileparams['PvalMSE'].values
        mues = fileparams['Mue'].values
        mups = fileparams['Mup'].values
        sigps = fileparams['Sigp'].values
        siges = fileparams['Sige'].values
        corrs = fileparams['Correlation'].values
        nids = fileparams['Neuron'].values
        
        #vp = pvals<sigthresh
        vp = []
        for eid in evselectiveneurons:
            try:
                vp.append(np.where(nids==eid)[0][0])
            except:
                print('missing idx in gaussian fit')
            
        #get only evidence selective parameters
        #evselectiveneurons = fileparams['Neuron'].values[vp]
        mues = mues[vp]
        mups = mups[vp]
        sigps = sigps[vp]
        nids = nids[vp]
        alldata = alldata[:, nids, :]
        
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
        
        
        
        
        
        if len(leftis)>0:
            #lposdist = stats.gaussian_kde(leftmups)(pos)
            leftactivity = alldata[:, leftis, :]
            lefttotal = np.nanmean(leftactivity, axis=1)
            for i, p in enumerate(pos):
                validns = np.where(np.abs(leftmups-p)<leftsigps)[0]
                lefttotal[:, i] = np.nanmean(leftactivity[:, validns, i], axis=1)
            
            #lefttotal = np.nanmean(leftactivity, axis = 1)
            #lefttotal = lefttotal/lposdist
            
        else:
            lefttotal = np.nan*np.zeros((len(correcttrials), n_pos))
            
        if len(rightis)>0:
            #rposdist = stats.gaussian_kde(rightmups)(pos)
            rightactivity = alldata[:, rightis, :] 
            righttotal = np.nanmean(rightactivity, axis = 1)
            for i, p in enumerate(pos):
                validns = np.where(np.abs(rightmups-p)<rightsigps)[0]
                righttotal[:, i] = np.nanmean(rightactivity[:, validns, i], axis=1)            
            #righttotal = righttotal/rposdist
        else:
            righttotal = np.nan*np.zeros((len(correcttrials), n_pos))
        
        
        allposdist = stats.gaussian_kde(mups)(pos)
        diff = lefttotal-righttotal
        #diff = diff/allposdist
        lefttotal = lefttotal[:, :]-np.mean(lefttotal[:, :], axis=0)
        righttotal = righttotal[:, :]-np.mean(righttotal[:, :], axis=0)
        diff = righttotal-lefttotal
        #diff = diff[:, 6:]
        #diff = diff-diff[:, 0].reshape(-1, 1)
        


        for i, e in enumerate(evidence[:, -1]):
            if (e>0) and (e<4): #(0,5)
                finaldiff1to4 = np.vstack((finaldiff1to4, diff[i, :]))
                leftfinaldiff1to4 = np.vstack((leftfinaldiff1to4, lefttotal[i, :]))
                rightfinaldiff1to4 = np.vstack((rightfinaldiff1to4, righttotal[i, :]))
                absfinaldiff1to4 = np.vstack((absfinaldiff1to4, diff[i, :]))
            if (e>3) and (e<7):
                finaldiff5to9 = np.vstack((finaldiff5to9, diff[i, :]))
                leftfinaldiff5to9 = np.vstack((leftfinaldiff5to9, lefttotal[i, :]))
                rightfinaldiff5to9 = np.vstack((rightfinaldiff5to9, righttotal[i, :]))
                absfinaldiff5to9 = np.vstack((absfinaldiff5to9, diff[i, :]))
            if e>6: #6 if 2 bins
                finaldiffgreater9 = np.vstack((finaldiffgreater9, diff[i, :]))
                leftfinaldiffgreater9 = np.vstack((leftfinaldiffgreater9, lefttotal[i, :]))
                rightfinaldiffgreater9 = np.vstack((rightfinaldiffgreater9, righttotal[i, :]))
                absfinaldiffgreater9 = np.vstack((absfinaldiffgreater9, diff[i, :]))
            if (e<0) and (e>-4):
                negfinaldiff1to4 = np.vstack((negfinaldiff1to4, diff[i, :]))
                leftnegfinaldiff1to4 = np.vstack((leftnegfinaldiff1to4, lefttotal[i, :]))
                rightnegfinaldiff1to4 = np.vstack((rightnegfinaldiff1to4, righttotal[i, :]))
                absfinaldiff1to4 = np.vstack((absfinaldiff1to4, -1*diff[i, :]))
            if (e<-3) and (e>-7):
                negfinaldiff5to9 = np.vstack((negfinaldiff5to9, diff[i, :]))
                leftnegfinaldiff5to9 = np.vstack((leftnegfinaldiff5to9, lefttotal[i, :]))
                rightnegfinaldiff5to9 = np.vstack((rightnegfinaldiff5to9, righttotal[i, :]))
                absfinaldiff5to9 = np.vstack((absfinaldiff5to9, -1*diff[i, :]))
            if e<-6: #-6 if two bins
                negfinaldiffgreater9 = np.vstack((negfinaldiffgreater9, diff[i, :]))
                leftnegfinaldiffgreater9 = np.vstack((leftnegfinaldiffgreater9, lefttotal[i, :])) 
                rightnegfinaldiffgreater9 = np.vstack((rightnegfinaldiffgreater9, righttotal[i, :])) 
                absfinaldiffgreater9 = np.vstack((absfinaldiffgreater9, -1*diff[i, :]))
                
    finaldiff1to4 = finaldiff1to4[1:]
    finaldiff5to9 = finaldiff5to9[1:]
    finaldiffgrater9 = finaldiffgreater9[1:]
    negfinaldiff1to4 = negfinaldiff1to4[1:]
    negfinaldiff5to9 = negfinaldiff5to9[1:]
    negfinaldiffgrater9 = negfinaldiffgreater9[1:]
    absfinaldiff1to4 = finaldiff1to4[1:]
    absfinaldiff5to9 = finaldiff5to9[1:]
    absfinaldiffgrater9 = finaldiffgreater9[1:]
    
    plt.figure()
    plt.plot(pos, finaldiff1to4[:2].T, color = 'r', alpha = .33)
    plt.plot(pos, finaldiff5to9[:2].T, color = 'r', alpha = .67)
    plt.plot(pos, finaldiffgreater9[:2].T, color = 'r')
    plt.plot(pos, negfinaldiff1to4[:2].T, color = 'b', alpha = .33)
    plt.plot(pos, negfinaldiff5to9[:2].T, color = 'b', alpha = .67)
    plt.plot(pos, negfinaldiffgreater9[:2].T, color = 'b')   
    plt.title(region)
    

    plt.figure()
    plt.errorbar(pos, np.nanmean(finaldiff1to4, axis=0), yerr = sem(finaldiff1to4, axis=0, nan_policy = 'omit'), color = 'r', alpha = .33)
    plt.errorbar(pos, np.nanmean(finaldiff5to9, axis=0), yerr = sem(finaldiff5to9, axis=0, nan_policy = 'omit'), color = 'r', alpha = .67)
    plt.errorbar(pos, np.nanmean(finaldiffgreater9, axis=0), yerr = sem(finaldiffgreater9, axis=0, nan_policy = 'omit'), color = 'r', alpha = 1)
    plt.errorbar(pos, np.nanmean(negfinaldiff1to4, axis=0), yerr = sem(negfinaldiff1to4, axis=0, nan_policy = 'omit'), color = 'b', alpha = .33)
    plt.errorbar(pos, np.nanmean(negfinaldiff5to9, axis=0), yerr = sem(negfinaldiff5to9, axis=0, nan_policy = 'omit'), color = 'b', alpha = .67)
    plt.errorbar(pos, np.nanmean(negfinaldiffgreater9, axis=0), yerr = sem(negfinaldiffgreater9, axis=0, nan_policy = 'omit'), color = 'b', alpha = 1)
    plt.title(region)
    #plt.xticks([0, 20, 40, 49.75, 59.5], labels=['0', 'cues', '200', 'delay', '300'])
    #plt.xticks([0, 5.5, 25.5, 45.5, 55.25, 65], labels=['-30', '0', 'cues', '200', 'delay', '300'])
    plt.ylabel('Activity (R-L)')
    plt.savefig('Figure4Plots/Ramping/'+region+'correctincorrect-evcells-trueevidence-HG.pdf')
        
    plt.figure()
    plt.errorbar(pos, np.nanmean(absfinaldiff1to4, axis=0), yerr = sem(absfinaldiff1to4, axis=0, nan_policy = 'omit'), color = 'k', alpha = .33)
    plt.errorbar(pos, np.nanmean(absfinaldiff5to9, axis=0), yerr = sem(absfinaldiff5to9, axis=0, nan_policy = 'omit'), color = 'k', alpha = .67)
    plt.errorbar(pos, np.nanmean(absfinaldiffgreater9, axis=0), yerr = sem(absfinaldiffgreater9, axis=0, nan_policy = 'omit'), color = 'k', alpha = 1)
    plt.title(region)
    #plt.xticks([0, 20, 40, 49.75, 59.5], labels=['0', 'cues', '200', 'delay', '300'])
    #plt.xticks([0, 5.5, 25.5, 45.5, 55.25, 65], labels=['-30', '0', 'cues', '200', 'delay', '300'])
    plt.ylabel('Activity (Pref.- Non. Pref.)')
    plt.ylim([-.1, .75])
    #plt.savefig('Figure4Plots/Ramping/'+region+'correctincorrect-evcells.pdf')
    
    plt.figure()
    plt.plot(pos, np.nanmean(leftfinaldiff1to4, axis=0), color = 'r', alpha = .33)
    plt.plot(pos, np.nanmean(leftfinaldiff5to9, axis=0), color = 'r', alpha = .67)
    plt.plot(pos, np.nanmean(leftfinaldiffgreater9, axis=0), color = 'r', alpha = 1)
    plt.plot(pos, np.nanmean(leftnegfinaldiff1to4, axis=0), color = 'b', alpha = .33)
    plt.plot(pos, np.nanmean(leftnegfinaldiff5to9, axis=0), color = 'b', alpha = .67)
    plt.plot(pos, np.nanmean(leftnegfinaldiffgreater9, axis=0), color = 'b', alpha = 1)
    plt.title(region)
    #plt.xticks([0, 20, 40, 49.75, 59.5], labels=['0', 'cues', '200', 'delay', '300'])
    #plt.xticks([0, 5.5, 25.5, 45.5, 55.25, 65], labels=['-30', '0', 'cues', '200', 'delay', '300'])
    plt.ylabel('Left Activity')
    
    plt.figure()
    plt.plot(pos, np.nanmean(rightfinaldiff1to4, axis=0), color = 'r', alpha = .33)
    plt.plot(pos, np.nanmean(rightfinaldiff5to9, axis=0), color = 'r', alpha = .67)
    plt.plot(pos, np.nanmean(rightfinaldiffgreater9, axis=0), color = 'r', alpha = 1)
    plt.plot(pos, np.nanmean(rightnegfinaldiff1to4, axis=0), color = 'b', alpha = .33)
    plt.plot(pos, np.nanmean(rightnegfinaldiff5to9, axis=0), color = 'b', alpha = .67)
    plt.plot(pos, np.nanmean(rightnegfinaldiffgreater9, axis=0), color = 'b', alpha = 1)
    plt.title(region)
    #plt.xticks([0, 20, 40, 49.75, 59.5], labels=['0', 'cues', '200', 'delay', '300'])
    #plt.xticks([0, 5.5, 25.5, 45.5, 55.25, 65], labels=['-30', '0', 'cues', '200', 'delay', '300'])
    plt.ylabel('Right Activity')
 
