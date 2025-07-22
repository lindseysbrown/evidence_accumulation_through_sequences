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
from sklearn.linear_model import LinearRegression, Ridge
from scipy.stats import sem, zscore

sigthresh = .05


def get_cum_evidence(lefts, rights, pos):
    cum_ev = np.zeros(len(pos))
    for i, p in enumerate(pos):
        cum_ev[i] = np.sum(lefts<p)-np.sum(rights<p)
    return cum_ev

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

#get cubic spline basis set
these_knots = np.linspace(0,300, 4)

x = np.linspace(0., 300, 60)
y = patsy.bs(x, df=7, degree=3, include_intercept=True)
plt.subplot(1,2,1)
plt.plot(x,y)
plt.title('B-spline basis')

y2 = patsy.bs(np.linspace(0, 230, 46), df=7, degree=3, include_intercept=True)

y3 = patsy.bs(np.linspace(0, 160, 32), df = 7, degree = 3, include_intercept=True)

#cuelocs = np.zeros(66,)
#cuelocs[5] = 1
#cuelocs[8] = 1

#for i in range(7):
 #   out = convolve(cuelocs, y[:, i], mode = 'full')
  #  plt.figure()
   # plt.plot(out)
    #plt.axvline(5)
    #plt.axvline(8)

regions = ['ACC', 'RSC'] #['ACC', 'DMS', 'HPC', 'RSC']

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

filelist = [ACCmatfiles, RSCmatfiles] #[ACCmatfiles, DMSmatfiles,  HPCmatfiles,  RSCmatfiles]



alphalist = [1] #[.001, 1, 100]

with open('numcorrecttrials.pkl', 'rb') as handle:
    numcorrecttrials = pickle.load(handle)

for region, matfiles in zip(regions, filelist):
    fitparams = pd.read_csv(region+'/paramfit/'+region+'allfitparams-boundedmue-MSE.csv')
    fits = fitparams['MSE'].values[fitparams['PvalMSE'].values < .05]
    lefttoLcuekernelsearly = {}
    lefttoRcuekernelsearly = {}#np.zeros((1, 60))
    lefttoLcuekernelsmed = {}
    lefttoRcuekernelsmed = {}
    righttoLcuekernelsearly = {}#np.zeros((1, 60))
    righttoRcuekernelsearly = {}#np.zeros((1, 60))
    lefttoLcuekernelslate = {}#np.zeros((1, 60))
    lefttoRcuekernelslate = {}#np.zeros((1, 60))
    righttoLcuekernelslate = {}#np.zeros((1, 60))
    righttoRcuekernelslate = {}#np.zeros((1, 60)) 
    performance = {}
    
    for a in alphalist:
        lefttoLcuekernelsearly[a] = np.zeros((1, 60))
        lefttoRcuekernelsearly[a] = np.zeros((1, 60))
        righttoLcuekernelsearly[a] = np.zeros((1, 60))
        righttoRcuekernelsearly[a] = np.zeros((1, 60))
        lefttoLcuekernelsmed[a] = np.zeros((1, 46))
        lefttoRcuekernelsmed[a] = np.zeros((1,46))
        lefttoLcuekernelslate[a] = np.zeros((1, 32))
        lefttoRcuekernelslate[a] = np.zeros((1, 32))
        righttoLcuekernelslate[a] = np.zeros((1, 32))
        righttoRcuekernelslate[a] = np.zeros((1, 32)) 
        performance[a] = []
    
    for f, file in enumerate(matfiles):
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
                cuebin = np.where((pos-l)>0)[0][0]-1 #previously np.argmin(np.abs(pos-l))
                eatcue = evidence[idx, cuebin-2]
                if l<70:
                    Lcuelocsearly[cuebin-2] = 1 #locked to cue appearance
                elif l<140:
                   Lcuelocsmed[cuebin-2] = 1 
                else:
                    Lcuelocslate[cuebin-2] = 1
            for r in rcues:
                cuebin = np.where((pos-r)>0)[0][0]-1
                eatcue = evidence[idx, cuebin-2]
                if r<70:
                    Rcuelocsearly[cuebin-2] = 1 #locked to cue appearance
                elif r<140:
                    Rcuelocsmed[cuebin-2] = 1
                else:
                    Rcuelocslate[cuebin-2] = 1
            for spl in range(7):
                zLearly = convolve(Lcuelocsearly[5:], y[:, spl], mode = 'full')[:(n_pos-5)]#[5:n_pos]
                zRearly = convolve(Rcuelocsearly[5:], y[:, spl], mode = 'full')[:(n_pos-5)]#[5:n_pos]
                zLmed = convolve(Lcuelocsmed[5:], y2[:, spl], mode = 'full')[:(n_pos-5)]#[5:n_pos]
                zRmed = convolve(Rcuelocsmed[5:], y2[:, spl], mode = 'full')[:(n_pos-5)]#[5:n_pos]
                zLlate = convolve(Lcuelocslate[5:], y3[:, spl], mode = 'full')[:(n_pos-5)]#[5:n_pos]
                zRlate = convolve(Rcuelocslate[5:], y3[:, spl], mode = 'full')[:(n_pos-5)]#[5:n_pos]
                feats = np.concatenate((feats, zLearly.reshape(-1, 1), zRearly.reshape(-1, 1), zLmed.reshape(-1,1), zRmed.reshape(-1,1), zLlate.reshape(-1, 1), zRlate.reshape(-1, 1)), axis = 1)
            X[idx*(n_pos-5):(idx+1)*(n_pos-5), :] = feats
        
        if len(leftis)>0 and len(rightis)>0:
            leftactivity = alldata[:, leftis, :]
            lefttotal = np.nanmean(leftactivity, axis=1)
            for i, p in enumerate(pos):
                validns = np.where(np.abs(leftmups-p)<leftsigps)[0]
                lefttotal[:, i] = np.nanmean(leftactivity[:, validns, i], axis=1)
            #lefttotal = lefttotal-np.nanmean(lefttotal, axis=0)
            #muL = np.nanmean(lefttotal[:, :6], axis=1)
            #sigL = np.nanstd(lefttotal[:, :6], axis = 1)
            #lefttotal = lefttotal - muL.reshape(-1, 1)
            #lefttotal = lefttotal/sigL.reshape(-1, 1)
            lefttotal = lefttotal[:, 5:]
            #lefttotal[lchoices] = lefttotal[lchoices]-np.nanmean(lefttotal[lchoices], axis=0)
            #lefttotal[rchoices] = lefttotal[rchoices]-np.nanmean(lefttotal[rchoices], axis=0)
            
            rightactivity = alldata[:, rightis, :] 
            righttotal = np.nanmean(rightactivity, axis = 1)
            for i, p in enumerate(pos):
                validns = np.where(np.abs(rightmups-p)<rightsigps)[0]
                righttotal[:, i] = np.nanmean(rightactivity[:, validns, i], axis=1) 
            #righttotal = righttotal-np.nanmean(righttotal, axis=0)
            #muR = np.nanmean(righttotal[:, :6], axis=1)
            #sigR = np.nanstd(righttotal[:, :6], axis = 1)
            #righttotal = righttotal - muR.reshape(-1, 1)
            #righttotal = righttotal/sigR.reshape(-1,1)
            righttotal = righttotal[:, 5:]
            #righttotal[lchoices] = righttotal[lchoices]-np.nanmean(righttotal[lchoices], axis=0)
            #righttotal[rchoices] = righttotal[rchoices]-np.nanmean(righttotal[rchoices], axis=0)
            
            difftotal = (lefttotal - np.nanmean(lefttotal, axis=0))-(righttotal-np.nanmean(righttotal, axis=0))
            
            #difftotal = difftotal - np.nanmean(difftotal, axis=0)
            #difftotal = lefttotal-righttotal

            #difftotal = difftotal-difftotal[:,0].reshape(-1, 1)
            yd = difftotal.flatten()            
            

            
            X = X[~np.isnan(yd)]
            yd = yd[~np.isnan(yd)]
            
            for a in alphalist:
                model = Ridge(alpha=a)
                model.fit(X, yd)
                coefs = model.coef_
                
                ypred = model.predict(X)
                
                '''
                if f<4:
                    plt.figure()
                    for ex in range(5):
                        efinal = evidence[ex, -1]
                        if efinal<0:
                            c = 'blue'
                        else:
                            c = 'red'
                        alph = np.abs(efinal)/8
                        plt.plot(yd[ex*60:(ex+1)*60], color = c, alpha = alph)
                        plt.plot(ypred[ex*60:(ex+1)*60], linestyle='--', color = c, alpha =alph)
                        plt.title(region)
                        
                    plt.figure()
                    plt.plot(yd[:300], color = 'black')
                    plt.plot(ypred[:300], linestyle = '--', color = 'black')
                    plt.axvline(60, color = 'grey', linestyle = '--')
                    plt.axvline(120, color = 'grey', linestyle = '--')
                    plt.axvline(180, color = 'grey', linestyle = '--')
                    plt.axvline(240, color = 'grey', linestyle = '--')
                    plt.title(region)
                '''
                    
                    
            
                Lcuekernelearly = y[:, :7]@coefs[2:44:6] 
                Rcuekernelearly = y[:, :7]@coefs[3:44:6]
                Lcuekernelmed = y2[:, :7]@coefs[4:44:6]
                Rcuekernelmed = y2[:, :7]@coefs[5:44:6]
                Lcuekernellate = y3[:, :7]@coefs[6:44:6]
                Rcuekernellate = y3[:, :7]@coefs[7:44:6]                
        
                lefttoLcuekernelsearly[a] = np.vstack((lefttoLcuekernelsearly[a], Lcuekernelearly))
                lefttoRcuekernelsearly[a] = np.vstack((lefttoRcuekernelsearly[a], Rcuekernelearly))
                lefttoLcuekernelsmed[a] = np.vstack((lefttoLcuekernelsmed[a], Lcuekernelmed))
                lefttoRcuekernelsmed[a] = np.vstack((lefttoRcuekernelsmed[a], Rcuekernelmed))                
                lefttoLcuekernelslate[a] = np.vstack((lefttoLcuekernelslate[a], Lcuekernellate))
                lefttoRcuekernelslate[a] = np.vstack((lefttoRcuekernelslate[a], Rcuekernellate)) 
                
                performance[a].append(model.score(X, yd))
            

              
    for a in alphalist:          
        lefttoLcuekernelsearly[a] = lefttoLcuekernelsearly[a][1:]
        lefttoRcuekernelsearly[a] = lefttoRcuekernelsearly[a][1:]
        lefttoLcuekernelsmed[a] = lefttoLcuekernelsmed[a][1:]
        lefttoRcuekernelsmed[a] = lefttoRcuekernelsmed[a][1:]
        lefttoLcuekernelslate[a] = lefttoLcuekernelslate[a][1:]
        lefttoRcuekernelslate[a] = lefttoRcuekernelslate[a][1:]

        
        perf = np.mean(performance[a])
        
        '''
        plt.figure()
        plt.errorbar(np.arange(0, 300, 5), np.nanmean(lefttoLcuekernelsearly[a], axis=0), yerr = sem(lefttoLcuekernelsearly[a], axis=0, nan_policy='omit'), color = 'red')
        plt.errorbar(np.arange(0, 300, 5), np.nanmean(lefttoRcuekernelsearly[a], axis=0), yerr = sem(lefttoRcuekernelsearly[a], axis=0, nan_policy='omit'), color = 'blue')
        #plt.errorbar(np.arange(0, 300, 5), np.nanmean(righttoLcuekernelsearly, axis=0), yerr = sem(righttoLcuekernelsearly, axis=0, nan_policy='omit'), color = 'blue', alpha=.5)
        #plt.errorbar(np.arange(0, 300, 5), np.nanmean(righttoRcuekernelsearly, axis=0), yerr = sem(righttoRcuekernelsearly, axis=0, nan_policy='omit'), color = 'blue')
        plt.xlabel('distance from cue appearance (cm)')
        plt.ylabel('kernel amplitude')
        plt.ylim([-.1, .1])
        plt.axhline(0, color = 'k', linestyle = '--')
        plt.title(region+'- low ev. cues'+', alpha = '+str(a)+', r2 ='+str(perf))
        
        plt.figure()
        plt.errorbar(np.arange(0, 200, 5), np.nanmean(lefttoLcuekernelsmed[a], axis=0), yerr = sem(lefttoLcuekernelsmed[a], axis=0, nan_policy='omit'), color = 'red')
        plt.errorbar(np.arange(0, 200, 5), np.nanmean(lefttoRcuekernelsmed[a], axis=0), yerr = sem(lefttoRcuekernelsmed[a], axis=0, nan_policy='omit'), color = 'blue')
        #plt.errorbar(np.arange(0, 300, 5), np.nanmean(righttoLcuekernelsearly, axis=0), yerr = sem(righttoLcuekernelsearly, axis=0, nan_policy='omit'), color = 'blue', alpha=.5)
        #plt.errorbar(np.arange(0, 300, 5), np.nanmean(righttoRcuekernelsearly, axis=0), yerr = sem(righttoRcuekernelsearly, axis=0, nan_policy='omit'), color = 'blue')
        plt.xlabel('distance from cue appearance (cm)')
        plt.ylabel('kernel amplitude')
        plt.ylim([-.1, .1])
        plt.axhline(0, color = 'k', linestyle = '--')
        plt.title(region+'- med ev. cues'+', alpha = '+str(a)+', r2 ='+str(perf))
    
        plt.figure()
        plt.errorbar(np.arange(0, 200, 5), np.nanmean(lefttoLcuekernelslate[a], axis=0), yerr = sem(lefttoLcuekernelslate[a], axis=0, nan_policy='omit'), color = 'red')
        plt.errorbar(np.arange(0, 200, 5), np.nanmean(lefttoRcuekernelslate[a], axis=0), yerr = sem(lefttoRcuekernelslate[a], axis=0, nan_policy='omit'), color = 'blue')
        #plt.errorbar(np.arange(0, 300, 5), np.nanmean(righttoLcuekernelslate, axis=0), yerr = sem(righttoLcuekernelslate, axis=0, nan_policy='omit'), color = 'blue', alpha=.5)
        #plt.errorbar(np.arange(0, 300, 5), np.nanmean(righttoRcuekernelslate, axis=0), yerr = sem(righttoRcuekernelslate, axis=0, nan_policy='omit'), color = 'blue')
        plt.xlabel('distance from cue appearance (cm)')
        plt.ylabel('kernel amplitude')
        plt.ylim([-.1, .1])
        plt.axhline(0, color = 'k', linestyle = '--')
        plt.title(region+'- high ev. cues'+', alpha = '+str(a)+', r2 ='+str(perf))
        '''
        
        plt.figure()
        plt.errorbar(np.arange(0, 300, 5), np.nanmean(lefttoLcuekernelsearly[a], axis=0), yerr = sem(lefttoLcuekernelsearly[a], axis=0, nan_policy='omit'), color = 'red', alpha = .3)
        plt.errorbar(np.arange(0, 300, 5), np.nanmean(lefttoRcuekernelsearly[a], axis=0), yerr = sem(lefttoRcuekernelsearly[a], axis=0, nan_policy='omit'), color = 'blue', alpha = .3)
        plt.errorbar(np.arange(0, 230, 5), np.nanmean(lefttoLcuekernelsmed[a], axis=0), yerr = sem(lefttoLcuekernelsmed[a], axis=0, nan_policy='omit'), color = 'red', alpha = .6)
        plt.errorbar(np.arange(0, 230, 5), np.nanmean(lefttoRcuekernelsmed[a], axis=0), yerr = sem(lefttoRcuekernelsmed[a], axis=0, nan_policy='omit'), color = 'blue', alpha = .6)
        plt.errorbar(np.arange(0, 160, 5), np.nanmean(lefttoLcuekernelslate[a], axis=0), yerr = sem(lefttoLcuekernelslate[a], axis=0, nan_policy='omit'), color = 'red')
        plt.errorbar(np.arange(0, 160, 5), np.nanmean(lefttoRcuekernelslate[a], axis=0), yerr = sem(lefttoRcuekernelslate[a], axis=0, nan_policy='omit'), color = 'blue')
        #plt.axvline(0, linestyle = '--', color = 'k')
        #plt.axvline(10, linestyle = '--', color = 'grey')
        plt.xlabel('distance from cue appearance (cm)')
        plt.ylabel('kernel amplitude')
        #plt.ylim([-.1, .1])
        plt.xlim([0, 200])
        plt.axhline(0, color = 'k', linestyle = '--')
        plt.title(region+', alpha = '+str(a)+', r2 ='+str(perf))
        plt.savefig('Figure4Plots/CueResponse/'+region+'3PosCueKernels.pdf')                         