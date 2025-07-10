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
from scipy.stats import sem, zscore, pearsonr

sigthresh = .05

def mse(x, y):
    return sum((x-y)**2)

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
plt.savefig('Figure4Plots/CueResponse/Splines.pdf')

y2 = patsy.bs(np.linspace(0, 200, 40), df=7, degree=3, include_intercept=True)

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

colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']

with open('numcorrecttrials.pkl', 'rb') as handle:
    numcorrecttrials = pickle.load(handle)

numplot = 0
toplot = False

for region, matfiles in zip(regions, filelist):
    fitparams = pd.read_csv(region+'/paramfit/'+region+'allfitparams-boundedmue-MSE.csv')
    fits = fitparams['MSE'].values[fitparams['PvalMSE'].values < .05]
    lefttoLcuekernels = {}
    lefttoRcuekernels = {}#np.zeros((1, 60))
    performance = {}
    
    for a in alphalist:
        lefttoLcuekernels[a] = np.zeros((1, 60))
        lefttoRcuekernels[a] = np.zeros((1, 60))
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
        X = np.zeros((len(correcttrials)*(n_pos-5), 16))
        
        
        for idx, t in enumerate(correcttrials):
            ps = pos[5:]
            es = 0*np.sign(evidence[idx, 5:])
            if region in ['ACC', 'DMS']:
                lcues = lefts[t][0]
                rcues = rights[t][0]
            else:
                lcues = lefts[t]
                rcues = rights[t]
            
            if (numplot<1) and len(lcues)>0 and len(rcues)>0:
                toplot = True
            
            Lcuelocs = np.zeros(n_pos,)
            Rcuelocs = np.zeros(n_pos,)
            feats = np.concatenate((0*ps.reshape(-1, 1), es.reshape(-1, 1)), axis = 1)
            
            if toplot:
                plt.figure()
                for l in lcues:
                    plt.axvline(l-10, color = 'b')
                plt.xlim([0, 300])
                plt.savefig('Figure4Plots/CueResponse/ExampleLefts.pdf')
                
                plt.figure()
                for r in rcues:
                    plt.axvline(r-10, color = 'r')
                plt.xlim([0, 300])
                plt.savefig('Figure4Plots/CueResponse/ExampleRights.pdf')
            
            for l in lcues:
                cuebin = np.where((pos-l)>0)[0][0]-1
                Lcuelocs[cuebin-2] = 1 #locked to cue appearance
            for r in rcues:
                cuebin = np.where((pos-r)>0)[0][0]-1
                Rcuelocs[cuebin-2] = 1
            for spl in range(7):
                zL = convolve(Lcuelocs[5:], y[:, spl], mode = 'full')[:(n_pos-5)]
                zR = convolve(Rcuelocs[5:], y[:, spl], mode = 'full')[:(n_pos-5)]

                feats = np.concatenate((feats, zL.reshape(-1, 1), zR.reshape(-1, 1)), axis = 1)
                
                if toplot:
                    lstouse = Lcuelocs.copy()
                    rstouse = Rcuelocs.copy()
                    plt.figure()
                    plt.plot(zL, color = 'b')
                    plt.plot(zR, color = 'r')
                    plt.savefig('Figure4Plots/CueResponse/ExampleSpline'+str(spl)+'.pdf')
                    
                
            if toplot:
                numplot = 1
                toplot = False
                    
            X[idx*(n_pos-5):(idx+1)*(n_pos-5), :] = feats
        
        if len(leftis)>0 and len(rightis)>0:
            leftactivity = alldata[:, leftis, :]
            lefttotal = np.nanmean(leftactivity, axis=1)
            #for i, p in enumerate(pos):
             #   validns = np.where(np.abs(leftmups-p)<leftsigps)[0]
              #  lefttotal[:, i] = np.nanmean(leftactivity[:, validns, i], axis=1)
            #lefttotal = lefttotal-np.nanmean(lefttotal, axis=0)
            #muL = np.nanmean(lefttotal[:, :6], axis=1)
            #sigL = np.nanstd(lefttotal[:, :6], axis = 1)
            #lefttotal = lefttotal - muL.reshape(-1, 1)
            #lefttotal = lefttotal/sigL.reshape(-1, 1)
            lefttotal = lefttotal[:, 5:]
            
            rightactivity = alldata[:, rightis, :] 
            righttotal = np.nanmean(rightactivity, axis = 1)
            #for i, p in enumerate(pos):
             #   validns = np.where(np.abs(rightmups-p)<rightsigps)[0]
              #  righttotal[:, i] = np.nanmean(rightactivity[:, validns, i], axis=1) 
            #righttotal = righttotal-np.nanmean(righttotal, axis=0)
            #muR = np.nanmean(righttotal[:, :6], axis=1)
            #sigR = np.nanstd(righttotal[:, :6], axis = 1)
            #righttotal = righttotal - muR.reshape(-1, 1)
            #righttotal = righttotal/sigR.reshape(-1,1)
            righttotal = righttotal[:, 5:]
            
            difftotal = (lefttotal - np.nanmean(lefttotal, axis=0))-(righttotal-np.nanmean(righttotal, axis=0))
            
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
                
                if f<5:
                    plt.figure()
                    plt.plot(yd[:300])
                    plt.plot(ypred[:300])
                    plt.title(region)
            
                Lcuekernel = y[:, :7]@coefs[2:16:2] 
                Rcuekernel = y[:, :7]@coefs[3:16:2]
                
                if f == 0 and region=='ACC':
                    plt.figure()
                    plt.plot(-1*Lcuekernel, c = 'b')
                    plt.ylim([-.1, .1])
                    plt.savefig('Figure4Plots/CueResponse/ExampleLkernel.pdf')
                    
                    plt.figure()
                    plt.plot(-1*Rcuekernel, c = 'r')
                    plt.ylim([-.1, .1])
                    plt.savefig('Figure4Plots/CueResponse/ExampleRkernel.pdf')
                    
                    kL = convolve(lstouse[5:], Lcuekernel, mode = 'full')[:(n_pos-5)]
                    kR = convolve(rstouse[5:], Rcuekernel, mode = 'full')[:(n_pos-5)]

                    plt.figure()
                    plt.plot(-1*kL, c = 'b')
                    plt.ylim([-.25, .25])
                    plt.savefig('Figure4Plots/CueResponse/ExampleLkernel-convolved.pdf')

                    plt.figure()
                    plt.plot(-1*kR, c = 'r')
                    plt.ylim([-.25, .25])
                    plt.savefig('Figure4Plots/CueResponse/ExampleRkernel-convolved.pdf')

                    plt.figure()
                    plt.plot(-kL-kR)
                    plt.ylim([-.25, .25])
                    plt.savefig('Figure4Plots/CueResponse/Examplesummed.pdf')                        

                
        
                lefttoLcuekernels[a] = np.vstack((lefttoLcuekernels[a], Lcuekernel))
                lefttoRcuekernels[a] = np.vstack((lefttoRcuekernels[a], Rcuekernel))

                
                performance[a].append(model.score(X, yd))
                
                
                if f<10:
                    try:
                        ypred_unflat = ypred.reshape(np.shape(difftotal))
                    
                        bestcorrs = np.zeros(len(difftotal))
                        #for i in range(len(difftotal)):
                         #   bestcorrs[i] = mse(difftotal[i], ypred_unflat[i])#[0]
                        #besttrials = np.argsort(bestcorrs)#[::-1]
                        
                        plt.figure()
                        for j in range(5):
                            plt.plot(difftotal[j], color = colors[j])
                            plt.plot(ypred_unflat[j], linestyle = '--', color = colors[j])
                            #plt.plot(difftotal[besttrials[j]], color = colors[j])
                            #plt.plot(ypred_unflat[besttrials[j]], linestyle = '--', color = colors[j])
                        
                        plt.xlim([0,60])
                        plt.ylim([-1.2, 1.2])
                        plt.savefig('Figure4Plots/CueResponse/'+region+'sessionexamples'+str(f)+'.pdf')
                    except:
                        print('contained nans')
                        
                
            

              
    for a in alphalist:          
        lefttoLcuekernels[a] = lefttoLcuekernels[a][1:]
        lefttoRcuekernels[a] = lefttoRcuekernels[a][1:]
        
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
        plt.title(region+'- early cues'+', alpha = '+str(a)+', r2 ='+str(perf))
    
        plt.figure()
        plt.errorbar(np.arange(0, 200, 5), np.nanmean(lefttoLcuekernelslate[a], axis=0), yerr = sem(lefttoLcuekernelslate[a], axis=0, nan_policy='omit'), color = 'red')
        plt.errorbar(np.arange(0, 200, 5), np.nanmean(lefttoRcuekernelslate[a], axis=0), yerr = sem(lefttoRcuekernelslate[a], axis=0, nan_policy='omit'), color = 'blue')
        #plt.errorbar(np.arange(0, 300, 5), np.nanmean(righttoLcuekernelslate, axis=0), yerr = sem(righttoLcuekernelslate, axis=0, nan_policy='omit'), color = 'blue', alpha=.5)
        #plt.errorbar(np.arange(0, 300, 5), np.nanmean(righttoRcuekernelslate, axis=0), yerr = sem(righttoRcuekernelslate, axis=0, nan_policy='omit'), color = 'blue')
        plt.xlabel('distance from cue appearance (cm)')
        plt.ylabel('kernel amplitude')
        plt.ylim([-.1, .1])
        plt.axhline(0, color = 'k', linestyle = '--')
        plt.title(region+'- late cues'+', alpha = '+str(a)+', r2 ='+str(perf))
        '''

        plt.figure()
        plt.errorbar(np.arange(0, 300, 5), np.nanmean(lefttoLcuekernels[a], axis=0), yerr = sem(lefttoLcuekernels[a], axis=0, nan_policy='omit'), color = 'red')
        plt.errorbar(np.arange(0, 300, 5), np.nanmean(lefttoRcuekernels[a], axis=0), yerr = sem(lefttoRcuekernels[a], axis=0, nan_policy='omit'), color = 'blue')
        plt.xlabel('distance from cue appearance (cm)')
        plt.ylabel('kernel amplitude')
        #plt.axvline(0, linestyle = '--', color = 'k')
        #plt.axvline(10, linestyle = '--', color = 'grey')
        plt.ylim([-.1, .1])
        plt.xlim([0, 200])
        plt.axhline(0, color = 'k', linestyle = '--')
        plt.title(region+', alpha = '+str(a)+', r2 ='+str(perf))
        plt.savefig('Figure4Plots/CueResponse/'+region+'LRCueKernels.pdf')         
                        