# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 10:19:35 2023

@author: lindseyb
"""

import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV, LogisticRegressionCV
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
from scipy import stats
import os
from scipy.stats import sem, f, pearsonr
from scipy.stats import f_oneway, kurtosis, skew
import pandas as pd
import pickle
import statsmodels.api as sm
from sklearn.model_selection import cross_validate, cross_val_predict
import sklearn

sigonly = True

alpha_list = [.0001, .001, .01, .1, 1, 10, 100, 1000]

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

def get_sig(XL, XR, yL, yR):
    evs = np.concatenate((yL, yR))
    frs = np.concatenate((XL, XR), axis = 0)
    cs = np.sign(evs)
    factors = np.zeros((len(evs),2))
    factors[:, 0] = evs
    factors[:, 1] = cs
    factors = sm.add_constant(factors)
    signs = []
    for i in range(np.shape(frs)[1]):
        result = sm.OLS(frs[:, i], factors).fit()
        if result.pvalues[1]<.1:
            signs.append(i)
    return np.array(signs) 

def bimodalC(data):
    if np.ptp(data) <1.01:
        return np.nan
    s = skew(data, nan_policy='omit')
    k = kurtosis(data, nan_policy='omit')
    n = sum(~np.isnan(data))
    nfactor = (n-1)**2/((n-2)*(n-3))
    return (s**2+1)/(k+3*nfactor)

def pearsonronly(X, y):
    return pearsonr(X, y)[0]

corrmetric = sklearn.metrics.make_scorer(pearsonronly)    
    
regions = ['ACC', 'RSC']#'DMS', 'HPC', 'RSC']

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

filelist = [ACCmatfiles, RSCmatfiles] #MSmatfiles,  HPCmatfiles,  RSCmatfiles]

with open('numcorrecttrials.pkl', 'rb') as handle:
    numcorrecttrials = pickle.load(handle)

for region, matfiles in zip(regions, filelist):
    #fitparams = np.load(region+'/paramfit/'+region+'allfitparams-linearencoding-signedevidence.npy')
    #fitneurondata = pd.read_csv(region+'/paramfit/'+region+'allfitparams-linearencoding-neuroninfo-signedevidence.csv')
    fitparams = pd.read_csv(region+'/paramfit/'+region+'allfitparams-boundedmue-MSE.csv')

        
    if region == 'RSC':
        shufflereps = 5
    else:
        shufflereps = 5 #previously 10
    
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
        session = file.split('.')[0]
        #ncorrect = numcorrecttrials[session]
        #sigF = f.ppf(.95, 1, ncorrect-3)
        #validsession = np.where(fitneurondata['Session'].values == session)[0]
        #neuronindices = fitneurondata['Index'][fitneurondata.Session==session].values
        
        
        data = loadmat(region+'/'+file)
        
        if region in ['DMS', 'ACC']:
            bfile = file.split('dFF_')[1]
            bfile = bfile.split('processedOutput.mat')[0]+'.mat'
            cuedata = loadmat(region+'/Behavior/'+bfile)
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
            
            
            #baseline subtraction from start of cue region
            baseline = alldata[:, :, 6]
        
            #center activity relative to start of trial
            #for p in range(n_pos):
             #   alldata[:, :, p] = alldata[:, :, p] - baseline
            
            
            #leftalldata = alldata[lchoices, :, :]
            #leftevidence = evidence[lchoices, :]
            
            #rightalldata = alldata[rchoices, :, :]
            #rightevidence = evidence[rchoices, :]
            
            for p in range(6, n_pos):
                validnsbyp = (np.abs(mups-p)<sigps)*1
                validnsbye = evsel*1
                #validns = [validnsbyp[d] and validnsbye[d] for d in range(len(evsel))]
                validns = validnsbyp
                validns = np.where(validns)[0]
                #validns = np.arange(n_neurons)
                
                evatp = evidence[:, p]
                
                if len(validns)>0:
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
                    
                    imputer = KNNImputer()
                    
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
                    #XL_train, XL_test, yL_train, yL_test = train_test_split(XL, yL, test_size=0.2, random_state=42)
                    #XR_train, XR_test, yR_train, yR_test = train_test_split(XR, yR, test_size=0.2, random_state=42)
                                                 
    
                    #standardize neural firing data
                    #Lscaler = StandardScaler()
                    #Lscaler.fit(XL_train)
                    #ZL_train = Lscaler.transform(XL_train)
                    #ZL_test = Lscaler.transform(XL_test)
                    #ZL_train = XL_train
                    #ZL_test = XL_test
                    
                    #Rscaler = StandardScaler()
                    #Rscaler.fit(XR_train)
                    #ZR_train = Rscaler.transform(XR_train)
                    #ZR_test = Rscaler.transform(XR_test)
                    #ZR_train = XR_train
                    #ZR_test = XR_test
                    
                    #perform crossvalidation
                    #Lridge = RidgeCV(alphas = alpha_list, cv=5)
                    #Lridge.fit(ZL_train, yL_train)
                    #yL_pred = Lridge.predict(ZL_test)
                    #decodingleftr[fi, p] = mean_squared_error(yL_pred, yL_test, squared = False)#stats.pearsonr(yL_pred, yL_test)[0]
                    #decodingleftr2[fi, p] = Lridge.score(ZL_test, yL_test)
                    #decodingleftcorr[fi, p] = np.nanmean(left_results['test_score']) #pearsonr(yL_pred, yL_test)[0]
                    
                    #Rridge = RidgeCV(alphas = alpha_list, cv=5)
                    #Rridge.fit(ZR_train, yR_train)
                    #yR_pred = Rridge.predict(ZR_test)
                    #decodingrightr[fi, p] = mean_squared_error(yR_pred, yR_test, squared = False)#stats.pearsonr(yR_pred, yR_test)[0]
                    #decodingrightr2[fi, p] = Rridge.score(ZR_test, yR_test)
                    #decodingrightcorr[fi, p] = np.nanmean(right_results['test_score']) #pearsonr(yR_pred, yR_test)[0]
                    
                    #bimodalleft[fi, p] = bimodalC(predyL)
                    #bimodalright[fi, p] = bimodalC(predyR)
                    #bimodallefttrue[fi, p] = bimodalC(yL)
                    #bimodalrighttrue[fi, p] = bimodalC(yR)
                    
                    #perform crossvalidation on shuffles
                    #ySL_train = yL_train.copy()
                    #ySL_test = yL_test.copy()
                    
                    #ySR_train = yR_train.copy()
                    #ySR_test = yR_test.copy()
                    
                    if len(yL)>10:
                        ySL = yL.copy()
    
                        for j in range(shufflereps):
                            np.random.shuffle(ySL)                      
                            
                            LSridge = RidgeCV(alphas = alpha_list, cv=5)
                            leftshuffle = cross_validate(LSridge, XL, ySL, scoring = corrmetric)
                            #decodingleftrS[fi*shufflereps+j, p] = mean_squared_error(ySL_pred, ySL_test, squared = False)#stats.pearsonr(ySL_pred, ySL_test)[0]
                            #decodingleftr2S[fi*shufflereps+j, p] = LSridge.score(ZL_test, ySL_test)
                            decodingleftcorrS[fi*shufflereps+j, p] = np.nanmean(leftshuffle['test_score'])
                            

                    if len(yR)>10:
                        ySR = yR.copy()
    
                        for j in range(shufflereps):
                    
                            np.random.shuffle(ySR)
                            
                            RSridge = RidgeCV(alphas = alpha_list, cv=5)
                            rightshuffle = cross_validate(RSridge, XR, ySR, scoring = corrmetric)
                            #RSridge.fit(ZR_train, ySR_train)
                            #ySR_pred = RSridge.predict(ZR_test)
                            #decodingrightrS[fi*shufflereps+j, p] = mean_squared_error(ySR_pred, ySR_test, squared = False)#stats.pearsonr(ySR_pred, ySR_test)[0]
                            #decodingrightr2S[fi*shufflereps+j, p] = RSridge.score(ZR_test, ySR_test)
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

    np.save(region+'leftdecodingcorr-no0.npy', decodingleftcorr)
    np.save(region+'rightdecodingcorr-no0.npy', decodingrightcorr)
    np.save(region+'leftdecodingcorrshuffle-no0.npy', decodingleftcorrS)
    np.save(region+'rightdecodingcorrshuffle-no0.npy', decodingrightcorrS)

    plt.figure()
    plt.plot(pos, np.nanmean(decodingleftcorr, axis=0), color='b')
    plt.fill_between(pos, np.nanmean(decodingleftcorr, axis=0)-sem(decodingleftcorr, axis=0, nan_policy='omit'), np.nanmean(decodingleftcorr, axis=0)+sem(decodingleftcorr, axis=0, nan_policy='omit'), alpha = .5, color = 'b')
    plt.plot(pos, np.nanmean(decodingrightcorr, axis=0), color='r')
    plt.fill_between(pos, np.nanmean(decodingrightcorr, axis=0)-sem(decodingrightcorr, axis=0, nan_policy='omit'), np.nanmean(decodingrightcorr, axis=0)+sem(decodingrightcorr, axis=0, nan_policy='omit'), alpha = .5, color = 'r')
    #plt.plot(pos, np.nanmean(decodingleftrS, axis=0), color='grey')
    plt.plot(pos, np.nanmean(decodingrightcorrS, axis=0), color='grey', linestyle=':')
    plt.fill_between(pos, np.nanmean(decodingrightcorrS, axis=0)-sem(decodingrightcorrS, axis=0, nan_policy='omit'), np.nanmean(decodingrightcorrS, axis=0)+sem(decodingrightcorrS, axis=0, nan_policy='omit'), alpha = .5, color = 'grey')
    plt.plot(pos, np.nanmean(decodingleftcorrS, axis=0), color='grey', linestyle='--')
    plt.fill_between(pos, np.nanmean(decodingleftcorrS, axis=0)-sem(decodingleftcorrS, axis=0, nan_policy='omit'), np.nanmean(decodingleftcorrS, axis=0)+sem(decodingleftcorrS, axis=0, nan_policy='omit'), alpha = .5, color = 'grey')    
    plt.ylabel('Pearson Correlation (r)')
    plt.xlabel('Position') 
    plt.xlim([0, 300])
    plt.title(region)
    plt.ylim([-.25, .3])
    plt.axhline(0, color = 'k', linestyle = '--')
    plt.savefig('Figure4Plots/Decoding/'+region+'fullcvridge-signofecontrolled-no0.pdf')  

       
