# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 15:06:44 2024

@author: lindseyb
"""

import numpy as np
import cebra
from cebra import CEBRA
from scipy.stats import pearsonr
from sklearn.model_selection import KFold


def get_shuffled_ev(trials, poses, evs, shufflereps, ltrials, rtrials):
    '''
    build shuffled datasets that preserve the animal's choice

    ==== INPUTS =====
    trials: number of trials
    evs: original evidence values that match the data
    shufflereps: number of shuffled datasets to produce
    ltrials: trials on which a left choice was made
    rtrials: trials on which a right choice was made

    === OUTPUTS ===
    shuffledev: returns shuffled evidence to use for each of the shuffles 
    '''
    shuffledev = np.zeros((len(evs), shufflereps))
    totaltrials = int(trials[-1])+1
    ev_data_shuffle = evs.copy()
    ev_data_shuffle  = ev_data_shuffle.reshape(-1, 60) #reshape evidence to be in the shape (num. trials, num. positions)
    for n in range(shufflereps):
        evnew = np.array([])
        z = np.arange(totaltrials)
        for ti in range(len(z)):
            if ti in ltrials:
                newti = np.random.choice(ltrials)
                evnew = np.concatenate((evnew, ev_data_shuffle[newti, :]))
            else:
                newti = np.random.choice(rtrials)
                evnew = np.concatenate((evnew, ev_data_shuffle[newti, :]))
        shuffledev[:, n] = evnew
    return shuffledev


max_iterations = 10000

animals = ['nicFR_E65_20180202']

manifold_dims = np.arange(1, 8)

shufflereps = 5

decodingleftcorr = np.nan*np.zeros((len(manifold_dims),len(animals), 60, 5))
decodingleftcorrS = np.nan*np.zeros((len(manifold_dims),shufflereps*len(animals), 60, 5))


decodingrightcorr = np.nan*np.zeros((len(manifold_dims),len(animals), 60, 5))
decodingrightcorrS = np.nan*np.zeros((len(manifold_dims),shufflereps*len(animals), 60, 5))

decodingallcorr = np.nan*np.zeros((len(manifold_dims),len(animals), 60, 5))
decodingallcorrS = np.nan*np.zeros((len(manifold_dims),shufflereps*len(animals), 60, 5))

decodingleftall = np.nan*np.zeros((len(manifold_dims), len(animals), 5))
decodingrightall = np.nan*np.zeros((len(manifold_dims), len(animals), 5))
decodingallall = np.nan*np.zeros((len(manifold_dims), len(animals), 5))


decodingleftallS = np.nan*np.zeros((len(manifold_dims), shufflereps*len(animals), 5))
decodingrightallS = np.nan*np.zeros((len(manifold_dims), shufflereps*len(animals), 5))
decodingallallS = np.nan*np.zeros((len(manifold_dims), shufflereps*len(animals), 5))


for fi, a in enumerate(animals): #load data and decode for each session
    print(a)
    a = a.strip()
    neural_data = cebra.load_data('ExampleData/CEBRAExampleData/'+a+'-neural.npy')
    pos_data = cebra.load_data('ExampleData/CEBRAExampleData/'+a+'-pos.npy')
    ev_data = cebra.load_data('ExampleData/CEBRAExampleData/'+a+'-ev.npy')
    trial_data = cebra.load_data('ExampleData/CEBRAExampleData/'+a+'-trials.npy')
    
    ltrials = []
    rtrials = []
    for i in range(1, len(pos_data)):
        if pos_data[i-1]>(pos_data[i]+5):
            if ev_data[i-1]>0:
                rtrials.append(trial_data[i-1])
            if ev_data[i-1]<0:
                ltrials.append(trial_data[i-1])
                
    evshuffle = get_shuffled_ev(trial_data, pos_data, ev_data, shufflereps, ltrials, rtrials) #create shuffled evidence sets

    kfL = KFold(n_splits=5, shuffle = True, random_state = 42) #5 fold cross validation
    lefttrains = {}
    lefttests = {}

    kfR = KFold(n_splits=5, shuffle = True, random_state = 42) #5 fold cross validation
    righttrains = {}
    righttests = {}
    
    foldcount = 0
    for train_idx, test_idx in kfL.split(ltrials):
        lefttests[foldcount] = test_idx
        lefttrains[foldcount] = train_idx
        foldcount = foldcount+1
    
    foldcount = 0
    for train_idx, test_idx in kfR.split(rtrials):
        righttests[foldcount] = test_idx
        righttrains[foldcount] = train_idx
        foldcount = foldcount+1

    for fold in range(5): #for each fold, divide into train and test sets
        print(fold)
        L_testtrials = np.array(ltrials)[lefttests[fold]]
        L_traintrials = np.array(ltrials)[lefttrains[fold]]
        R_testtrials = np.array(rtrials)[righttests[fold]]
        R_traintrials = np.array(rtrials)[righttrains[fold]]

        alltrialstrain = [t in L_traintrials or t in R_traintrials for t in trial_data]
        alltrialstest = [t in L_testtrials or t in R_testtrials for t in trial_data]
        
        train_data = neural_data[alltrialstrain]
        train_label = ev_data[alltrialstrain]
        
        valid_data = neural_data[alltrialstest]
        valid_label = ev_data[alltrialstest]


        
        
        for d_idx, d in enumerate(manifold_dims):
            #fit the manifold
            cebra_model = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=3e-4,
                        temperature_mode='auto',
                        output_dimension=d,
                        max_iterations=max_iterations,
                        distance='cosine',
                        conditional='time_delta',
                        device='cuda_if_available',
                        verbose=False,
                        time_offsets=10)
            

            cebra_model.fit(train_data, train_label)

            train_embedding = cebra_model.transform(train_data)
            valid_embedding = cebra_model.transform(valid_data)
            
            decoder = cebra.KNNDecoder() #use K nearest neighbors decoder to predict evidence from the manifold
            decoder.fit(train_embedding, train_label)
            
            prediction = decoder.predict(valid_embedding)
            
            decodingallall[d_idx, fi, fold] = pearsonr(prediction, valid_label)[0]
            
            for p in range(60):
                decodingallcorr[d_idx, fi, p, fold] = pearsonr(prediction[np.abs(pos_data[alltrialstest]-5*p)<5], valid_label[np.abs(pos_data[alltrialstest]-5*p)<5])[0] #evaluate decoder performance
            
        print('starting shuffle') #perform decoding on choice matched shuffled evidence sets
        for e in range(shufflereps):
            train_data = neural_data[alltrialstrain]
            train_label = evshuffle[:, e][alltrialstrain]
            
            valid_data = neural_data[alltrialstest]
            valid_label = evshuffle[:, e][alltrialstest]
    
    
            
            
            for d_idx, d in enumerate(manifold_dims):
                #fit the manifold
                cebra_model = CEBRA(model_architecture='offset10-model',
                            batch_size=512,
                            learning_rate=3e-4,
                            temperature_mode='auto',
                            output_dimension=d,
                            max_iterations=max_iterations,
                            distance='cosine',
                            conditional='time_delta',
                            device='cuda_if_available',
                            verbose=False,
                            time_offsets=10)
                
    
                cebra_model.fit(train_data, train_label)
    
                train_embedding = cebra_model.transform(train_data)
                valid_embedding = cebra_model.transform(valid_data)
                
                decoder = cebra.KNNDecoder()
                decoder.fit(train_embedding, train_label)
                
                prediction = decoder.predict(valid_embedding)
                
                decodingallallS[d_idx, fi*e+e, fold] = pearsonr(prediction, valid_label)[0]
                
                for p in range(60):
                    decodingallcorrS[d_idx, fi*e+e, p, fold] = pearsonr(prediction[np.abs(pos_data[alltrialstest]-5*p)<5], valid_label[np.abs(pos_data[alltrialstest]-5*p)<5])[0]
                        
#save decoding at individual positions
decodingallcorr = np.nanmean(decodingallcorr, axis=-1)
decodingallcorrS = np.nanmean(decodingallcorrS, axis=-1)

np.save('decodingallcorr'+a+'.npy', decodingallcorr)
np.save('decodingallcorrS'+a+'.npy', decodingallcorrS)


#save overall decoding performance
decodingallall = np.nanmean(decodingallall, axis=-1)
decodingallallS = np.nanmean(decodingallallS, axis=-1)

np.save('decodingallall'+a+'.npy', decodingallall)
np.save('decodingallallS'+a+'.npy', decodingallallS)

     
             
            
