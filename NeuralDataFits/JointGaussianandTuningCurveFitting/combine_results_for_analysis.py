# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 14:35:00 2023

@author: lindseyb
"""

import os
import numpy as np
import pandas as pd


regions = ['FFchains', 'MIchains', 'bump', 'ACC', 'DMS', 'HPC', 'V1', 'RSC','inputs']

for region in regions:
    print(region)
    regionparams = pd.DataFrame()
    path = region + '/paramfit' #find all the files containing the individual cell fits for that region
    files = os.listdir(path)
    for f in files:
        if not f.startswith(region): #don't run for a previously compiled list of parameters
            try:
                ps = np.load(region+'/paramfit-boundedmue/'+f) #load fit parameters
                obs = np.load(region+'/firingdata/'+f) #load corresponding data, where 1st column is the firing rates, 2nd column is positions, and 3rd column is cumulative evidences
                evs = obs[:, 2] #cumulative evidence, 3rd column
                poses = obs[:, 1] #position, 2nd column
                #find positions within 1 fit position standard deviation (ps[1]) or fit position mean (ps[0])
                validps1 = poses>(ps[0]-ps[1])
                validps2 = poses<(ps[0]+ps[1])
                validps = validps1 & validps2
                
                #determine the range of evidence levels observed at the active position
                if sum(validps)>0:
                    unique, counts = np.unique(evs[validps], return_counts=True) #all observed evidence levels within active position
                    #determine the typical range (defined by at least 3 observations)
                    es = np.argsort(unique)
                    i = 0
                    while counts[es[i]]<3:
                        i = i+1
                    minE = unique[es[i]] #typical min
                    i = -1
                    while counts[es[i]]<3:
                        i = i-1
                    maxE = unique[es[i]] #typical max
                    maxEraw = np.max(evs[validps]) #raw max
                    minEraw = np.min(evs[validps]) #raw min
                else:
                    maxE = np.nan
                    minE = np.nan
                    maxEraw = np.nan
                    minEraw = np.nan

                #determine range of evidence at the mean position
                validps1 = poses>(ps[0]-5) 
                validps2 = poses<(ps[0]+5)
                validps = validps1&validps2
                if sum(validps)>0:
                    maxEmeanp = np.max(evs[validps])
                    minEmeanp = np.min(evs[validps])
                else:
                    maxEmeanp = np.nan
                    minEmeanp = np.nan
                
                ps = np.concatenate((ps, np.array([maxE, minE, maxEraw, minEraw, maxEmeanp, minEmeanp]))) #combine all parameters
                session, index = f.split('neuron')
                index = index.split('.')[0]
                entry = {'Session':session, 'Neuron':index, 'Mup': ps[0], 'Sigp':ps[1], 'Mue':ps[2], 'Sige':ps[3], 'Correlation':ps[4], 'Pval':ps[5], 'MaxE':ps[6], 'MinE':ps[7], 'MaxERaw': ps[8], 'MinERaw':ps[9], 'MaxEMean':ps[10], 'MinEMean':ps[11]} #add to dataframe
                regionparams = regionparams.append(entry, ignore_index=True)
            except:
                print('Error loading '+region+f)
        regionparams.to_csv(region+'/paramfit/'+region+'allfitparams.csv', index=False) #save resulting dataframe for future analysis

