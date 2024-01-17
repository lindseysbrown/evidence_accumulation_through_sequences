# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 14:35:00 2023

@author: lindseyb
"""

import os
from scipy.io import loadmat
import numpy as np
import h5py
import pandas as pd


regions = ['FFchains', 'MIchains', 'bump', 'ACC', 'DMS', 'HPC', 'V1', 'RSC','inputs']

for region in regions:
    print(region)
    regionparams = pd.DataFrame()
    path = region + '/paramfit-boundedmue'
    files = os.listdir(path)
    #fitparams = np.zeros((10, 1))
    for f in files:
        if not f.startswith(region):
            try:
                ps = np.load(region+'/paramfit-boundedmue/'+f)
                try:
                    obs = np.load(region+'/firingdata/'+f)
                except:
                    obs = np.load(region+'/firingdata-split/'+f)
                evs = obs[:, 2]
                poses = obs[:, 1]
                validps1 = poses>(ps[0]-ps[1]) 
                validps2 = poses<(ps[0]+ps[1])
                validps = validps1 & validps2
                if sum(validps)>0:
                    unique, counts = np.unique(evs[validps], return_counts=True)
                    es = np.argsort(unique)
                    i = 0
                    while counts[es[i]]<3:
                        i = i+1
                    minE = unique[es[i]]
                    i = -1
                    while counts[es[i]]<3:
                        i = i-1
                    maxE = unique[es[i]]
                    maxEraw = np.max(evs[validps])
                    minEraw = np.min(evs[validps])
                else:
                    maxE = np.nan
                    minE = np.nan
                    maxEraw = np.nan
                    minEraw = np.nan
                validps1 = poses>(ps[0]-5)
                validps2 = poses<(ps[0]+5)
                validps = validps1&validps2
                if sum(validps)>0:
                    maxEmeanp = np.max(evs[validps])
                    minEmeanp = np.min(evs[validps])
                else:
                    maxEmeanp = np.nan
                    minEmeanp = np.nan
                
                ps = np.concatenate((ps, np.array([maxE, minE, maxEraw, minEraw, maxEmeanp, minEmeanp])))
                session, index = f.split('neuron')
                index = index.split('.')[0]
                entry = {'Session':session, 'Neuron':index, 'Mup': ps[0], 'Sigp':ps[1], 'Mue':ps[2], 'Sige':ps[3], 'Correlation':ps[4], 'Pval':ps[5], 'MaxE':ps[6], 'MinE':ps[7], 'MaxERaw': ps[8], 'MinERaw':ps[9], 'MaxEMean':ps[10], 'MinEMean':ps[11]}
                regionparams = regionparams.append(entry, ignore_index=True)
            except:
                print('Error loading '+region+f)
        regionparams.to_csv(region+'/paramfit-boundedmue/'+region+'allfitparams-boundedmue-NEW.csv', index=False)
            #fitparams = np.concatenate((fitparams, ps.reshape(-1,1)), axis=1)
    #fitparams = fitparams[:, 1:]
    #np.save(region+'/paramfit/'+region+'allfitparams.npy', fitparams)
