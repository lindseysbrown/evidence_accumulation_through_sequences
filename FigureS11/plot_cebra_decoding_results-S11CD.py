# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 15:10:44 2024

@author: lindseyb
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem

demo = True #plot results for a single session
if demo:
    animals = ['nicFR_E65_20180202']
else:
    animals = ['nicFR_E22_20170227', 'nicFR_E39_20171103', 'nicFR_E43_20170802', 'nicFR_E44_20171018', 'nicFR_E47_20170927', 'nicFR_E48_20170829', 'nicFR_E65_20180202']
manifold_dims = np.arange(2, 8)
shufflereps = 5

decodingallcorr = np.nan*np.zeros((len(manifold_dims),len(animals), 60))
decodingallcorrS = np.nan*np.zeros((len(manifold_dims),shufflereps*len(animals), 60))

decodingallall = np.nan*np.zeros((len(manifold_dims), len(animals)))

decodingallallS = np.nan*np.zeros((len(manifold_dims), shufflereps*len(animals)))

for fi, a in enumerate(animals):
    decodingallall[:, fi] = np.load('ExampleData/CEBRAExampleData/decodingallall'+a+'.npy').reshape(-1)
    decodingallallS[:, fi*shufflereps:(fi+1)*shufflereps] = np.load('ExampleData/CEBRAExampleData/decodingallallS'+a+'.npy')
    decodingallcorr[:, fi, :] = np.load('ExampleData/CEBRAExampleData/decodingallcorr'+a+'.npy')[:, 0, :]
    decodingallcorrS[:, fi*shufflereps:(fi+1)*shufflereps, :] = np.load('ExampleData/CEBRAExampleData/decodingallcorrS'+a+'.npy')
    
#plot evoluation of accuracy vs. position
pos = np.arange(0, 300, 5)    
for d_idx, d in enumerate(manifold_dims[:1]):
    plt.figure()
    plt.plot(pos, np.nanmean(decodingallcorr[d_idx, :, :], axis=0), color='purple')
    plt.fill_between(pos, np.nanmean(decodingallcorr[d_idx, :, :], axis=0)-sem(decodingallcorr[d_idx, :, :], axis=0, nan_policy='omit'), np.nanmean(decodingallcorr[d_idx, :, :], axis=0)+sem(decodingallcorr[d_idx, :, :], axis=0, nan_policy='omit'), alpha = .5, color = 'purple')   
    plt.ylabel('Pearson Correlation (r)')
    plt.xlabel('Position') 
    plt.xlim([0, 300])
    plt.title('n_dims = '+str(d))
    plt.fill_between(pos, np.nanmean(decodingallcorrS[d_idx, :, :], axis=0)-sem(decodingallcorrS[d_idx, :, :], axis=0, nan_policy='omit'), np.nanmean(decodingallcorrS[d_idx, :, :], axis=0)+sem(decodingallcorrS[d_idx, :, :], axis=0, nan_policy='omit'), alpha = .5, color = 'grey')  
    plt.show()

#plot overall accuracy for each manifold dimension
plt.figure()
plt.bar(manifold_dims-.2, np.nanmean(decodingallall, axis = 1), .4) 
plt.bar(manifold_dims+.2, np.nanmean(decodingallallS, axis = 1), .4, color = 'grey')
for d, dim in enumerate(manifold_dims):
    for i in range(len(animals)):
        plt.plot(np.array([dim-.2, dim+.2]), np.array([decodingallall[d, i], np.nanmean(decodingallallS[d, i*5:(i+1)*5])]), color = 'k', marker = 'o', markersize=3)
plt.xlabel('Manifold Dimensions')
plt.ylabel('Decoding Accuracy (r)')
plt.show() 