# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 11:17:01 2023

@author: lindseyb
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
matplotlib.rcParams.update({'font.size': 12})
import pandas as pd
from sklearn.preprocessing import minmax_scale



def plot_ndata(data, evs, poses):
    '''
    === INPUTS ===
    data: array of firing rates (1st column), positions (2nd column), and cumulative evidences (3rd column)
    evs: evidence bins
    poses: position bins

    === OUTPUTS ===
    obs: average firing rate at each evidence x position bin, for which there are at least two observations
    '''

    frs = data[:, 0]
    data = data[~np.isnan(frs), :]
    frs = data[:, 0]
    pos = data[:, 1].reshape(-1, 1)
    ev = data[:, 2].reshape(-1, 1)
    frs = minmax_scale(frs)
         
    obs = np.zeros((len(evs), len(poses)))
    counts = np.zeros((len(evs), len(poses)))
    for i, e in enumerate(evs):
        for j, p in enumerate(poses):
            vp = pos[:, 0]==p
            ve = ev[:, 0]==e
            os = vp & ve
            counts[i][j] = sum(os)
            if sum(os)>1:
                obs[i][j] = np.mean(frs[os])
            else:
                obs[i][j] = np.nan
    return obs

#demo
demo = True #True to run with example datasets, False to run for full data in structured folders
            
current_cmap = matplotlib.cm.get_cmap('YlOrRd')
current_cmap.set_bad(color='black', alpha=.3) 

 
evs = np.arange(-15, 16)

exampleneurons = {'ACC': [(10, 'dFF_tetO_8_08022021_T10processedOutput'), (153, 'dFF_tetO_8_07282021_T10processedOutput'), (17, 'dFF_tetO_8_08052021_T11processedOutput'),
                          (26,  'dFF_tetO_7_07302021_T10processedOutput'),(37,  'dFF_tetO_8_07282021_T10processedOutput'),(8,  'dFF_tetO_9_07302021_T10processedOutput'),(138,  'dFF_tetO_8_08052021_T11processedOutput'),(4,  'dFF_tetO_8_08022021_T10processedOutput'),(14,  'dFF_tetO_8_08052021_T11processedOutput'),(9,  'dFF_tetO_8_08022021_T10processedOutput'),(8,  'dFF_tetO_7_07302021_T10processedOutput'),(17,  'dFF_tetO_9_07272021_T10processedOutput'),(7,  'dFF_tetO_7_07302021_T10processedOutput'),(21,  'dFF_tetO_7_07302021_T10processedOutput'),(53,  'dFF_tetO_9_07272021_T10processedOutput'),(91,  'dFF_tetO_8_08052021_T11processedOutput'),(15,  'dFF_tetO_9_07272021_T10processedOutput'),(14,  'dFF_tetO_7_07302021_T10processedOutput'),(22,  'dFF_tetO_8_08052021_T11processedOutput'),(67,  'dFF_tetO_8_07282021_T10processedOutput'),(43,  'dFF_tetO_8_08052021_T11processedOutput'),(85,  'dFF_tetO_8_07282021_T10processedOutput'),(58,  'dFF_tetO_8_08052021_T11processedOutput'),(11,  'dFF_tetO_8_08052021_T11processedOutput'),(12,  'dFF_tetO_7_07302021_T10processedOutput'),(146,  'dFF_tetO_8_07282021_T10processedOutput'),(50,  'dFF_tetO_8_08052021_T11processedOutput'),(150,  'dFF_tetO_9_07272021_T10processedOutput'),(10,  'dFF_tetO_8_08052021_T11processedOutput'),(49,  'dFF_tetO_8_07282021_T10processedOutput'),(19,  'dFF_tetO_8_07282021_T10processedOutput'),(184,  'dFF_tetO_7_07302021_T10processedOutput'),(10,  'dFF_tetO_8_07282021_T10processedOutput'),(147,  'dFF_tetO_9_07272021_T10processedOutput'),(2,  'dFF_tetO_8_07282021_T10processedOutput'),(9,  'dFF_tetO_8_07282021_T10processedOutput'),(65,  'dFF_tetO_8_07282021_T10processedOutput'),(131,  'dFF_tetO_8_08052021_T11processedOutput'),(154,  'dFF_tetO_8_07282021_T10processedOutput')],
                  'DMS': [(4, 'dFF_scott_d2_857_20190426processedOutput'), (7, 'dFF_scott_a2a_64_11072019processedOutput'),(1,'dFF_scott_d1_67_20190418processedOutput'), 
                          (57,  'dFF_scott_a2a_62_11072019processedOutput'),(21,  'dFF_scott_a2a_62_11052019processedOutput'),(66,  'dFF_scott_a2a_62_11142019processedOutput'),(9,  'dFF_scott_a2a_63_10312019processedOutput'),(105,  'dFF_scott_a2a_62_11072019processedOutput'),(47,  'dFF_scott_d2_856_20190425processedOutput'),(30,  'dFF_scott_a2a_62_11052019processedOutput'),(23,  'dFF_scott_a2a_62_11072019processedOutput'),(45,  'dFF_scott_a2a_62_11072019processedOutput'),(30,  'dFF_scott_a2a_62_11142019processedOutput'),(29,  'dFF_scott_a2a_64_11142019processedOutput'),(62,  'dFF_scott_a2a_62_11052019processedOutput'),(39,  'dFF_scott_d1_67_20190418processedOutput'),(19,  'dFF_scott_a2a_64_11072019processedOutput'),(14,  'dFF_scott_a2a_64_11142019processedOutput'),(11,  'dFF_scott_a2a_64_11072019processedOutput'),(10,  'dFF_scott_a2a_62_11072019processedOutput'),(80,  'dFF_scott_a2a_64_11142019processedOutput'),(14,  'dFF_scott_d1_67_20190418processedOutput'),(15,  'dFF_scott_a2a_64_11072019processedOutput'),(28,  'dFF_scott_d1_709_20190410processedOutput'),(46,  'dFF_scott_a2a_64_11142019processedOutput'),(76,  'dFF_scott_a2a_62_11142019processedOutput'),(1,  'dFF_scott_a2a_64_11142019processedOutput'),(2,  'dFF_scott_a2a_63_10312019processedOutput'),(55,  'dFF_scott_a2a_62_11142019processedOutput'),(21,  'dFF_scott_a2a_64_11142019processedOutput'),(10,  'dFF_scott_d1_67_20190418processedOutput'),(3,  'dFF_scott_d2_857_20190426processedOutput'),(8,  'dFF_scott_d1_67_20190418processedOutput'),(0,  'dFF_scott_a2a_63_10312019processedOutput'),(8,  'dFF_scott_a2a_63_10312019processedOutput'),(4,  'dFF_scott_a2a_63_10312019processedOutput'),(3,  'dFF_scott_d1_67_20190418processedOutput'),(5,  'dFF_scott_d2_857_20190426processedOutput'),(56,  'dFF_scott_a2a_64_11142019processedOutput')],
                  'HPC':[(50, 'nicFR_E39_20171103'), (25, 'nicFR_E43_20170802'), (118, 'nicFR_E22_20170227'),
                         (70,  'nicFR_E39_20171103'),(280,  'nicFR_E44_20171018'),(37,  'nicFR_E65_20180202'),(42,  'nicFR_E65_20180202'),(299,  'nicFR_E22_20170227'),(195,  'nicFR_E44_20171018'),(473,  'nicFR_E39_20171103'),(106,  'nicFR_E43_20170802'),(47,  'nicFR_E65_20180202'),(4,  'nicFR_E39_20171103'),(350,  'nicFR_E39_20171103'),(88,  'nicFR_E65_20180202'),(135,  'nicFR_E44_20171018'),(224,  'nicFR_E22_20170227'),(307,  'nicFR_E22_20170227'),(264,  'nicFR_E39_20171103'),(94,  'nicFR_E48_20170829'),(35,  'nicFR_E44_20171018'),(283,  'nicFR_E39_20171103'),(102,  'nicFR_E39_20171103'),(79,  'nicFR_E39_20171103'),(74,  'nicFR_E43_20170802'),(296,  'nicFR_E44_20171018'),(79,  'nicFR_E65_20180202'),(471,  'nicFR_E44_20171018'),(304,  'nicFR_E22_20170227'),(56,  'nicFR_E39_20171103'),(333,  'nicFR_E22_20170227'),(36,  'nicFR_E44_20171018'),(20,  'nicFR_E48_20170829'),(242,  'nicFR_E22_20170227'),(248,  'nicFR_E48_20170829'),(127,  'nicFR_E65_20180202'),(8,  'nicFR_E48_20170829'),(2,  'nicFR_E65_20180202'),(5,  'nicFR_E65_20180202')],
                  'RSC':[(3,'nicFR_k50_20160811_RSM_400um_114mW_zoom2p2processedFR'), (44,'nicFR_k31_20160114_RSM2_350um_98mW_zoom2p2processedFR'), (10, 'nicFR_k42_20160512_RSM_150um_65mW_zoom2p2processedFR'),
                         (2, 'nicFR_k42_20160519_RSM_300um_99mW_zoom2p2processedFR'),(12, 'nicFR_k42_20160512_RSM_150um_65mW_zoom2p2processedFR'),(23, 'nicFR_k31_20160114_RSM2_350um_98mW_zoom2p2processedFR'),(8, 'nicFR_k46_20160719_RSM_175um_83mW_zoom2p2processedFR'),(15, 'nicFR_k50_20160811_RSM_400um_114mW_zoom2p2processedFR'),(59, 'nicFR_k50_20160812_RSM_425um_1284mW_zoom2p2processedFR'),(7, 'nicFR_k50_20160812_RSM_425um_1284mW_zoom2p2processedFR'),(26, 'nicFR_k50_20160520_RSM_200um_41mW_zoom2p2processedFR'),(11, 'nicFR_k31_20160110_RSM_150um_65mW_zoom2p2processedFR'),(3, 'nicFR_k50_20160805_RSM_250um_57mW_zoom2p2processedFR'),(26, 'nicFR_k50_20160520_RSM_200um_41mW_zoom2p2processedFR'),(4, 'nicFR_k50_20160811_RSM_400um_114mW_zoom2p2processedFR'),(2, 'nicFR_k50_20160729_RSM_175um_47mW_zoom2p2processedFR'),(10, 'nicFR_k50_20160728_RSM_150um_41mW_zoom2p2processedFR'),(30, 'nicFR_k50_20160803_RSM_225um_65mW_zoom2p2processedFR'),(125, 'nicFR_k40_20160715_RSM_150um_65mW_zoom2p2processedFR'),(28, 'nicFR_k50_20160811_RSM_400um_114mW_zoom2p2processedFR'),(4, 'nicFR_k36_20160111_RSA_150um_65mW_zoom2p2processedFR'),(2, 'nicFR_k50_20160805_RSM_250um_57mW_zoom2p2processedFR'),(0, 'nicFR_k50_20160805_RSM_250um_57mW_zoom2p2processedFR'),(14, 'nicFR_k40_20160715_RSM_150um_65mW_zoom2p2processedFR'),(110, 'nicFR_k50_20160810_RSM_300um_75mW_zoom2p2processedFR'),(55, 'nicFR_k42_20160523_RSM_350um_99mW_zoom2p2processedFR'),(35, 'nicFR_k50_20160812_RSM_425um_1284mW_zoom2p2processedFR'),(17, 'nicFR_k40_20160720_RSM_200um_83mW_zoom2p2processedFR'),(27, 'nicFR_k40_20160715_RSM_150um_65mW_zoom2p2processedFR'),(48, 'nicFR_k46_20160719_RSM_175um_83mW_zoom2p2processedFR'),(111, 'nicFR_k50_20160810_RSM_300um_75mW_zoom2p2processedFR'),(22, 'nicFR_k42_20160520_RSM_325um_99mW_zoom2p2processedFR'),(13, 'nicFR_k42_20160523_RSM_350um_99mW_zoom2p2processedFR'),(101, 'nicFR_k55_20160613_RSM_125um_41mW_zoom2p2processedFR'),(30, 'nicFR_k50_20160729_RSM_175um_47mW_zoom2p2processedFR'),(3, 'nicFR_k40_20160817_RSM_375um_148mW_zoom2p2processedFR'),(11, 'nicFR_k50_20160727_RSM_200um_41mW_zoom2p2processedFR'),(22, 'nicFR_k36_20160111_RSA_150um_65mW_zoom2p2processedFR'),(39, 'nicFR_k50_20160728_RSM_150um_41mW_zoom2p2processedFR')]}


if demo:
    regions = ['ACC']
else:
    regions = ['ACC', 'HPC', 'DMS', 'RSC']

for region in regions:
    if region in ['ACC', 'DMS']:
        poses = np.arange(-27.5, 302.5, 5)
    else:
        poses = np.arange(-30, 300, 5) 
    
    if not demo:
        nstoplot = exampleneurons[region] 
    else:
        nstoplot = [(153, 'dFF_tetO_8_07282021_T10processedOutput')]  

    #load parameters from joint gaussian fit
    if not demo:
        fitparams = pd.read_csv(region+'/paramfit/'+region+'allfitparams.csv')
    else:
        fitparams = pd.read_csv('ExampleData/ACCparamfitexample.csv')


    for num, exampleplotneuron in enumerate(nstoplot):
        n = exampleplotneuron[0]
        s = exampleplotneuron[1]
        params = fitparams[(fitparams.Neuron==exampleplotneuron[0]) & (fitparams.Session==exampleplotneuron[1])].iloc[0]
        
        if demo:
            ndata = np.load('ExampleData/exampleneuron.npy') #example data corresponding to example plot neuron
        else:
            try:
                ndata = np.load(region+'/firingdata/'+s+'neuron'+str(n)+'.npy')
            except:
                ndata = np.load(region+'/firingdata-split/'+s+'neuron'+str(n)+'.npy') 
            
        mup = params['Mup']
        sigp = params['Sigp']
        

        obs = plot_ndata(ndata, evs, poses, params['Mue'], params['Mup'], params['Sigp'], n, s, params['Correlation'], 'early', num)            
        vm = np.nanmin(obs)
        vM = np.nanmax(obs)
        
        plt.figure()
        plt.imshow(obs, cmap = 'YlOrRd', interpolation = 'none', vmin=vm, vmax=vM)
        plt.xticks([0, 5.5, 25.5, 45.5, 55.25, 65], labels = ['-30', '0', 'cues', '200', 'delay', '300'])
        plt.yticks([0, 15, 30], labels = ['15', '0', '-15'])
        plt.axvline(min((mup+.5*sigp+30)/5, 65), color = 'k', linestyle = '--')
        plt.axvline(max((mup-.5*sigp+30)/5, 0), color = 'k', linestyle = '--') 
        plt.show()        

    
        


    
        
        
    