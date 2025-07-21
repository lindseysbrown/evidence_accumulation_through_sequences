# evidence_accumulation_through_sequences

This repository contains models and analysis code for Brown et al. 2023, bioRxiv to reproduce each figure.

# System Requirements
Code has been developed and run on Windows 10. All code can be run on a standard personal computer.
All standard python package dependencies are included in the requirements.txt file.

# Installation Guide
Code for the main and supplementary figures can be installed by cloning this repository and installing the necessary requirements from the requirements.txt file. Code for the spike inference algorithm was previously developed (Jewell et al., 2020), and instructions for download and use can be found: https://github.com/jewellsean/FastLZeroSpikeInference.

Installation of all requirements should take less than 15 minutes.

Example datasets are included in the ExampleData folder, which can be used to run the scripts for producing the figures. Full datasets associated with this paper will be made available on figshare upon publication. 

# Demo
All scripts may be run to produce the desired output figures from correctly formatted input data. Example datasets are available in the ExampleData folder. In order to test code, all files which rely on the use of full neural datasets have a parameter called 'demo'. When 'demo' is set to True, the script will run using just the subset of data available in the ExampleData folder.

For figures relying on model output (Figure3, Figure4), the scripts for the three models should be run first to produce output data from the simulations.

Within each Figure folder, all scripts should produce the corresponding figure panel or supplementary figure panel. (Note: That in demo mode, these scripts will only produce a subset of the figure, with data corresponding to ACC or in some cases, the single example ACC neuron.) Thus, this code provides methods for producing visualizations of choice-selective sequences (Figure 1), psychometric curves (Figure 3, 4, S1), tuning curves to evidence (Figure 3, 4, 5, S1, S5), simulated responses to optogenetic perturbations (Figure 3, 4, S1), average individual neuron activity at position x evidence bins (Figure 5), summary plots of the joint gaussian fits (Figure 5), results of linear encoding models (Figure 6), and population tuning curves (Figure 6).

A demonstration of the fitting procedure for the joint gaussian and linear evidence encoding model for a single neuron can be obtained by the calls:
'python NeuralDataFits/JointGuassianandTuningCurveFitting/get_joint_gaussian_fit.py ExampleData/exampleneuron.npy'
and
'python NeuralDataFits/LinearEncodingModel/linear_encoding_model.py ExampleData/exampleneuron.npy'
respectively. Results may be compared to the results included in ExampleData for neuron 153 from session dFF_tetO_8_07282021processedOutput.

Code for Figures 1, 5, 6, S1, and S5 should produce plots in less than 5 minutes. Producing the baseline model data simulations for 1000 trials may take up to 6 hours. After this data is obtained, Figure 3, 4, and S1 tuning curves and optogenetic simulations should take less than 15 minutes, but simulations for the psychometric curves with noisy inputs may again take up to 6 hours.

Fits of the joint gaussian and linear encoding model to individual cells should each take less than 10 minutes.

# Instructions for Use
Each folder contains the scripts necessary to reproduce each subpanel in the figure. To reproduce the full results, scripts should be run with the input files replaced with the appropriate data files, which will be made available via figshare.

Code to produce data panels in Figure S4 and Figure S5 is identical to that from the corresponding plots in the main figures (Figure 5CD and Figure 6LM, respectively) but run for different datasets, corresponding to either the model simulations or recordings from DMS. Panels in Figures S8-S10 are generated identically to the plots in Figure 5A, but for different example neurons.


