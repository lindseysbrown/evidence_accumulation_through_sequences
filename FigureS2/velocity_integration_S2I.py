# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 13:42:39 2021

@author: lindseyb
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Arial']})

#load trial data on which to test the model, lists of left and right cues for each trial
trialdata = pd.read_pickle('ExampleData/trialdata.pkl')

#initialize neural ring
neurons = 20
timepoints = 1 #single ring for all positions
Cring = np.zeros((neurons*timepoints,))

#set parameters
a = 1.1
w0 = 5

#set up synaptic weights
def phi(i): #map each neuron to an angle
    return i*2*np.pi/neurons

def F(x): #activation function
    return .5*(1+np.tanh(x))

W = np.zeros((neurons*timepoints, neurons*timepoints)) #synaptic connection matrix

for i in range(neurons):
    for j in range(neurons):
        W[i, j] = w0*(np.cos(phi(i)-phi(j))-.9) #define weights between different neurons with cosine connectivity



def I(t, C, v):
    '''
    calculate external input to the ring
    t: position
    C: current state of the ring
    v: velocity input
    Lcues: positions of left cues
    Rcues: positions of right cues
    '''
    if v(t)<0: 
        IL = np.roll(C, -1) #input to each neuron from its left neighbor
    else:
        IL = np.zeros((neurons*timepoints,)) #no backward velocity

    #same for right cues
    if v(t)>0:
        IR = np.roll(C, 1)
    else:
        IR = np.zeros((neurons*timepoints,)) #no forward velocity
    return v(t)*(IL+IR) #total velocity input

def correct(sol, Lcues, Rcues):
    '''
    returns whether the final solution gives the correct behavior for a given set of cues
    '''
    imax = np.argmax(sol[-1, :])
    if len(Lcues)>len(Rcues): #if there are more left cues than right cues, center of the peak should be left of the midpoint
        return imax<(neurons*timepoints-.5*neurons-1)
    return imax>(neurons*timepoints-.5*neurons)

def simulate(v):
    '''
    Simulate a single trial for a given velocity
    '''

    #reset simulation
    Cring = np.zeros((neurons*timepoints,))
    
    #initialize bump of activity at the center
    Cring[-1] = .6
    Cring[0] = .5
    Cring[-2] = .5
    
   
   #differential equation for the bump attractor
    def ring(y, t):
        dydt = -a*y+(F(W@y+.18*I(t, y, v)))
        return dydt
    
    y0 = Cring
    t = np.linspace(0, 330, 3301)
    
    #integrate the differential equation
    sol = odeint(ring, y0, t, hmax=5)
    return sol

for t in range(1):
    def v(t):
        #if t>40 and t<100: #can redefine velocity to show affects of speed increases on network activity
         #   return 1.5
        return 1
    sol = simulate(v)
    
plt.figure()
plt.plot(sol, color = 'k')
plt.xticks([0, 300, 1300, 2300], ['-30', '0', '100', '200'])
plt.xlim([0, 3301])
plt.ylabel('neural activity')
plt.xlabel('position')
plt.title('constant velocity')
plt.show()
