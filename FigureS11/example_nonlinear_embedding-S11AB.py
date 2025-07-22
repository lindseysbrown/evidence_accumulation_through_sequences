# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 15:06:44 2024

@author: lindseyb
"""

import numpy as np
import matplotlib.pyplot as plt
import cebra
from cebra import CEBRA
from sklearn.model_selection import train_test_split


def plot_2d(points, points_color, title, cmap, vmin, vmax):
    '''
    Function for plotting the points on the manifold
    '''
    fig, ax = plt.subplots(figsize=(3, 3), facecolor="white", constrained_layout=True)
    fig.suptitle(title, size=16)
    x, y = points.T
    col = ax.scatter(x, y, c=points_color, s=5, alpha=0.8, cmap = cmap, vmin = vmin, vmax = vmax)
    fig.colorbar(col, ax=ax, orientation="horizontal", shrink=0.6, aspect=60, pad=0.01)
    ax.set_title(title)
    plt.show()

#CEBRA settings
max_iterations = 10000
output_dimension = 32


#load data formatted for CEBRA
neural_data = cebra.load_data('ExampleData/CebraExampleData/nicFR_E65_20180202-neural.npy')
pos_data = cebra.load_data('ExampleData/CebraExampleData/nicFR_E65_20180202-pos.npy')
ev_data = cebra.load_data('ExampleData/CebraExampleData/nicFR_E65_20180202-ev.npy')
trial_data = cebra.load_data('ExampleData/CebraExampleData/nicFR_E65_20180202-trials.npy')

#split data into train and test set for a single fold
trials = list(set(list(trial_data)))
traintrials, testtrials = train_test_split(trials, test_size = .3, random_state = 42)

trainingidxs = [t in traintrials for t in trial_data]
testidxs = [not i for i in trainingidxs]
train_data = neural_data[trainingidxs]
valid_data = neural_data[testidxs]
train_label = ev_data[trainingidxs]
valid_label = ev_data[testidxs]

cebra_model = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=3e-4,
                        temperature_mode='auto',
                        output_dimension=2,
                        max_iterations=max_iterations,
                        distance='cosine',
                        conditional='time_delta',
                        device='cuda_if_available',
                        verbose=True,
                        time_offsets=10)


cebra_model.fit(train_data, train_label)

train_embedding = cebra_model.transform(train_data)
valid_embedding = cebra_model.transform(valid_data)

plt.figure()
plot_2d(train_embedding, train_label, 'Training Embedding', 'bwr', -10, 10)
plt.show()

plt.figure()
plot_2d(valid_embedding, valid_label, 'Testing Embedding', 'bwr', -10, 10)
plt.show()
