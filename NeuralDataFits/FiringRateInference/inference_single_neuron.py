import numpy as np
from spike_tools import data_preprocessing, estimate_spikes #download spike_tools package from https://github.com/jewellsean/FastLZeroSpikeInference
from matplotlib import pyplot as plt
from scipy.io import savemat
from scipy.io import loadmat
import pickle
import sys

""" 
parse input arguments:
[1] input directory where suite2p output lives
[2] cell id
[3] optional output directory, default to input directory
"""

arguments=sys.argv[1:]

if len(arguments) < 2:
	print("not enough arguments")
	sys.exit()
indir = arguments[0]
cellid = arguments[1]
outdir = indir
if len(arguments)>2:
	outdir = arguments[2]

# SETTINGS FOR INFERENCE
fps_in = 30
# decay_rate = 0.9704588 for GCaMP6f
decay_rate = 0.9885 #for GCaMP6s
target_firing_rate = 2 #goal firing rate

# LOAD RELEVANT FILES
print("Loading files")

# LOAD PREPROCESSED (BASELINE-CORRECTED) MAT FILES
wd=indir
mat = loadmat('%s/suite2p_processed.mat'%wd)
dF_index=np.where(np.array(mat['Output'].dtype.names)=="dF_bc")[0][0]
F=mat['Output'][0][0][dF_index]

# ISOLATE CELLS OF INTEREST
c = int(cellid)
dF = F[c]

# START WORKING
print("Starting inference")
data = {'calcium': dF,'fps': fps_in}
data = data_preprocessing.preprocess(data, factor = 2, low_perc = 1, up_perc = 80)
out = estimate_spikes.estimate_spikes_by_firing_rate(data, decay_rate, target_firing_rate)

# SAVE DATA AS MAT FILE
c = c + 1
savemat("%s/spikedeconv_cell_%04d.mat"%(outdir,c),out)
print("Completed inference on cell %d"%c)