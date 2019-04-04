from micro_doppler_2 import data_transform
import scipy.io as scio
import numpy as np 

data_neg = []
data_pos = []

data_pos = []
data_t = scio.loadmat('./data/swing hand/4800.mat')
data_t = data_t['savedata']
data_t = data_t.T
data_pos.append(data_t)

data_transform(data_pos, 0)

