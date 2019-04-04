import scipy.io as scio
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import os

def read_data(filename):
    data = []
    for root, dir, files in os.walk(filename):
        for file in files:
            data_t = scio.loadmat(os.path.join(root, file))
            data_t = data_t['savedata']
            data_t = data_t.T
            data.append(data_t)
    return data


def data_transform(data_all, label):
    training_data = []
    label_list = []
    for data in data_all:
        frame_index = 0
        frame = []
        doppler = []
        snr = []
        dict = {}
        for i in range(len(data)):
            if (data[i] == np.array([0, 0, 0, 0])).all():
                frame_index = frame_index + 1
                list = [d for d in dict]
                doppler = doppler + list
                list = [dict[d] for d in dict]
                #归一化
                if list != []:
                    amin, amax = np.min(list), np.max(list)
                    if amax == amin:
                        list = [1] * len(list)
                    else:
                        list = (list - amin) / (amax - amin)
                        list = list.tolist()
                snr = snr + list
                dict = {}
                if frame_index % 20 == 0:  # one-second data
                    # make doppler, frame, snr a heatmap
                    matrix_heatmap = np.zeros((128, 20))
                    if doppler != []:   
                        doppler_index = np.array(np.round(np.array(doppler) / 0.0825), dtype=int)
                        matrix_heatmap[doppler_index + 64, np.array(frame)] = np.array(snr)

                    plt.figure()
                    sns.heatmap(matrix_heatmap, cmap='jet')      
                    plt.show()
                    training_data.append(matrix_heatmap)
                    label_list.append(label)
                    # clear doppler, snr, frame
                    doppler = []
                    snr = []
                    frame = []
                    frame_index = 0

            else:
                if dict.get(data[i][2]) != None:
                    dict[data[i][2]] = data[i][3] + dict[data[i][2]] 
                else:
                    dict[data[i][2]] = data[i][3]
                    frame.append(frame_index)

        list = [d for d in dict]
        doppler = doppler + list
        list = [dict[d] for d in dict]
        if list != []:
            amin, amax = np.min(list), np.max(list)
            if amax == amin:
                list = [1] * len(list)
            else:
                list = (list - amin) / (amax - amin)
                list = list.tolist()
        snr = snr + list
    return training_data, label_list

# plt.plot(snr)
# plt.show()
# norm = colors.Normalize(vmin=0, vmax=1.0)
# plt.scatter(frame, doppler, c=snr,norm=norm)
# plt.xlabel("frame")
# plt.ylabel("doppler")
# plt.colorbar()
# plt.show()

import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

# Device configuration
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

# Device configuration
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ConvNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.05))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.05))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.05))      
        
        self.layer4 = nn.Sequential(
            nn.Linear(4096, 128),
            nn.LeakyReLU(),
            nn.Dropout2d(0.05)            
        )
        self.outputlayer = nn.Linear(128, num_classes)
       
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.layer4(out)
        out = self.outputlayer(out)
        out = F.softmax(out, dim=1)
        return out