import copy
import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np

def FedAvg(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):

            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))

        if np.isnan(w_avg[key].item()): 
            print('w in fedavg: ', w, '   ******')
    return w_avg

def FedFairAvg(w, scores):
    """
    Returns the weighted avg of the weights.
    """
    
    # method 1: reweight based on scores
    sum = 0
    for each in scores:
        sum += each
    scores /= sum

    #method 2: reweight based on softmax of scores
    # scores = F.softmax(torch.FloatTensor(scores))


    w_avg = scores[0] * copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += scores[i] * w[i][key]
        
    return w_avg



