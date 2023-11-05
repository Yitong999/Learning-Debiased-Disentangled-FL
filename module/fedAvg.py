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

        if np.isnan(w_avg[key].cpu().numpy()).any():
            raise NameError('fedAvg')
            print('w in fedavg: ', w, '   ******')
    return w_avg

def FedWt_v1(w, scores):
    """
    Returns the weighted avg of the weights.
    """
    
    # reweight based on scores
    total = sum(scores)
    scores = [x / total for x in scores]
    print('scores in FedAVG: ', scores)


    w_avg = scores[0] * copy.deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = w_avg[key] * scores[0]
        for i in range(1, len(w)):
            w_avg[key] += scores[i] * w[i][key]
        
    return w_avg


def FedWt_v2(w, scores):
    """
    Returns the weighted avg of the weights.
    """
    
    # reweight based on scores
    exp_values = [math.exp(n) for n in scores]
    sum_of_exp_values = sum(exp_values)
    scores = [exp_value / sum_of_exp_values for exp_value in exp_values]
    print('scores in FedAVG: ', scores)

    w_avg = scores[0] * copy.deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = w_avg[key] * scores[0]
        for i in range(1, len(w)):
            w_avg[key] += scores[i] * w[i][key]
        
    return w_avg


