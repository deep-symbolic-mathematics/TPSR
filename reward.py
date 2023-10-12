import numpy as np
import re
import torch
import sys
import sympy as sp
import os
from symbolicregression.metrics import compute_metrics

def evaluate_metrics(y_gt, tree_gt, y_pred):
    metrics = []
    results_fit = compute_metrics(
        {
            "true": [y_gt],
            "predicted": [y_pred],
            "tree": tree_gt,
            "predicted_tree": tree_gt,
        },
        metrics='accuracy_l1',
    )
    for k, v in results_fit.items():
        metrics.append(v[0])
    
    return metrics

def compute_reward(params,samples, y_pred, model_str, generations_tree):  

    # NMSE
    penalty = -2
    if y_pred is None:
        reward = penalty
    else:
        y = samples['y_to_fit'][0].reshape(-1)
        eps = 1e-9
        NMSE = np.sqrt( np.mean((y - y_pred)**2) / (np.mean((y)**2)+eps) ) 

        if (not np.isnan(NMSE)):
            reward = 1/(1+NMSE)
            
        elif np.isnan(NMSE):
            reward = penalty

        if generations_tree != []:
            complexity = len(generations_tree[0].prefix().split(","))
            ### Length penalty
            lam = params.lam
            reward = reward + lam * np.exp(-complexity/200)
        
    return reward