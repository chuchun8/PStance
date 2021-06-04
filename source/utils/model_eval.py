import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


# Evaluation
def compute_f1(preds, y):
    
    rounded_preds = F.softmax(preds)
    _, indices = torch.max(rounded_preds, 1)
                
    correct = (indices == y).float()
    acc = correct.sum()/len(correct) # compute accuracy
    
    y_pred = np.array(indices.cpu().numpy())
    y_true = np.array(y.cpu().numpy())
    result = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0,1])
    f1_average = (result[2][0]+result[2][1])/2 # average F1 score of Favor and Against
        
    return acc, f1_average, result[0], result[1]