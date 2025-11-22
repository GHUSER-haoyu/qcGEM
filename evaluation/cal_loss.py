from torch.nn import functional as F
import torch
from torch import nn
import numpy as np
from sklearn.metrics import roc_auc_score
import sys

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, outputs, targets):
        ce_loss = torch.nn.functional.cross_entropy(
            outputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

"""
The loss function for classification tasks.
"""
def Classification_loss(prediction, label, batch_num, label_num, c_loss = 'ce'):
    prediction = prediction.reshape(batch_num, label_num, 2).transpose(1,2)
    label = label.reshape(batch_num, label_num, 2).transpose(1,2)
    if c_loss == 'ce':
        loss_function = nn.CrossEntropyLoss(reduction='mean')
    elif c_loss == 'focal':
        loss_function = FocalLoss(alpha=0.25, gamma=2)
    else:
        pass

    loss_multi_term_average = loss_function(prediction, label)
    softmax = nn.Softmax(dim = 1)
    prediction_softmaxed = softmax(prediction).cpu().detach().numpy()
    prediction_acc = np.argmax(prediction_softmaxed, axis = 1)
    label_ = label.cpu().detach().numpy()
    label_acc = np.argmax(label_, axis = 1)
    MissLabel = label_.sum(axis = 1)
    miss_label_index = np.where(MissLabel.reshape(1,-1).squeeze() == 0)
    prediction_acc = prediction_acc.reshape(1,-1).squeeze()
    prediction_acc_filtered = np.delete(prediction_acc, miss_label_index)
    label_acc = label_acc.reshape(1,-1).squeeze()
    label_acc_filtered = np.delete(label_acc, miss_label_index)
    correct_num = np.sum(prediction_acc_filtered == label_acc_filtered) / label_num
    auc_prediction = prediction_softmaxed[:, 1, :]
    auc_label = label_[:, 1, :]


    return loss_multi_term_average, correct_num, auc_prediction, auc_label, MissLabel

"""
The loss function for regression tasks.
"""
def Regression_loss(prediction, label, batch_num, label_num, r_loss = 'rmse'):
    if r_loss == 'rmse':
        label = torch.unsqueeze(label, -1)
        loss_function = nn.MSELoss(reduction='mean')
        loss = torch.sqrt(loss_function(prediction, label))
        
    else:
        pass

    return loss

def calc_cliff_rmse(y_test_pred, y_test, cliff_mols_test):
    # Convert to numpy array if it is none
    y_test_pred = np.array(y_test_pred) if type(y_test_pred) is not np.array else y_test_pred
    y_test = np.array(y_test) if type(y_test) is not np.array else y_test

    # Get the index of the activity cliff molecules
    cliff_test_idx = [i for i, cliff in enumerate(cliff_mols_test) if cliff == 1]

    # Filter out only the predicted and true values of the activity cliff molecules
    y_pred_cliff_mols = y_test_pred[cliff_test_idx]
    y_test_cliff_mols = y_test[cliff_test_idx]

    return calc_rmse(y_pred_cliff_mols, y_test_cliff_mols)

def calc_rmse(true, pred):
    # Convert to 1-D numpy array if it's not
    if type(pred) is not np.array:
        pred = np.array(pred)
    if type(true) is not np.array:
        true = np.array(true)
    return np.sqrt(np.mean(np.square(true - pred)))

# 20250101