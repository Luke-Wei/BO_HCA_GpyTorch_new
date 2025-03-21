import numpy as np
import pandas as pd
import scipy.stats  
from pyDOE import *
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF,Matern,WhiteKernel,ConstantKernel
import os
from sklearn import decomposition
from scipy.stats import norm
import torch
import os
import pandas as pd
from matplotlib import pyplot as plt
import scipy.stats
import math
from pyDOE import *
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF,Matern,WhiteKernel,ConstantKernel
from sklearn import decomposition
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import gpytorch

dtype = torch.double
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def EI(mu, sigma, X_, y_best):
    mu = mu.reshape(-1, 1)
    sigma = sigma.reshape(-1, 1)
    improvement = y_best - mu
    with np.errstate(divide='warn'):
        Z = improvement / sigma
    ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
    X_new = X_[np.argmax(ei)]
    return X_new


# def TS(X_, gp, n_points=1, eps=1e-9):
#     mu, std = gp.predict(X_, return_std=True)
#     mu = mu.reshape(-1, 1)
#     std = std.reshape(-1, 1)
#     std_clipped = np.maximum(std, eps)
#     y_samples = []
#     for _ in range(n_points):
#         y_sample = np.random.normal(loc=mu, scale=std_clipped)
#         y_samples.append(y_sample)
#     all_samples = np.hstack(y_samples)  
#     min_index = np.argmin(all_samples.mean(axis=1))  
#     X_new = X_[min_index]
#     return X_new

# ... 其他代码保持不变 ...

def gp_predict(model, likelihood, X):
    """
    用 gpytorch 的单输出模型对 X 做预测，返回 (均值, 标准差)
    其中均值/标准差 shape = (N,)
    """
    model.eval()
    likelihood.eval()
    # 转成 torch tensor
    test_x = torch.tensor(X, dtype=dtype, device=device)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        preds = likelihood(model(test_x))
    mu = preds.mean.detach().cpu().numpy()        # shape (N,)
    sigma = np.sqrt(preds.variance.detach().cpu().numpy())  # shape (N,)
    return mu, sigma

def TS(X_, model, likelihood, n_points=1, eps=1e-9):
    """
    Thompson Sampling for gpytorch-based single output model
    """
    # 获取预测均值和标准差，现在是一维数组
    mu, std = gp_predict(model, likelihood, X_)
    
    # 重塑为列向量
    mu = mu.reshape(-1, 1)       # shape (N,1)
    std = std.reshape(-1, 1)     # shape (N,1)
    std_clipped = np.maximum(std, eps)

    # 重复采样 n_points 次
    y_samples = []
    for _ in range(n_points):
        y_sample = np.random.normal(loc=mu, scale=std_clipped)  # shape (N,1)
        y_samples.append(y_sample)
    all_samples = np.hstack(y_samples)  # shape (N, n_points)

    # 最小化目标函数的 Thompson Sampling
    min_index = np.argmin(all_samples.mean(axis=1))
    X_new = X_[min_index]  # shape (D,)

    return X_new

# ... 其他代码保持不变 ...




def UCB(mu, sigma, X_, const=2.5):
    mu = mu.reshape(-1, 1)
    sigma = sigma.reshape(-1, 1)
    ucb_line = mu - const * sigma
    X_new = X_[np.argmin(ucb_line)]
    return X_new



