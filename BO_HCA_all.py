#!/usr/bin/env python
# coding: utf-8

import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
import gpytorch
import warnings
from matplotlib import pyplot as plt
from pyDOE import lhs
from acquisition_functions import EI, TS, UCB
import matplotlib.pyplot as plt
from scipy.stats import norm
from pyDOE import lhs
from scipy.spatial.distance import cdist

from test_functions import (
    Ackley,
    Branin,
    Eggholder,
    griewank,
    Griewank_f,
    Hartmann6,
    Langermann,
    Levy,
    Schwefel,
    StyblinskiTang,
    Rastrigin,
    Powell,
    PermDB,
    Rosenbrock,
    MLP_Diabete,
)

warnings.filterwarnings("ignore")

# 统一使用 double 精度
dtype = torch.double
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#############################################
# 模块1: 多输出GP模型与异构噪声 (来自multi_v3)
#############################################

# 创建异构噪声似然函数
class HeteroscedasticLikelihood(gpytorch.likelihoods.MultitaskGaussianLikelihood):
    def __init__(self, num_tasks=2, **kwargs):
        super().__init__(num_tasks=num_tasks, **kwargs)
        
    def forward(self, function_samples, **kwargs):
        # 从function_samples获取均值和协方差
        mean = function_samples.mean
        covar = function_samples.lazy_covariance_matrix
        
        # 如果提供了credit信息，则用于调整噪声
        if 'credits' in kwargs and kwargs['credits'] is not None:
            credits = kwargs['credits']
            
            # 防止除零，确保最小值
            credits = torch.clamp(credits, min=1e-6)
            
            # 获取当前的噪声项
            noise_diag = self.task_noises.expand(credits.shape[0], self.num_tasks)
            
            # 创建调整后的噪声矩阵 - 任务0（目标函数）的噪声受credit影响
            adjusted_noise = noise_diag.clone()
            # 增强credit对噪声的影响，平方关系使得高credit样本的噪声更低
            adjusted_noise[:, 0] = adjusted_noise[:, 0] / (credits * credits)
            
            # 使用调整后的噪声创建分布
            return gpytorch.distributions.MultitaskMultivariateNormal(
                mean, self._add_task_noises(covar, adjusted_noise)
            )
        
        # 如果没有提供credit，使用默认行为
        return super().forward(function_samples)

# 定义多任务 GP 模型
class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks=2):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.num_tasks = num_tasks
        # 多任务均值
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        )
        # 多任务核
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=num_tasks, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

# 训练多任务 GP 模型
def train_gp_model(train_x, train_y, training_iter=50, lr=0.1):
    """
    train_x: torch.tensor, shape (N, D), dtype=torch.double
    train_y: torch.tensor, shape (N, 2), dtype=torch.double
    """
    # 使用自定义的异构噪声似然函数
    likelihood = HeteroscedasticLikelihood(num_tasks=2)
    model = MultitaskGPModel(train_x, train_y, likelihood, num_tasks=2)

    # 统一到指定 device 和 dtype
    model = model.to(device=device, dtype=dtype)
    likelihood = likelihood.to(device=device, dtype=dtype)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

    return model, likelihood

# 利用训练好的模型做预测
def gp_predict(model, likelihood, X, credits=None):
    """
    X: numpy 数组, shape (N, D)
    credits: numpy 数组, shape (N,) 或 None - 用于调整预测噪声
    返回:
      mu: shape (N, 2) —— 每列对应一个任务的预测均值
      sigma: shape (N, 2) —— 每列对应一个任务的预测标准差
    """
    model.eval()
    likelihood.eval()

    test_x = torch.tensor(X, dtype=dtype, device=device)
    
    # 将credits转为tensor（如果提供）
    credits_tensor = None
    if credits is not None:
        credits_tensor = torch.tensor(credits, dtype=dtype, device=device)
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # 使用model获取潜在函数分布
        f_preds = model(test_x)
        # 将credits传递给似然函数以调整噪声
        preds = likelihood(f_preds, credits=credits_tensor)
        
    mu = preds.mean.detach().cpu().numpy()          # shape (N, 2)
    sigma = np.sqrt(preds.variance.detach().cpu().numpy())  # shape (N, 2)
    return mu, sigma

# 更新数据
def update_data(x_new, y_new, credit_new, D_old, Y_old):
    """
    x_new: numpy 数组，形状 (1, D)
    y_new: 标量，目标函数值
    credit_new: 标量，credit 值
    D_old: numpy 数组，形状 (N, D)
    Y_old: numpy 数组，形状 (N, 2)
    """
    D_new = np.atleast_2d(x_new)
    y_new = np.atleast_2d(y_new)            # (1, 1)
    credit_new = np.atleast_2d(credit_new)  # (1, 1)
    new_Y = np.hstack([y_new, credit_new])  # (1, 2)
    D_update = np.concatenate((D_old, D_new), axis=0)
    Y_update = np.concatenate((Y_old, new_Y), axis=0)
    return D_update, Y_update

#############################################
# 模块2: 基于credit的候选点生成 (来自Sample)
#############################################

def generate_candidates_with_credit_bias(n_candidate, dimension_x, lb, ub, X_history=None, credits=None, 
                                         uniform_ratio=0.5, sigma_scale=0.1, 
                                         smooth_power=0.5, min_prob=0.05, 
                                         use_repulsion=True, repulsion_strength=1.0):
    """
    生成候选点，对credit高的区域进行更密集的采样
    """
    # 如果没有历史数据或credits，回退到均匀采样
    if X_history is None or credits is None or len(X_history) == 0:
        X_uniform = np.random.rand(n_candidate, dimension_x)
        for dim_i in range(dimension_x):
            X_uniform[:, dim_i] = X_uniform[:, dim_i] * (ub[dim_i] - lb[dim_i]) + lb[dim_i]
        return X_uniform
    
    # 确保credits是一维数组
    if len(credits.shape) > 1:
        credits = credits.flatten()
    
    # 计算均匀采样和基于credit采样的点数
    n_uniform = int(n_candidate * uniform_ratio)
    n_biased = n_candidate - n_uniform
    
    # 生成均匀采样的点
    X_uniform = np.random.rand(n_uniform, dimension_x)
    for dim_i in range(dimension_x):
        X_uniform[:, dim_i] = X_uniform[:, dim_i] * (ub[dim_i] - lb[dim_i]) + lb[dim_i]
    
    # 如果没有足够的非均匀采样点，直接返回均匀采样结果
    if n_biased <= 0:
        return X_uniform
    
    # 对credit分布进行平滑处理
    credits_positive = np.maximum(credits, 1e-10)
    smoothed_credits = np.power(credits_positive, smooth_power)
    
    # 设置最小概率阈值
    if min_prob > 0:
        min_val = min_prob * np.max(smoothed_credits)
        smoothed_credits = np.maximum(smoothed_credits, min_val)
    
    # 归一化为概率分布
    if np.sum(smoothed_credits) <= 1e-8:
        normalized_credits = np.ones_like(smoothed_credits) / len(smoothed_credits)
    else:
        normalized_credits = smoothed_credits / np.sum(smoothed_credits)
        normalized_credits = normalized_credits / np.sum(normalized_credits)
    
    # 区域排斥机制
    if use_repulsion and len(X_history) >= 2:
        dist_matrix = cdist(X_history, X_history)
        max_dist = np.max(dist_matrix)
        if max_dist > 0:
            rel_dist = dist_matrix / max_dist
            repulsion = 1.0 - rel_dist
            np.fill_diagonal(repulsion, 0)
            credit_mat = normalized_credits.reshape(-1, 1)
            total_repulsion = np.sum(repulsion * credit_mat.T, axis=1)
            repulsion_factor = 1.0 / (1.0 + repulsion_strength * total_repulsion)
            normalized_credits = normalized_credits * repulsion_factor
            if np.sum(normalized_credits) > 1e-8:
                normalized_credits = normalized_credits / np.sum(normalized_credits)
    
    # 选择参考点
    selected_indices = np.random.choice(
        np.arange(len(X_history)), 
        size=n_biased,
        p=normalized_credits,
        replace=True
    )
    
    # 高效邻近点生成
    reference_points = X_history[selected_indices]
    adaptive_sigma_scale = sigma_scale / np.sqrt(max(1, dimension_x / 10))
    sigma_vector = adaptive_sigma_scale * np.minimum(ub - lb, np.ones(dimension_x))
    
    # 根据维度选择生成方式
    if dimension_x > 50:
        mask = np.random.rand(n_biased, dimension_x) < 0.3
        zero_rows = np.sum(mask, axis=1) == 0
        if np.any(zero_rows):
            rand_dims = np.random.randint(0, dimension_x, size=np.sum(zero_rows))
            mask[zero_rows, rand_dims] = True
            
        noise = np.zeros((n_biased, dimension_x))
        noise[mask] = np.random.normal(0, 1, size=np.sum(mask))
        
        for d in range(dimension_x):
            noise[:, d] = noise[:, d] * sigma_vector[d]
    else:
        noise = np.random.normal(0, 1, size=(n_biased, dimension_x))
        for d in range(dimension_x):
            noise[:, d] = noise[:, d] * sigma_vector[d]
    
    # 生成新点 = 参考点 + 噪声
    X_biased = reference_points + noise
    
    # 确保点在边界内
    X_biased = np.maximum(X_biased, lb)
    X_biased = np.minimum(X_biased, ub)
    
    # 合并均匀采样和基于credit的采样结果
    X_candidates = np.concatenate([X_uniform, X_biased], axis=0)
    
    return X_candidates

#############################################
# 模块3: Credit加权的采集函数 (来自AC)
#############################################

def get_knn_credits(X_candidates, X_history, credits, k=5):
    """
    使用K近邻方法为候选点分配credit值
    """
    if len(credits.shape) > 1:
        credits = credits.flatten()
    
    dist_matrix = cdist(X_candidates, X_history)
    k = min(k, len(X_history))
    nearest_indices = np.argpartition(dist_matrix, k, axis=1)[:, :k]
    
    candidate_credits = np.zeros(X_candidates.shape[0])
    for i in range(X_candidates.shape[0]):
        candidate_credits[i] = np.mean(credits[nearest_indices[i]])
    
    return candidate_credits

def EI_credit(mu, sigma, X, y_best, credits=None, X_history=None):
    """基于Credit加权的期望改进(EI)采集函数"""
    mu = mu.reshape(-1, 1)
    sigma = sigma.reshape(-1, 1)
    gamma = (y_best - mu) / (sigma + 1e-9)
    ei_values = (y_best - mu) * norm.cdf(gamma) + sigma * norm.pdf(gamma)
    
    if credits is not None and X_history is not None:
        candidate_credits = get_knn_credits(X, X_history, credits)
        candidate_credits = candidate_credits.reshape(-1, 1)
        ei_values = ei_values * candidate_credits
    
    x_new = X[np.argmax(ei_values), :]
    return x_new

def UCB_credit(mu, sigma, X, const=2.0, credits=None, X_history=None):
    """基于Credit加权的上置信界(UCB)采集函数"""
    mu = mu.reshape(-1, 1)
    sigma = sigma.reshape(-1, 1)
    
    if credits is not None and X_history is not None:
        candidate_credits = get_knn_credits(X, X_history, credits)
        candidate_credits = candidate_credits.reshape(-1, 1)
        exploration_term = const * sigma * candidate_credits
        ucb_values = mu - exploration_term
    else:
        ucb_values = mu - const * sigma
    
    x_new = X[np.argmin(ucb_values), :]
    return x_new

def TS_credit(mu, sigma, X, credits=None, X_history=None, n_points=5):
    """基于Credit加权的Thompson采样(TS)采集函数"""
    mu = mu.reshape(-1, 1)
    sigma = sigma.reshape(-1, 1)
    eps = 1e-9
    std_clipped = np.maximum(sigma, eps)
    
    if credits is not None and X_history is not None:
        candidate_credits = get_knn_credits(X, X_history, credits)
        candidate_credits = candidate_credits.reshape(-1, 1)
        std_clipped = std_clipped * np.sqrt(candidate_credits)
    
    y_samples = []
    for _ in range(n_points):
        y_sample = np.random.normal(loc=mu, scale=std_clipped)
        y_samples.append(y_sample)
    all_samples = np.hstack(y_samples)
    
    min_index = np.argmin(all_samples.mean(axis=1))
    return X[min_index]

#############################################
# 主实验函数：结合三个模块
#############################################
def Experiments(repet_time):
    print("重复实验:", repet_time + 1)
    np.random.seed(repet_time)
    start_all = time.time()
    date = args.time
    heter = args.heter
    dy = args.noise_std

    func = args.function
    select = args.select_method
    dimension_x = args.dimension_x  
    n_sample = args.n_sample  
    iteration = args.iteration 
    algorithm = args.algorithm
    Number_Z = args.n_Z
    print("算法:", algorithm)

    # 选择实验函数
    if func == 'MLP_Diabete':
        dimension_x = 4
        f = MLP_Diabete(dimension_x, dimension_x)
    elif func == "Hart6":
        dimension_x = 6
        f = Hartmann6(dimension_x, dimension_x, dy)
    elif func == "Ackley":
        dimension_x = 16
        f = Ackley(dimension_x, dimension_x, dy)
    elif func == "Griewank":
        dimension_x = 50
        f = Griewank_f(dimension_x, dimension_x, dy)
    elif func == "Levy":
        dimension_x = 20
        f = Levy(dimension_x, dimension_x, dy)
    elif func == "Schwefel":
        dimension_x = 2
        f = Schwefel(dimension_x, dimension_x, dy)
    elif func == "Eggholder":
        dimension_x = 2
        f = Eggholder(dimension_x, dimension_x, dy)
    elif func == "Branin":
        dimension_x = 2
        f = Branin(dimension_x, dimension_x, dy)
    elif func == "Langermann":
        dimension_x = 20
        f = Langermann(dimension_x, dimension_x, dy)
    elif func == "StyblinskiTang":
        dimension_x = 50
        f = StyblinskiTang(dimension_x, dimension_x, dy)
    elif func == "Rastrigin":
        dimension_x = 100
        f = Rastrigin(dimension_x, dimension_x, dy)        
    elif func == "Powell":
        dimension_x = 50
        f = Powell(dimension_x, dimension_x, dy)     
    elif func == "PermDB":
        dimension_x = 50
        f = PermDB(dimension_x, dimension_x, dy) 
    elif func == "Rosenbrock":
        dimension_x = 50
        f = Rosenbrock(dimension_x, dimension_x, dy) 
    else:
        f = Ackley(dimension_x, dimension_x, dy)

    x_star = f.x_star
    f_star = f.f_star
    print("最优值 f_star:", f_star)

    # 设置采样总数
    N_total = args.n_sample + args.iteration

    # 生成初始候选点（均匀采样）
    n_candidate = 10000
    a = f.lb
    b = f.ub
    X_ = np.random.rand(n_candidate, dimension_x)
    for dim_i in range(dimension_x):
        X_[:, dim_i] = X_[:, dim_i] * (b[dim_i] - a[dim_i]) + a[dim_i]

    # 初始拉丁超立方采样
    lhd = lhs(dimension_x, samples=n_sample, criterion="maximin")
    D1 = np.zeros((n_sample, dimension_x))
    Y1 = np.zeros((n_sample, 1))
    for dim_i in range(dimension_x):
        D1[:, dim_i] = lhd[:, dim_i] * (b[dim_i] - a[dim_i]) + a[dim_i]
    for i in range(n_sample):
        Y1[i, :] = f(D1[i, :])
    
    # 初始样本的 credit 均设为 1
    credit_init = np.ones((n_sample, 1))
    Y1 = np.hstack([Y1, credit_init])  # shape (n_sample, 2)

    # 转为 torch.tensor 用于 gpytorch 训练
    train_x = torch.tensor(D1, dtype=dtype, device=device)
    train_y = torch.tensor(Y1, dtype=dtype, device=device)

    # 初始化多输出 GP 模型
    model, likelihood = train_gp_model(train_x, train_y, training_iter=50, lr=0.1)
    
    # 使用当前credit进行预测
    credits = Y1[:, 1]
    mu_pred, sigma_pred = gp_predict(model, likelihood, X_, credits=None)  # 初始预测时没有credits
    # 仅取任务0（目标函数）的预测值
    mu_g = mu_pred[:, 0]
    sigma_g = sigma_pred[:, 0]

    # 初始化 minimizer 列表
    if dy == 0:
        minimizer = [D1[np.argmin(Y1[:, 0]), :]]
    else:
        mu_initial, _ = gp_predict(model, likelihood, D1, credits=Y1[:, 1])
        minimizer = [D1[np.argmin(mu_initial[:, 0]), :]]

    TIME_RE = []
    D_update = D1.copy()
    Y_update = Y1.copy()

    # 将历史采样点去重存放在 X_update 中
    X_update = np.unique(D_update, axis=0)

    for i in range(iteration):
        if i % 20 == 0:
            print("迭代次数", i + 1)
            # 输出模型部分参数信息以便调试
            state_dict_small = {
                k: v.detach().cpu().numpy()
                for k, v in model.state_dict().items()
                if v.numel() < 10
            }
            print("GP模型参数:", state_dict_small)

        start = time.time()

        # =============== (1) 基于credit生成候选点 ===============
        if heter and i > 0:  # 第一轮迭代还没有credit信息，从第二轮开始使用非均匀采样
            # 自适应参数调整
            sigma_scale = max(0.05, 0.2 * (1 - i / iteration))
            uniform_ratio = max(0.2, 0.5 * (1 - i / (2 * iteration)))
            smooth_power = max(0.3, 0.5 * (1 - i / iteration))
            repulsion_strength = min(2.0, 1.0 + i / (iteration * 0.5))
            
            X_ = generate_candidates_with_credit_bias(
                n_candidate=10000,
                dimension_x=dimension_x,
                lb=f.lb,
                ub=f.ub,
                X_history=D_update,
                credits=Y_update[:, 1],  # 使用多输出GP中的credit列
                uniform_ratio=uniform_ratio,
                sigma_scale=sigma_scale,
                smooth_power=smooth_power,
                min_prob=0.05,
                use_repulsion=True,
                repulsion_strength=repulsion_strength
            )
            
            # 对新生成的候选点进行预测
            mu_pred, sigma_pred = gp_predict(model, likelihood, X_, credits=Y_update[:, 1])
            mu_g, sigma_g = mu_pred[:, 0], sigma_pred[:, 0]  # 只取目标函数的预测

        # =============== (2) 采集函数 - 使用Credit加权 ===============
        if algorithm == "EI":
            y_best = np.min(Y_update[:, 0])  # 最小化问题
            x_new = EI_credit(mu_g, sigma_g, X_, y_best, 
                          credits=Y_update[:, 1] if heter else None,
                          X_history=D_update if heter else None)
        elif algorithm == "TS":
            # 使用已生成的非均匀候选点，或重新生成均匀点
            if heter and i > 0:
                X_temp = X_
            else:
                X_temp = np.random.rand(10000, dimension_x)
                for dim_i in range(dimension_x):
                    X_temp[:, dim_i] = X_temp[:, dim_i] * (b[dim_i] - a[dim_i]) + a[dim_i]
            
            # 获取预测
            mu_pred_temp, sigma_pred_temp = gp_predict(model, likelihood, X_temp, credits=Y_update[:, 1])
            
            # 使用Credit加权的TS
            x_new = TS_credit(mu_pred_temp[:, 0], sigma_pred_temp[:, 0], X_temp, 
                          credits=Y_update[:, 1] if heter else None,
                          X_history=D_update if heter else None, 
                          n_points=5)
        elif algorithm == "UCB":
            print("选择 UCB")
            # 使用Credit加权的UCB - 增大探索系数
            x_new = UCB_credit(mu_g, sigma_g, X_, const=3.5, 
                          credits=Y_update[:, 1] if heter else None,
                          X_history=D_update if heter else None)
        else:
            # 默认使用 EI
            y_best = np.min(Y_update[:, 0])  # 最小化问题
            x_new = EI_credit(mu_g, sigma_g, X_, y_best, 
                         credits=Y_update[:, 1] if heter else None,
                         X_history=D_update if heter else None)

        # =============== (3) 评估新点，更新数据 ===============
        y_new = f(x_new)  # 目标函数值
        # 新采样时，credit 初始设为1；在下面会更新
        credit_new = 1.0
        D_update, Y_update = update_data(x_new, y_new, credit_new, D_update, Y_update)
        
        # 使用 np.unique 去重
        X_update = np.unique(D_update, axis=0)

        # =============== (4) 计算 credit 并更新 ===============
        if heter:
            # 对历史数据做预测（仅对任务0）- 使用当前credit
            current_credits = Y_update[:, 1]
            mu_hist, sigma_hist = gp_predict(model, likelihood, D_update, credits=current_credits)
            mu_hist_task0 = mu_hist[:, 0]
            sigma_hist_task0 = sigma_hist[:, 0]
            Z_val = np.min(Y_update[:, 0])  # 使用最小值作为目标值
            eps = 1e-6
            h_z_vals = norm.pdf(Z_val, loc=mu_hist_task0, scale=sigma_hist_task0 + eps)
            importance_ratios = np.ones_like(h_z_vals)
            
            # 计算新采样点的credit值
            if D_update.shape[0] > n_sample:
                new_indices = np.arange(n_sample, D_update.shape[0])
                new_h_z = h_z_vals[new_indices]
                # 假设新采样点原先的先验概率 ~ 1 / 新采样数
                new_pi_vals = np.full(new_h_z.shape, 1.0 / len(new_h_z))
                new_importance = new_h_z / new_pi_vals
                # 扩大credit值的范围，增强异构噪声影响
                new_importance = np.clip(new_importance, 5e-2, 2e1)
                importance_ratios[new_indices] = new_importance
            
            # 更新所有数据点的credit（第二列）
            Y_update[:, 1] = importance_ratios
            
            # 定期打印credit值统计信息
            if i % 25 == 0:
                print(f"Credit值统计: 最小值={Y_update[:, 1].min():.4f}, 最大值={Y_update[:, 1].max():.4f}, 均值={Y_update[:, 1].mean():.4f}")
                print(f"Credit值分位数: 25%={np.percentile(Y_update[:, 1], 25):.4f}, 50%={np.percentile(Y_update[:, 1], 50):.4f}, 75%={np.percentile(Y_update[:, 1], 75):.4f}")

        # =============== (5) 重新训练多输出 GP 模型 ===============
        train_x = torch.tensor(D_update, dtype=dtype, device=device)
        train_y = torch.tensor(Y_update, dtype=dtype, device=device)
        model, likelihood = train_gp_model(train_x, train_y, training_iter=50, lr=0.1)

        # 对候选点重新预测
        mu_pred, sigma_pred = gp_predict(model, likelihood, X_, credits=Y_update[:, 1])
        mu_g = mu_pred[:, 0]
        sigma_g = sigma_pred[:, 0]

        # =============== (6) 更新 minimizer 列表 ===============
        if dy == 0:
            minimizer.append(D_update[np.argmin(Y_update[:, 0]), :])
        else:
            mu_train, _ = gp_predict(model, likelihood, D_update, credits=Y_update[:, 1])
            minimizer.append(D_update[np.argmin(mu_train[:, 0]), :])

        Training_time = time.time() - start
        print(f"迭代 {i + 1} 用时 {Training_time:.2f} 秒。")
        TIME_RE.append(Training_time)

    end_all = time.time() - start_all
    print(f"******* 总共用时 {end_all:.2f} 秒。*******")
    TIME_RE = np.array(TIME_RE).reshape(1, -1)

    # =============== 评估收敛表现 ===============
    minimum2 = []
    minimizer2 = []
    for i in range(Y_update.shape[0]):
        minimum2.append(np.min(Y_update[: (i + 1), 0]))  # 最小化问题：取最小目标值
        minimizer2.append(D_update[np.argmin(Y_update[: (i + 1), 0]), :])
    minimizer2 = np.array(minimizer2)
    minimum2 = np.array(minimum2)

    minimizer_all = np.array(minimizer)
    if n_sample - 1 > 0:
        minimizer_all = np.concatenate((minimizer2[: (n_sample - 1), :], minimizer_all), axis=0)
    print("minimizer_all shape:", minimizer_all.shape)

    minimum = np.zeros(N_total)
    if dy == 0:
        minimum = minimum2
    else:
        for index in range(N_total):
            minimum[index] = f.evaluate_true(minimizer_all[index, :])
    minimum = minimum.reshape(-1, 1)
    GAP = np.abs(minimum - f_star).reshape(-1, 1)

    if x_star is not None:
        # 计算与最优解的距离 gap
        xGAP = np.min(np.linalg.norm(minimizer_all - np.array(x_star), axis=2), axis=0).reshape(-1, 1)

    SimRegret = []
    for i in range(len(GAP)):
        SimRegret.append(np.min(GAP[: (i + 1)]))
    SimRegret = np.array(SimRegret).reshape(-1, 1)

    Regret = np.zeros(N_total)
    for index in range(N_total):
        Regret[index] = f.evaluate_true(D_update[index, :]) - f_star
    CumRegret = np.cumsum(Regret)
    AvgRegret = np.cumsum(Regret) / np.arange(1, N_total + 1)
    Regret = Regret.reshape(-1, 1)
    CumRegret = CumRegret.reshape(-1, 1)
    AvgRegret = AvgRegret.reshape(-1, 1)

    true_values = np.array([f.evaluate_true(d) for d in D_update])
    rmse = np.sqrt(np.mean((f_star - true_values) ** 2))
    print(f"RMSE: {rmse:.4f}")

    # 创建结果目录
    if not os.path.exists("Results"):
        os.makedirs("Results")
    if not os.path.exists("Images"):
        os.makedirs("Images")

    Budget = range(0, N_total)
    file = "_".join([
        date, str(Number_Z), select, func, algorithm,
        str(dy), str(n_sample), str(iteration), str(dimension_x)
    ])
    imagefile = "Images/" + file
    file = "Results/" + file

    # 绘制结果图
    plt.figure(figsize=(15, 6))
    subplot_configs = [
        (1, "CumRegret", CumRegret),
        (2, "AvgRegret", AvgRegret),
        (3, "SimRegret", SimRegret),
        (4, "GAP", GAP)
    ]
    for pos, ylabel, data in subplot_configs:
        plt.subplot(2, 3, pos)
        plt.plot(Budget, data)
        plt.ylabel(ylabel)
        plt.xlabel("Budget")
        plt.axhline(0, color="black", ls="--")

    if x_star is not None:
        plt.subplot(2, 3, 5)
        plt.plot(Budget, xGAP)
        plt.ylabel("xGAP")
        plt.xlabel("Budget")
        plt.axhline(0, color="black", ls="--")

    plt.savefig(f"{imagefile}_{repet_time}.pdf", format="pdf")
    plt.close()

    # 保存结果数据
    TIME_RE = np.array(TIME_RE).reshape(1, -1)
    file_configs = [
        ("_X_update.csv", X_update, False),
        ("_D_update.csv", D_update, False),
        ("_Y_update.csv", Y_update, True),  # 包含目标值和credit
        ("_minimizer_all.csv", minimizer_all, False),
        ("_minimum.csv", minimum, True),
        ("_Time.csv", TIME_RE, False),
        ("_CumRegret.csv", CumRegret, True),
        ("_Regret.csv", Regret, True),
        ("_AvgRegret.csv", AvgRegret, True),
        ("_SimRegret.csv", SimRegret, True),
        ("_GAP.csv", GAP, True),
        ("_RMSE.csv", np.array([[rmse]]), False)
    ]
    if x_star is not None:
        file_configs.append(("_xGAP.csv", xGAP, True))

    for suffix, data, needs_transpose in file_configs:
        with open(f"{file}{suffix}", "ab") as f_out:
            output_data = data.T if needs_transpose else data
            np.savetxt(f_out, output_data, delimiter=",")

#############################################
# 参数解析及主函数入口
#############################################
parser = argparse.ArgumentParser(description='综合HCA-贝叶斯优化方法 (多输出GP+非均匀采样+Credit加权)')
parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")
parser.add_argument('-t', '--time', type=str, help='实验日期', default='HCA_ALL_v1')
parser.add_argument('-select', '--select_method', type=str, help='选择方法', default="No_select") 
parser.add_argument('-func', '--function', type=str, help='测试函数', default='Hart6')  
parser.add_argument('-algo', '--algorithm', type=str, help='算法: EI/TS/UCB', default="UCB")
parser.add_argument('-dimensionx', '--dimension_x', type=int, help='问题维度', default=6)
parser.add_argument('-noise', '--noise_std', type=float, help='噪声标准差', default=0.1)
parser.add_argument('-heter', '--heter', help='是否使用异构噪声', action='store_true', default=True)
parser.add_argument('-nsample', '--n_sample', type=int, help='初始样本数', default=20)
parser.add_argument('-i', '--iteration', type=int, help='迭代次数', default=400)
parser.add_argument('-macro', '--repet_num', type=int, help='重复实验次数', default=100)
parser.add_argument('-smacro', '--start_num', type=int, help='重复实验起始编号', default=0)
parser.add_argument('-core', '--n_cores', type=int, help='CPU核心数', default=2)
parser.add_argument('-thre', '--threshold', type=float, help='阈值', default=1e-4)
parser.add_argument('-Z', '--n_Z', type=int, help='超参数Z', default=4)
parser.add_argument('-k', '--knn', type=int, help='KNN中的K值', default=5)

args = parser.parse_args()

def run_experiment(experiment_id):
    print(f"开始实验 {experiment_id}")
    Experiments(experiment_id)
    print(f"实验 {experiment_id} 完成")

def run_experiments(start, end):
    for i in range(start, end + 1):
        run_experiment(i)

if __name__ == "__main__":
    start_experiment = args.start_num
    end_experiment = args.start_num + args.repet_num - 1
    run_experiments(start_experiment, end_experiment)