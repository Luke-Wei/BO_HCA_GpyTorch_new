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
# 定义标准GP模型（单输出，目标函数）
#############################################
class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        # 使用标准均值：常量均值
        self.mean_module = gpytorch.means.ConstantMean()
        # 使用标准核函数：RBF核
        self.covar_module = gpytorch.kernels.RBFKernel()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

#############################################
# 训练GP模型
#############################################
def train_gp_model(train_x, train_y, training_iter=50, lr=0.1):
    """
    train_x: torch.tensor, shape (N, D), dtype=torch.double
    train_y: torch.tensor, shape (N,), dtype=torch.double - 单输出目标函数值
    """
    # 使用标准高斯似然
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GPModel(train_x, train_y, likelihood)

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

#############################################
# 利用训练好的模型做预测，返回均值和标准差
#############################################
def gp_predict(model, likelihood, X):
    """
    X: numpy 数组, shape (N, D)
    返回:
      mu: shape (N,) —— 预测均值
      sigma: shape (N,) —— 预测标准差
    """
    model.eval()
    likelihood.eval()

    test_x = torch.tensor(X, dtype=dtype, device=device)
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # 使用model获取预测分布
        f_preds = model(test_x)
        # 通过似然函数获取观测值分布
        preds = likelihood(f_preds)
        
    mu = preds.mean.detach().cpu().numpy()        # shape (N,)
    sigma = preds.stddev.detach().cpu().numpy()   # shape (N,)
    return mu, sigma

#############################################
# 辅助函数：K近邻credit传递
#############################################
def get_knn_credits(X_candidates, X_history, credits, k=5):
    """
    使用K近邻方法为候选点分配credit值
    
    参数:
    - X_candidates: 候选点集，形状 (n_candidates, dim)
    - X_history: 历史数据点，形状 (n_history, dim)
    - credits: 历史数据点的credit值，形状 (n_history,) 或 (n_history, 1)
    - k: 近邻数量，默认5
    
    返回:
    - candidate_credits: 候选点的credit值，形状 (n_candidates,)
    """
    # 确保credits是一维数组
    if len(credits.shape) > 1:
        credits = credits.flatten()
    
    # 计算候选点与历史点之间的距离矩阵
    dist_matrix = cdist(X_candidates, X_history)
    
    # 对于每个候选点，找到k个最近的历史点的索引
    k = min(k, len(X_history))  # 确保k不超过历史点数量
    nearest_indices = np.argpartition(dist_matrix, k, axis=1)[:, :k]
    
    # 计算对应的credit平均值
    candidate_credits = np.zeros(X_candidates.shape[0])
    for i in range(X_candidates.shape[0]):
        candidate_credits[i] = np.mean(credits[nearest_indices[i]])
    
    return candidate_credits

#############################################
# Credit加权的采集函数
#############################################
def EI_credit(mu, sigma, X, y_best, credits=None, X_history=None):
    """基于Credit加权的期望改进(EI)采集函数"""
    mu = mu.reshape(-1, 1)
    sigma = sigma.reshape(-1, 1)
    # 计算标准EI值
    gamma = (y_best - mu) / (sigma + 1e-9)
    ei_values = (y_best - mu) * norm.cdf(gamma) + sigma * norm.pdf(gamma)
    
    # 如果提供了credits和历史数据，使用K近邻方法调整EI值
    if credits is not None and X_history is not None:
        # 为每个候选点分配credit值
        candidate_credits = get_knn_credits(X, X_history, credits)
        candidate_credits = candidate_credits.reshape(-1, 1)
        
        # 使用candidate_credits直接加权EI值 - 高credit值的点获得更高的采集值
        ei_values = ei_values * candidate_credits
    
    # 返回EI值最大的点
    x_new = X[np.argmax(ei_values), :]
    return x_new

def UCB_credit(mu, sigma, X, const=2.0, credits=None, X_history=None):
    """基于Credit加权的上置信界(UCB)采集函数"""
    mu = mu.reshape(-1, 1)
    sigma = sigma.reshape(-1, 1)
    
    # 如果提供了credits和历史数据，使用K近邻方法调整UCB值
    if credits is not None and X_history is not None:
        # 为每个候选点分配credit值
        candidate_credits = get_knn_credits(X, X_history, credits)
        candidate_credits = candidate_credits.reshape(-1, 1)
        
        # 高credit的点会增大探索系数，加强探索
        exploration_term = const * sigma * candidate_credits
        ucb_values = mu - exploration_term
    else:
        # 标准UCB
        ucb_values = mu - const * sigma
    
    # 返回UCB值最小的点（最小化问题）
    x_new = X[np.argmin(ucb_values), :]
    return x_new

def TS_credit(mu, sigma, X, credits=None, X_history=None, n_points=5):
    """基于Credit加权的Thompson采样(TS)采集函数"""
    mu = mu.reshape(-1, 1)
    sigma = sigma.reshape(-1, 1)
    eps = 1e-9
    std_clipped = np.maximum(sigma, eps)
    
    # 如果提供了credits和历史数据，使用K近邻方法调整采样标准差
    if credits is not None and X_history is not None:
        # 为每个候选点分配credit值
        candidate_credits = get_knn_credits(X, X_history, credits)
        candidate_credits = candidate_credits.reshape(-1, 1)
        
        # 高credit的点会增大采样标准差，增强探索
        std_clipped = std_clipped * np.sqrt(candidate_credits)
    
    # 进行Thompson采样
    y_samples = []
    for _ in range(n_points):
        y_sample = np.random.normal(loc=mu, scale=std_clipped)
        y_samples.append(y_sample)
    all_samples = np.hstack(y_samples)
    
    # 选择最小值对应的点（最小化问题）
    min_index = np.argmin(all_samples.mean(axis=1))
    return X[min_index]

#############################################
# 更新数据：新采样点 x_new, 对应目标值 y_new 加入历史数据中
#############################################
def update_data(x_new, y_new, D_old, Y_old):
    """
    x_new: numpy 数组，形状 (1, D)
    y_new: 标量，目标函数值
    D_old: numpy 数组，形状 (N, D)
    Y_old: numpy 数组，形状 (N,)
    """
    D_new = np.atleast_2d(x_new)
    y_new = np.atleast_1d(y_new)            # (1,)
    D_update = np.concatenate((D_old, D_new), axis=0)
    Y_update = np.concatenate((Y_old, y_new), axis=0)
    return D_update, Y_update

#############################################
# 主实验函数
#############################################
def Experiments(repet_time):
    print("Repetition:", repet_time + 1)
    np.random.seed(repet_time)
    start_all = time.time()
    date = args.time
    heter = args.heter
    # dy 在此版本中不直接用于 GP 模型，但保留以兼容测试函数（noise_std）
    dy = args.noise_std

    func = args.function
    select = args.select_method
    dimension_x = args.dimension_x  
    n_sample = args.n_sample  
    iteration = args.iteration 
    algorithm = args.algorithm
    Number_Z = args.n_Z
    print("Algorithm:", algorithm)

    # ----------------- 选择实验函数 -----------------
    if func == 'MLP_Diabete':
        dimension_x = 4
        f = MLP_Diabete(dimension_x, dimension_x)
    elif func == "Hart6":
        dimension_x = 6
        f = Hartmann6(dimension_x, dimension_x, dy)
    elif func == "Ackley":
        dimension_x = 8
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
    print("f_star:", f_star)

    # 设置采样总数
    N_total = args.n_sample + args.iteration

    # ----------------- 生成候选点 -----------------
    n_candidate = 10000
    a = f.lb
    b = f.ub
    X_ = np.random.rand(n_candidate, dimension_x)
    for dim_i in range(dimension_x):
        X_[:, dim_i] = X_[:, dim_i] * (b[dim_i] - a[dim_i]) + a[dim_i]

    # ----------------- 初始拉丁超立方采样 -----------------
    lhd = lhs(dimension_x, samples=n_sample, criterion="maximin")
    D1 = np.zeros((n_sample, dimension_x))
    Y1 = np.zeros(n_sample)
    for dim_i in range(dimension_x):
        D1[:, dim_i] = lhd[:, dim_i] * (b[dim_i] - a[dim_i]) + a[dim_i]
    for i in range(n_sample):
        Y1[i] = f(D1[i, :])
    
    # 初始化credit值 - 所有点初始值为1
    all_credits = np.ones(n_sample)

    # 转为 torch.tensor 用于 gpytorch 训练（double 精度）
    train_x = torch.tensor(D1, dtype=dtype, device=device)
    train_y = torch.tensor(Y1, dtype=dtype, device=device)

    # ----------------- 初始化 GP 模型（单输出 GP） -----------------
    model, likelihood = train_gp_model(train_x, train_y, training_iter=50, lr=0.1)
    
    mu_pred, sigma_pred = gp_predict(model, likelihood, X_)  # 预测
    mu_g = mu_pred
    sigma_g = sigma_pred

    # 初始化 minimizer 列表
    if dy == 0:
        minimizer = [D1[np.argmin(Y1), :]]
    else:
        mu_initial, _ = gp_predict(model, likelihood, D1)
        minimizer = [D1[np.argmin(mu_initial), :]]

    TIME_RE = []
    D_update = D1.copy()
    Y_update = Y1.copy()

    # 将历史采样点去重存放在 X_update 中（如不需去重，可直接 X_update = D_update）
    X_update = np.unique(D_update, axis=0)

    for i in range(iteration):
        if i % 20 == 0:
            print("Iteration", i + 1)
            # 可输出模型部分参数信息以便调试
            state_dict_small = {
                k: v.detach().cpu().numpy()
                for k, v in model.state_dict().items()
                if v.numel() < 10
            }
            print("GP model parameters:", state_dict_small)

        start = time.time()

        # =============== (1) 采集函数 - 使用Credit加权 ===============
        if algorithm == "EI":
            y_best = np.min(Y_update)  # 最小化问题
            x_new = EI_credit(mu_g, sigma_g, X_, y_best, 
                          credits=all_credits if heter else None,
                          X_history=D_update if heter else None)
        elif algorithm == "TS":
            X_temp = np.random.rand(10000, dimension_x)
            for dim_i in range(dimension_x):
                X_temp[:, dim_i] = X_temp[:, dim_i] * (b[dim_i] - a[dim_i]) + a[dim_i]
            
            # 获取预测
            mu_pred_temp, sigma_pred_temp = gp_predict(model, likelihood, X_temp)
            
            # 使用Credit加权的TS
            x_new = TS_credit(mu_pred_temp, sigma_pred_temp, X_temp, 
                          credits=all_credits if heter else None,
                          X_history=D_update if heter else None, 
                          n_points=5)
        elif algorithm == "UCB":
            print("select UCB")
            # 使用Credit加权的UCB - 增大探索系数以增强不确定性影响
            x_new = UCB_credit(mu_g, sigma_g, X_, const=3.5, 
                          credits=all_credits if heter else None,
                          X_history=D_update if heter else None)
        else:
            # 默认使用 EI
            y_best = np.min(Y_update)  # 最小化问题
            x_new = EI_credit(mu_g, sigma_g, X_, y_best, 
                         credits=all_credits if heter else None,
                         X_history=D_update if heter else None)

        # =============== (2) 评估新点，更新数据 ===============
        y_new = f(x_new)  # 目标函数值
        # 更新数据点和目标值
        D_update, Y_update = update_data(x_new, y_new, D_update, Y_update)
        
        # 使用 np.unique 去重，避免 set + vstack 报错
        X_update = np.unique(D_update, axis=0)

        # =============== (3) 计算 credit 并更新 ===============
        if heter:
            # 扩展credit数组以匹配新增的点
            all_credits = np.append(all_credits, 1.0)  # 新采样点初始credit为1
            
            # 对历史数据做预测
            mu_hist, sigma_hist = gp_predict(model, likelihood, D_update)
            Z_val = np.min(Y_update)  # 修改为最小值
            eps = 1e-6
            h_z_vals = norm.pdf(Z_val, loc=mu_hist, scale=sigma_hist + eps)
            
            # 仅更新新采样点的credit值
            if D_update.shape[0] > n_sample:
                new_indices = np.arange(n_sample, D_update.shape[0])
                new_h_z = h_z_vals[new_indices]
                # 假设新采样点原先的先验概率 ~ 1 / 新采样数
                new_pi_vals = np.full(new_h_z.shape, 1.0 / len(new_h_z))
                new_importance = new_h_z / new_pi_vals
                # 扩大credit值的范围，从[0.1, 10]扩大到[0.05, 20]以增强影响
                new_importance = np.clip(new_importance, 5e-2, 2e1)
                all_credits[new_indices] = new_importance
            
            # 每25次迭代打印credit值统计信息
            if i % 25 == 0:
                print(f"Credit值统计: 最小值={all_credits.min():.4f}, 最大值={all_credits.max():.4f}, 均值={all_credits.mean():.4f}")
                print(f"Credit值分位数: 25%={np.percentile(all_credits, 25):.4f}, 50%={np.percentile(all_credits, 50):.4f}, 75%={np.percentile(all_credits, 75):.4f}")

        # 重新训练单输出 GP 模型
        train_x = torch.tensor(D_update, dtype=dtype, device=device)
        train_y = torch.tensor(Y_update, dtype=dtype, device=device)
        model, likelihood = train_gp_model(train_x, train_y, training_iter=50, lr=0.1)

        # 利用训练好的模型预测候选点上的值
        mu_pred, sigma_pred = gp_predict(model, likelihood, X_)
        mu_g = mu_pred
        sigma_g = sigma_pred

        # =============== (4) 更新 minimizer 列表 ===============
        if dy == 0:
            minimizer.append(D_update[np.argmin(Y_update), :])
        else:
            mu_train, _ = gp_predict(model, likelihood, D_update)
            minimizer.append(D_update[np.argmin(mu_train), :])

        Training_time = time.time() - start
        print(f"Iteration {i + 1} took {Training_time:.2f} seconds.")
        TIME_RE.append(Training_time)

    end_all = time.time() - start_all
    print(f"******* All took {end_all:.2f} seconds.*******")
    TIME_RE = np.array(TIME_RE).reshape(1, -1)

    # =============== 评估收敛表现 ===============
    minimum2 = []
    minimizer2 = []
    for i in range(Y_update.shape[0]):
        minimum2.append(np.min(Y_update[: (i + 1)]))  # 最小化问题：取最小目标值
        minimizer2.append(D_update[np.argmin(Y_update[: (i + 1)]), :])
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

    # 创建结果目录（如果不存在）
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

    TIME_RE = np.array(TIME_RE).reshape(1, -1)
    file_configs = [
        ("_X_update.csv", X_update, False),
        ("_D_update.csv", D_update, False),
        ("_Y_update.csv", Y_update.reshape(-1, 1), True),  # 修改为单列
        ("_Credits.csv", all_credits.reshape(-1, 1), True),  # 新增，保存Credits
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
parser = argparse.ArgumentParser(description='KNN-Credit-Weighted Acquisition Functions for Bayesian Optimization')
parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")
parser.add_argument('-t', '--time', type=str, help='Date of the experiments', default='HCA_KNN_v1')
parser.add_argument('-select', '--select_method', type=str, help='select_method', default="No_select") 
parser.add_argument('-func', '--function', type=str, help='Function', default='Hart6')  
parser.add_argument('-algo', '--algorithm', type=str, help='algorithm: EI/TS/UCB', default="UCB")
parser.add_argument('-dimensionx', '--dimension_x', type=int, help='total dimension', default=6)
parser.add_argument('-noise', '--noise_std', type=float, help='noise', default=0.1)
parser.add_argument('-heter', '--heter', help='if noise is hetero', action='store_true', default=True)
parser.add_argument('-nsample', '--n_sample', type=int, help='number of initial samples', default=20)
parser.add_argument('-i', '--iteration', type=int, help='Number of iteration', default=400)
parser.add_argument('-macro', '--repet_num', type=int, help='Number of macroreplication', default=100)
parser.add_argument('-smacro', '--start_num', type=int, help='Start number of macroreplication', default=0)
parser.add_argument('-core', '--n_cores', type=int, help='Number of Cores', default=2)
parser.add_argument('-thre', '--threshold', type=float, help='threshold for EINgu', default=1e-4)
parser.add_argument('-Z', '--n_Z', type=int, help='Hyperparameter Z', default=4)
parser.add_argument('-k', '--knn', type=int, help='K in KNN for credit transfer', default=5)

args = parser.parse_args()

def run_experiment(experiment_id):
    print(f"Starting Experiment {experiment_id}")
    Experiments(experiment_id)
    print(f"Experiment {experiment_id} completed")

def run_experiments(start, end):
    for i in range(start, end + 1):
        run_experiment(i)

if __name__ == "__main__":
    start_experiment = 0
    end_experiment = 50
    run_experiments(start_experiment, end_experiment)