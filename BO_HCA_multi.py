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
# 定义多任务 GP 模型（两个输出：任务0为目标函数，任务1为 credit）
#############################################
class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks=2):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.num_tasks = num_tasks
        # 多任务均值：每个任务均使用 ConstantMean
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        )
        # 多任务核：先用共享的 RBFKernel，再通过 MultitaskKernel 建立任务间协方差
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=num_tasks, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

#############################################
# 训练多任务 GP 模型（利用 gpytorch 训练）
#############################################
def train_gp_model(train_x, train_y, training_iter=50, lr=0.1):
    """
    train_x: torch.tensor, shape (N, D), dtype=torch.double
    train_y: torch.tensor, shape (N, 2), dtype=torch.double
    """
    # 将 likelihood 与模型都放到同一个 device 且使用 double
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
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

#############################################
# 利用训练好的模型做预测，返回均值和标准差（多任务，每列对应一个任务）
#############################################
def gp_predict(model, likelihood, X):
    """
    X: numpy 数组, shape (N, D)
    返回:
      mu: shape (N, 2) —— 每列对应一个任务的预测均值
      sigma: shape (N, 2) —— 每列对应一个任务的预测标准差
    """
    model.eval()
    likelihood.eval()

    test_x = torch.tensor(X, dtype=dtype, device=device)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        preds = likelihood(model(test_x))
    mu = preds.mean.detach().cpu().numpy()          # shape (N, 2)
    sigma = np.sqrt(preds.variance.detach().cpu().numpy())  # shape (N, 2)
    return mu, sigma

#############################################
# 更新数据：新采样点 x_new, 对应目标值 y_new 和 credit_new 加入历史数据中
#############################################
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
    Y1 = np.zeros((n_sample, 1))
    for dim_i in range(dimension_x):
        D1[:, dim_i] = lhd[:, dim_i] * (b[dim_i] - a[dim_i]) + a[dim_i]
    for i in range(n_sample):
        Y1[i, :] = f(D1[i, :])
    # 初始样本的 credit 均设为 1
    credit_init = np.ones((n_sample, 1))
    Y1 = np.hstack([Y1, credit_init])  # shape (n_sample, 2)

    # 转为 torch.tensor 用于 gpytorch 训练（double 精度）
    train_x = torch.tensor(D1, dtype=dtype, device=device)
    train_y = torch.tensor(Y1, dtype=dtype, device=device)

    # ----------------- 初始化 GP 模型（多输出 GP） -----------------
    model, likelihood = train_gp_model(train_x, train_y, training_iter=50, lr=0.1)
    mu_pred, sigma_pred = gp_predict(model, likelihood, X_)
    # 仅取任务0（目标函数）的预测值
    mu_g = mu_pred[:, 0]
    sigma_g = sigma_pred[:, 0]

    # 初始化 minimizer 列表
    if dy == 0:
        minimizer = [D1[np.argmin(Y1[:, 0]), :]]
    else:
        mu_initial, _ = gp_predict(model, likelihood, D1)
        minimizer = [D1[np.argmin(mu_initial[:, 0]), :]]

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

        # =============== (1) 采集函数 ===============
        if algorithm == "EI":
            y_best = np.max(Y_update[:, 0])  # 最大化问题
            x_new = EI(mu_g, sigma_g, X_, y_best)
        elif algorithm == "TS":
            X_temp = np.random.rand(500, dimension_x)
            for dim_i in range(dimension_x):
                X_temp[:, dim_i] = X_temp[:, dim_i] * (b[dim_i] - a[dim_i]) + a[dim_i]
            # 注意这里要把 model, likelihood 也传给 TS
            x_new = TS(X_temp, model, likelihood)
        elif algorithm == "UCB":
            print("select UCB")
            x_new = UCB(mu_g, sigma_g, X_, np.sqrt(2))
        else:
            # 默认使用 EI
            x_new = EI(mu_g, sigma_g, X_, np.max(Y_update[:, 0]))

        # =============== (2) 评估新点，更新数据 ===============
        y_new = f(x_new)  # 目标函数值
        # 新采样时，credit 初始设为1；若 heter==True，下面会更新 credit
        credit_new = 1.0
        D_update, Y_update = update_data(x_new, y_new, credit_new, D_update, Y_update)
        
        # 使用 np.unique 去重，避免 set + vstack 报错
        X_update = np.unique(D_update, axis=0)

        # =============== (3) 利用 HCA 计算 credit，并更新多任务 GP 模型 ===============
        if heter:
            # 对历史数据做预测（仅对任务0）
            mu_hist, sigma_hist = gp_predict(model, likelihood, D_update)
            mu_hist_task0 = mu_hist[:, 0]
            sigma_hist_task0 = sigma_hist[:, 0]
            Z_val = np.max(Y_update[:, 0])
            eps = 1e-6
            h_z_vals = norm.pdf(Z_val, loc=mu_hist_task0, scale=sigma_hist_task0 + eps)
            importance_ratios = np.ones_like(h_z_vals)
            if D_update.shape[0] > n_sample:
                new_indices = np.arange(n_sample, D_update.shape[0])
                new_h_z = h_z_vals[new_indices]
                # 假设新采样点原先的先验概率 ~ 1 / 新采样数
                new_pi_vals = np.full(new_h_z.shape, 1.0 / len(new_h_z))
                new_importance = new_h_z / new_pi_vals
                new_importance = np.clip(new_importance, 1e-1, 1e1)
                importance_ratios[new_indices] = new_importance
            # 更新所有数据点的 credit（第二列）
            Y_update[:, 1] = importance_ratios

        # 重新训练多任务 GP 模型（更新后的训练数据包含目标值和 credit）
        train_x = torch.tensor(D_update, dtype=dtype, device=device)
        train_y = torch.tensor(Y_update, dtype=dtype, device=device)
        model, likelihood = train_gp_model(train_x, train_y, training_iter=50, lr=0.1)

        # 利用训练好的模型预测候选点上的目标函数值（任务0）
        mu_pred, sigma_pred = gp_predict(model, likelihood, X_)
        mu_g = mu_pred[:, 0]
        sigma_g = sigma_pred[:, 0]

        # =============== (4) 更新 minimizer 列表 ===============
        if dy == 0:
            minimizer.append(D_update[np.argmin(Y_update[:, 0]), :])
        else:
            mu_train, _ = gp_predict(model, likelihood, D_update)
            minimizer.append(D_update[np.argmin(mu_train[:, 0]), :])

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
        minimum2.append(np.max(Y_update[: (i + 1), 0]))  # 最大化问题：取最大目标值
        minimizer2.append(D_update[np.argmax(Y_update[: (i + 1), 0]), :])
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
        ("_Y_update.csv", Y_update, True),
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
parser = argparse.ArgumentParser(description='HCA-Improved with Multioutput GP (gpytorch)')
parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")
parser.add_argument('-t', '--time', type=str, help='Date of the experiments', default='HCA_multi')
parser.add_argument('-select', '--select_method', type=str, help='select_method', default="No_select") 
parser.add_argument('-func', '--function', type=str, help='Function', default='Hart6')  
parser.add_argument('-algo', '--algorithm', type=str, help='algorithm: EI/TS/UCB', default="EI")
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
