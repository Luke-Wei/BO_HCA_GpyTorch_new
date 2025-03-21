#!/usr/bin/env python
# coding: utf-8

import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
import warnings
from matplotlib import pyplot as plt
from pyDOE import lhs
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from acquisition_functions import EI, TS, UCB
import matplotlib.pyplot as plt
from scipy.stats import norm

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

def GP_model(D, Y, dy=0, noise_vector=None):
    """构建或更新 GP 模型，支持异方差噪声 (noise_vector)。"""
    # 如果没有指定 noise_vector，则使用一个极小的常数噪声，防止 alpha=None 报错
    if noise_vector is None:
        noise_vector = 1e-10
    
    if dy == 0:
        kernel = ConstantKernel(1.0, (1e-3, 1e1)) * RBF(1e-3, (1e-2, 1e1))
    else:
        kernel = (
            ConstantKernel(1.0, (1e-3, 1e1))
            * RBF(1e-3, (1e-2, 1e1))
            + WhiteKernel(0.1**2)
        )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=5,
        normalize_y=True,
        alpha=noise_vector
    )
    gp.fit(D, Y)
    return gp

def update_data(x_new, y_new, D_old, Y_old, repet_time=None):
    """将新采样 (x_new, y_new) 加入历史数据 D_old, Y_old。"""
    D_new = np.atleast_2d(x_new)
    Y_new = np.atleast_2d(y_new)

    D_update = np.concatenate((D_old, D_new), axis=0)
    Y_update = np.concatenate((Y_old, Y_new), axis=0)

    return D_update, Y_update

def Experiments(repet_time):
    print(repet_time + 1)
    np.random.seed(repet_time)

    start_all = time.time()

    date = args.time
    heter = args.heter
    if heter:
        dy = args.heter  # heter=True 时使用异方差 GP
    else:
        dy = args.noise_std  

    func = args.function
    select = args.select_method

    dimension_x = args.dimension_x  
    n_sample = args.n_sample  
    iteration = args.iteration 
    algorithm = args.algorithm
    Number_Z = args.n_Z
    print(algorithm)

    # ----------------- 选择实验函数 -----------------
    if func == 'MLP_Diabete':
        dimension_x = 4
        dy = 'nozero'
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

    x_star = f.x_star
    f_star = f.f_star
    print(f_star)

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
    from pyDOE import lhs
    lhd = lhs(dimension_x, samples=n_sample, criterion="maximin")
    D1 = np.zeros((n_sample, dimension_x))
    Y1 = np.zeros((n_sample, 1))
    for dim_i in range(dimension_x):
        D1[:, dim_i] = lhd[:, dim_i] * (b[dim_i] - a[dim_i]) + a[dim_i]
    for i in range(n_sample):
        Y1[i, :] = f(D1[i, :])

    # ----------------- 初始化 GP 模型 -----------------
    gp = GP_model(D1, Y1, dy)
    mu_g, sigma_g = gp.predict(X_, return_std=True)

    # 初始化 minimizer
    if dy == 0:
        # 纯 GP 情况下，可能直接取当前最优
        minimizer = list([D1[np.argmin(Y1), :]])
    else:
        # 有噪声时，先用 GP 均值选一个
        mu_AEI, _ = gp.predict(D1, return_std=True)
        minimizer = list([D1[np.argmin(mu_AEI), :]])

    TIME_RE = []
    D_update = D1
    Y_update = Y1
    X_update = np.vstack(list({tuple(row) for row in D_update}))

    initial_target_size = args.iteration
    target_size = initial_target_size
    twentieth_iteration_time = None
    recorded_sample_count = None

    kl_results = []

    for i in range(iteration):
        if i % 20 == 0:
            print("Iteration", i + 1)
            print(gp.kernel_)

        start = time.time()

        # =============== (1) 采集函数 ===============
        if algorithm == "EI":
            # 最大化问题: 这里 y_best = np.max(Y_update)
            y_best = np.max(Y_update)
            x_new = EI(mu_g, sigma_g, X_, y_best)
        elif algorithm == "TS":
            # 随机选 500 个点做 Thompson Sampling
            X_temp = np.random.rand(500, dimension_x)
            for dim_i in range(dimension_x):
                X_temp[:, dim_i] = X_temp[:, dim_i] * (b[dim_i] - a[dim_i]) + a[dim_i]
            x_new = TS(X_temp, gp)
        elif algorithm == "UCB":
            print("select UCB")
            x_new = UCB(mu_g, sigma_g, X_, np.sqrt(2))

        # =============== (2) 评估新点，更新数据 ===============
        y_new = f(x_new)
        D_update, Y_update = update_data(x_new, y_new, D_update, Y_update, repet_time)
        X_update = np.vstack(list({tuple(row) for row in D_update}))

        # =============== (3) 使用 HCA + 异方差 GP 更新模型 ===============
        if heter:
            # 对历史数据做预测
            mu_hist, sigma_hist = gp.predict(D_update, return_std=True)
            # 注意：最大化问题 -> 取当前最大值
            Z_val = np.max(Y_update)

            eps = 1e-6
            # 计算正态 PDF: h_z_vals[i] = pdf(Z_val | mu_hist[i], sigma_hist[i]^2)
            h_z_vals = norm.pdf(Z_val, loc=mu_hist, scale=sigma_hist + eps)

            # 原始采样概率假设均匀: pi(x_i) = 1/len(D_update)
            pi_vals = np.full_like(h_z_vals, 1.0 / len(h_z_vals))

            # 计算重要性采样比
            importance_ratios = h_z_vals / pi_vals

            # ---- 1) 对 importance_ratios 做截断 ----
            ratio_min, ratio_max = 1e-2, 1e2
            importance_ratios = np.clip(importance_ratios, ratio_min, ratio_max)

            # ---- 2) 计算噪声向量并做上下限约束 ----
            noise_floor, noise_ceil = 1e-6, 1.0
            raw_noise = (args.noise_std**2) / (importance_ratios + eps)
            noise_vector = np.clip(raw_noise, noise_floor, noise_ceil)

            # 重新构建 GP 模型
            gp = GP_model(D_update, Y_update, dy, noise_vector=noise_vector)
        else:
            # 不使用异方差
            gp = GP_model(D_update, Y_update, dy)

        # 预测新一轮候选集
        mu_g, sigma_g = gp.predict(X_, return_std=True)

        # =============== (4) 更新 minimizer 列表 ===============
        if dy == 0:
            minimizer.append(D_update[np.argmin(Y_update), :])
        else:
            mu_AEI, _ = gp.predict(D_update, return_std=True)
            minimizer.append(D_update[np.argmin(mu_AEI), :])

        # 时间统计
        Training_time = time.time() - start
        print(f"Iteration {i + 1} took {Training_time:.2f} seconds.")

        if i == 19:
            twentieth_iteration_time = Training_time

        if (
            twentieth_iteration_time 
            and Training_time > Number_Z * twentieth_iteration_time
            and recorded_sample_count is None
        ):
            recorded_sample_count = len(D_update)
            target_size = recorded_sample_count

        TIME_RE.append(Training_time)

    end_all = time.time() - start_all
    print(f"******* All took {end_all:.2f} seconds.*******")

    TIME_RE = np.array(TIME_RE).reshape(1, -1)

    # 计算迭代过程中的最优值序列
    minimum2 = []
    minimizer2 = []
    for i in range(Y_update.shape[0]):
        # 这里如果要最大化, 就应该用 max
        # 但为了兼容最小化, 这里写 min
        # 你可根据需要改成 np.max(...)
        # 下面以“最小值”为例
        # minimum2.append(np.min(Y_update[: (i + 1)]))
        # ...
        # 如果你是最大化，就改成:
        minimum2.append(np.max(Y_update[: (i + 1)]))
        minimizer2.append(D_update[np.argmax(Y_update[: (i + 1), 0]), :])

    minimizer2 = np.array(minimizer2)
    minimum2 = np.array(minimum2)

    # 拼接 minimizer_all
    minimizer_all = np.array(minimizer)
    minimizer_all = np.concatenate((minimizer2[: (n_sample - 1), :], minimizer_all), axis=0)
    print(minimizer_all.shape)

    # 计算最优值序列
    minimum = np.zeros(N_total)
    if dy == 0:
        minimum = minimum2
    else:
        for index in range(N_total):
            # 这里同样要注意最大化还是最小化
            # f.evaluate_true(...) 可能是要最小化, 你要自行改
            minimum[index] = f.evaluate_true(minimizer_all[index, :])

    minimum = minimum.reshape(-1, 1)

    # GAP = | minimum - f_star |
    GAP = np.abs(minimum - f_star).reshape(-1, 1)

    if x_star is not None:
        xGAP = np.min(
            np.linalg.norm(minimizer_all - np.array(x_star), axis=2), axis=0
        ).reshape(-1, 1)

    # SimRegret: 累计最优
    SimRegret = []
    for i in range(len(GAP)):
        SimRegret.append(np.min(GAP[: (i + 1)]))
    SimRegret = np.array(SimRegret).reshape(-1, 1)

    # 计算 Regret
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
        with open(f"{file}{suffix}", "ab") as f:
            output_data = data.T if needs_transpose else data
            np.savetxt(f, output_data, delimiter=",")

parser = argparse.ArgumentParser(description='EITar-algo')
parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")
parser.add_argument('-t', '--time', type=str, help='Date of the experiments', default='Baseline2')
parser.add_argument('-select', '--select_method', type=str, help='select_method', default="No_select") 
parser.add_argument('-func', '--function', type=str, help='Function', default='Hart6')  
parser.add_argument('-algo', '--algorithm', type=str, help='algorithm: EI/TS/UCB', default="UCB")
parser.add_argument('-dimensionx', '--dimension_x', type=int, help='total dimension', default=6)
parser.add_argument('-noise', '--noise_std', type=float, help='noise', default=0.1)
parser.add_argument('-heter', '--heter', help='if noise is hetero', action='store_true', default=False)
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
