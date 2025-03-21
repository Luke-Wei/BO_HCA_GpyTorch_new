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

def GP_model(D, Y, dy=0, noise_vector=None, heter=False):
    """
    构建或更新 GP 模型。
    当 heter=True 时（使用 HCA 异方差），不添加 WhiteKernel，
    以避免噪声重复建模。
    """
    if noise_vector is None:
        noise_vector = 1e-10  # 默认极小噪声
    
    if not heter:
        # 非异方差模式：可加入 WhiteKernel（如果 dy != 0）
        if dy == 0:
            kernel = ConstantKernel(1.0, (1e-3, 1e1)) * RBF(1e-3, (1e-2, 1e1))
        else:
            kernel = ConstantKernel(1.0, (1e-3, 1e1)) * RBF(1e-3, (1e-2, 1e1)) + WhiteKernel(0.1**2)
    else:
        # heter=True: 不使用 WhiteKernel，使用 RBF 构成核
        kernel = ConstantKernel(1.0, (1e-3, 1e1)) * RBF(1e-3, (1e-2, 1e1))
        
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

def softmax(vec, alpha=1.0):
    """Softmax处理避免极端值."""
    exps = np.exp(alpha * vec)
    return exps / (np.sum(exps) + 1e-12)

def dynamic_clip(values, iteration, max_iter, low_range, high_range):
    """
    动态收窄截断范围。
    iteration: 当前迭代次数
    max_iter: 总迭代次数
    low_range, high_range: (low_min, low_max), (high_min, high_max)
    """
    # 简单示例：随迭代线性内插
    alpha = iteration / (max_iter + 1e-12)
    # 动态低截断
    lrange = low_range[0] + alpha * (low_range[1] - low_range[0])
    # 动态高截断
    hrange = high_range[0] + alpha * (high_range[1] - high_range[0])
    return np.clip(values, lrange, hrange)

def get_acquisition_value(x_points, gp):
    """示例：结合acquisition function(如EI)进行全局评价."""
    # 可以根据需求，用EI或UCB来给每个点附加全局价值
    # 这里只做简单EI示例
    mu_vals, sigma_vals = gp.predict(x_points, return_std=True)
    y_best = np.max(gp.y_train_)  # 当前最优
    z = (mu_vals - y_best) / (sigma_vals + 1e-9)
    ei_values = (mu_vals - y_best) * norm.cdf(z) + (sigma_vals)*norm.pdf(z)
    # 返回EI向量
    return ei_values

def info_gain_approx(mu, sigma, prior_sigma=1.0):
    """
    用信息增益的角度(简化示例).
    IG(x) = 0.5 * log(prior_sigma^2 / sigma^2)
    你可以改成更合理的方式，如熵差/kl散度等.
    """
    return 0.5 * np.log((prior_sigma**2 + 1e-9)/(sigma**2 + 1e-9))

def Experiments(repet_time):
    print(repet_time + 1)
    np.random.seed(repet_time)
    start_all = time.time()
    date = args.time
    heter = args.heter
    if heter:
        dy = args.noise_std
    else:
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

    # ----------------- 初始化 GP 模型 -----------------
    # 初始阶段不使用异方差（heter=False），以确保模型稳定
    gp = GP_model(D1, Y1, dy, heter=False)
    mu_g, sigma_g = gp.predict(X_, return_std=True)

    # 初始化 minimizer 列表
    if dy == 0:
        minimizer = list([D1[np.argmin(Y1), :]])
    else:
        mu_AEI, _ = gp.predict(D1, return_std=True)
        minimizer = list([D1[np.argmin(mu_AEI), :]])

    TIME_RE = []
    D_update = D1
    Y_update = Y1
    X_update = np.vstack(list({tuple(row) for row in D_update}))

    max_iter = iteration  # 用于动态截断
    for i in range(iteration):
        if i % 20 == 0:
            print("Iteration", i + 1)
            print("GP kernel:", gp.kernel_)

        start = time.time()

        # =============== (1) 采集函数 ===============
        # 假设是最大化场景
        if algorithm == "EI":
            y_best = np.max(Y_update)
            x_new = EI(mu_g, sigma_g, X_, y_best)
        elif algorithm == "TS":
            X_temp = np.random.rand(500, dimension_x)
            for dim_i in range(dimension_x):
                X_temp[:, dim_i] = X_temp[:, dim_i] * (b[dim_i] - a[dim_i]) + a[dim_i]
            x_new = TS(X_temp, gp)
        elif algorithm == "UCB":
            print("select UCB")
            x_new = UCB(mu_g, sigma_g, X_, np.sqrt(2))

        # =============== (2) 评估新点，更新数据 ===============
        y_new = f(x_new)
        D_update, Y_update = update_data(x_new, y_new, D_update, Y_update)
        X_update = np.vstack(list({tuple(row) for row in D_update}))

        # =============== (3) 若 heter=True，则使用 HCA + 异方差 GP 更新模型 ===============
        if heter:
            mu_hist, sigma_hist = gp.predict(D_update, return_std=True)
            # 对于最大化问题，取当前最大值作为目标 Z_val
            Z_val = np.max(Y_update)
            eps = 1e-6

            # --------- 3.1 用信息增益 or PDF 计算贡献（可二选一或结合） ---------
            # 3.1.1 用原有 pdf 计算
            pdf_vals = norm.pdf(Z_val, loc=mu_hist, scale=sigma_hist + eps)
            
            # 3.1.2 用信息增益近似
            # prior_sigma 可用某个全局均值，如 np.mean(sigma_hist) 或固定
            prior_sigma = 1.0
            ig_vals = info_gain_approx(mu_hist, sigma_hist, prior_sigma)
            
            # 3.1.3 结合局部与全局 (EI) -> 全局信息
            # 这里演示把 EI 值也引入
            # 先简单 predict EI for D_update
            y_current_best = np.max(Y_update)
            zvals = (mu_hist - y_current_best)/(sigma_hist + eps)
            ei_local = (mu_hist - y_current_best)*norm.cdf(zvals) + sigma_hist*norm.pdf(zvals)
            ei_local = np.maximum(ei_local, 0)

            # --------- 3.2 最终贡献 cred(x_i) -----------
            # 结合pdf, IG, EI等（这里只是演示，可权重组合）
            cred = pdf_vals + 0.5*ig_vals + 0.2*ei_local  # 示例混合
            
            # --------- 3.3 只对新采样点(>=n_sample)计算-----------
            importance_ratios = np.ones_like(cred)
            if D_update.shape[0] > n_sample:
                new_indices = np.arange(n_sample, D_update.shape[0])
                new_cred = cred[new_indices]

                # softmax平滑，避免极端
                alpha_soft = 1.0
                new_cred_smooth = softmax(new_cred, alpha=alpha_soft)

                # 动态截断区间
                # early iteration => [1e-1, 1e1], later => [0.5,2]
                clipped_ratios = dynamic_clip(
                    new_cred_smooth,
                    i, max_iter,
                    low_range=(1e-1, 0.5),
                    high_range=(1e1, 2.0)
                )
                importance_ratios[new_indices] = clipped_ratios

            # 3.4 根据 importance_ratios 调整噪声
            noise_floor, noise_ceil = 1e-4, 0.1
            raw_noise = (args.noise_std**2)/(importance_ratios + eps)
            noise_vector = np.clip(raw_noise, noise_floor, noise_ceil)

            # --------- 3.5 更新异方差GP -----------
            gp = GP_model(D_update, Y_update, dy, noise_vector=noise_vector, heter=True)
        else:
            gp = GP_model(D_update, Y_update, dy, heter=False)

        # =============== (4) 预测下一轮候选集 ===============
        mu_g, sigma_g = gp.predict(X_, return_std=True)

        # =============== (5) 更新 minimizer 列表 ===============
        if dy == 0:
            minimizer.append(D_update[np.argmin(Y_update), :])
        else:
            mu_AEI, _ = gp.predict(D_update, return_std=True)
            minimizer.append(D_update[np.argmin(mu_AEI), :])

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
        # 最大化问题
        minimum2.append(np.max(Y_update[: (i + 1)]))
        minimizer2.append(D_update[np.argmax(Y_update[: (i + 1), 0]), :])
    minimizer2 = np.array(minimizer2)
    minimum2 = np.array(minimum2)

    minimizer_all = np.array(minimizer)
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
    file = "_".join([date, str(Number_Z), select, func, algorithm,
                     str(dy), str(n_sample), str(iteration), str(dimension_x)])
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
        with open(f"{file}{suffix}", "ab") as f_out:
            output_data = data.T if needs_transpose else data
            np.savetxt(f_out, output_data, delimiter=",")

parser = argparse.ArgumentParser(description='HCA-Improved')
parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")
parser.add_argument('-t', '--time', type=str, help='Date of the experiments', default='HCA_v5')
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
