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
from acquisition_functions import EI, TS, UCB # Standard acquisition functions
import matplotlib.pyplot as plt
from scipy.stats import norm
from pyDOE import lhs
from scipy.spatial.distance import cdist # Needed for biased sampling

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

# Use double precision consistently
dtype = torch.double
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#############################################
# Create Heteroscedastic Likelihood - From BO_HCA_multi_v3.py
#############################################
class HeteroscedasticLikelihood(gpytorch.likelihoods.MultitaskGaussianLikelihood):
    def __init__(self, num_tasks=2, **kwargs):
        super().__init__(num_tasks=num_tasks, **kwargs)

    def forward(self, function_samples, **kwargs):
        # Get mean and covariance from function_samples
        mean = function_samples.mean
        covar = function_samples.lazy_covariance_matrix

        # If credit information is provided, use it to adjust noise
        if 'credits' in kwargs and kwargs['credits'] is not None:
            credits = kwargs['credits']

            # Prevent division by zero, ensure minimum value
            credits = torch.clamp(credits, min=1e-6)

            # Get the current noise terms
            noise_diag = self.task_noises.expand(credits.shape[0], self.num_tasks)

            # Create adjusted noise matrix - Task 0 (objective) noise is affected by credit
            # Only adjust Task 0 noise, Task 1 (credit itself) noise remains unchanged
            adjusted_noise = noise_diag.clone()
            # Enhance credit's influence on noise, squared relationship makes noise lower for high-credit samples
            adjusted_noise[:, 0] = adjusted_noise[:, 0] / (credits * credits) # Use squared relationship

            # Create distribution using adjusted noise
            return gpytorch.distributions.MultitaskMultivariateNormal(
                mean, self._add_task_noises(covar, adjusted_noise)
            )

        # If no credit provided, use default behavior
        return super().forward(function_samples)

#############################################
# Define Multi-Task GP Model (Two outputs: Task 0 is objective, Task 1 is credit) - From BO_HCA_multi_v3.py
#############################################
class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks=2):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.num_tasks = num_tasks
        # Multi-task mean: Use ConstantMean for each task
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        )
        # Multi-task kernel: Use a shared RBFKernel, then MultitaskKernel for inter-task covariance
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=num_tasks, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

#############################################
# Train Multi-Task GP Model (using gpytorch) - Modified to use new likelihood - From BO_HCA_multi_v3.py
#############################################
def train_gp_model(train_x, train_y, training_iter=50, lr=0.1):
    """
    train_x: torch.tensor, shape (N, D), dtype=torch.double
    train_y: torch.tensor, shape (N, 2), dtype=torch.double (objective, credit)
    """
    # Use custom heteroscedastic noise likelihood
    likelihood = HeteroscedasticLikelihood(num_tasks=2)
    model = MultitaskGPModel(train_x, train_y, likelihood, num_tasks=2)

    # Move to specified device and dtype
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
# Predict using the trained model, return mean and std dev (multi-task, each column per task) - Modified to pass credit - From BO_HCA_multi_v3.py
#############################################
def gp_predict(model, likelihood, X, credits=None):
    """
    X: numpy array, shape (N, D)
    credits: numpy array, shape (N,) or None - Used to adjust prediction noise
    Returns:
      mu: shape (N, 2) —— Predicted mean for each task per column
      sigma: shape (N, 2) —— Predicted standard deviation for each task per column
    """
    model.eval()
    likelihood.eval()

    test_x = torch.tensor(X, dtype=dtype, device=device)

    # Convert credits to tensor (if provided)
    credits_tensor = None
    if credits is not None:
        # Ensure credits is 1D before converting
        if len(credits.shape) > 1:
           credits = credits.flatten()
        credits_tensor = torch.tensor(credits, dtype=dtype, device=device)

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # Get latent function distribution using the model
        f_preds = model(test_x)
        # Pass credits to the likelihood to adjust noise
        # Pass only the credits dimension needed by the likelihood's forward method
        if credits_tensor is not None:
            preds = likelihood(f_preds, credits=credits_tensor)
        else:
            preds = likelihood(f_preds) # Use default noise if no credits given

    mu = preds.mean.detach().cpu().numpy()          # shape (N, 2)
    sigma = np.sqrt(preds.variance.detach().cpu().numpy())  # shape (N, 2)
    return mu, sigma

#############################################
# Credit-Biased Candidate Generation - From BO_HCA_Sample.py
#############################################
def generate_candidates_with_credit_bias(n_candidate, dimension_x, lb, ub, X_history=None, credits=None,
                                         uniform_ratio=0.5, sigma_scale=0.1,
                                         smooth_power=0.5, min_prob=0.05,
                                         use_repulsion=True, repulsion_strength=1.0):
    """
    Generates candidate points with denser sampling in high-credit regions.

    Args:
    - n_candidate: Total number of candidate points to generate.
    - dimension_x: Problem dimension.
    - lb, ub: Lower and upper bounds for each dimension, shape (dimension_x,).
    - X_history: Historical sample points, shape (n_history, dimension_x).
    - credits: Credit values for historical points, shape (n_history,).
    - uniform_ratio: Proportion of uniformly sampled points, default 0.5.
    - sigma_scale: Standard deviation scaling for Gaussian sampling, default 0.1.
    - smooth_power: Smoothing exponent for credit values (<1 reduces differences, >1 increases), default 0.5 (sqrt).
    - min_prob: Minimum probability threshold for low-credit regions, prevents ignoring areas, default 0.05.
    - use_repulsion: Whether to use region repulsion mechanism, default True.
    - repulsion_strength: Repulsion intensity, larger value means stronger repulsion, default 1.0.

    Returns:
    - X_candidates: Generated candidate points, shape (n_candidate, dimension_x).
    """
    # Fallback to uniform sampling if no history or credits
    if X_history is None or credits is None or len(X_history) == 0:
        X_uniform = np.random.rand(n_candidate, dimension_x)
        for dim_i in range(dimension_x):
            X_uniform[:, dim_i] = X_uniform[:, dim_i] * (ub[dim_i] - lb[dim_i]) + lb[dim_i]
        return X_uniform

    # Ensure credits is a 1D array
    if len(credits.shape) > 1:
        credits = credits.flatten()

    # Calculate number of uniform and biased points
    n_uniform = int(n_candidate * uniform_ratio)
    n_biased = n_candidate - n_uniform

    # Generate uniformly sampled points
    X_uniform = np.random.rand(n_uniform, dimension_x)
    for dim_i in range(dimension_x):
        X_uniform[:, dim_i] = X_uniform[:, dim_i] * (ub[dim_i] - lb[dim_i]) + lb[dim_i]

    # If no biased points needed, return uniform samples
    if n_biased <= 0:
        return X_uniform

    # ========== Improvement 1: Smooth credit distribution ==========
    # Ensure credits are positive
    credits_positive = np.maximum(credits, 1e-10)

    # Smooth credit distribution using power function (sqrt compresses differences)
    smoothed_credits = np.power(credits_positive, smooth_power)

    # Set minimum probability threshold to prevent ignoring low-credit regions
    if min_prob > 0:
        # Calculate the minimum value threshold
        min_val = min_prob * np.max(smoothed_credits)
        # Elevate values below the threshold
        smoothed_credits = np.maximum(smoothed_credits, min_val)

    # Normalize to a probability distribution
    if np.sum(smoothed_credits) <= 1e-8:
        normalized_credits = np.ones_like(smoothed_credits) / len(smoothed_credits)
    else:
        normalized_credits = smoothed_credits / np.sum(smoothed_credits)
        # Re-normalize to ensure sum is exactly 1 due to potential floating point issues
        normalized_credits = normalized_credits / np.sum(normalized_credits)

    # ========== Improvement 3: Region Repulsion Mechanism ==========
    if use_repulsion and len(X_history) >= 2:
        # Calculate distance matrix between historical points
        dist_matrix = cdist(X_history, X_history)

        # Closer points have stronger mutual repulsion
        max_dist = np.max(dist_matrix)
        if max_dist > 0: # Avoid division by zero
            # Relative distance matrix (0~1, closer points have smaller values)
            rel_dist = dist_matrix / max_dist

            # Repulsion intensity matrix (closer points have stronger repulsion)
            repulsion = 1.0 - rel_dist
            np.fill_diagonal(repulsion, 0) # No self-repulsion

            # Calculate total repulsion effect on each point
            # High-credit points have greater repulsive influence on others
            credit_mat = normalized_credits.reshape(-1, 1) # Column vector
            # Transpose credit_mat to align for element-wise multiplication broadcasting
            total_repulsion = np.sum(repulsion * credit_mat.T, axis=1)

            # Apply repulsion effect, reducing weights in high-repulsion areas
            repulsion_factor = 1.0 / (1.0 + repulsion_strength * total_repulsion)
            normalized_credits = normalized_credits * repulsion_factor

            # Re-normalize
            if np.sum(normalized_credits) > 1e-8:
                normalized_credits = normalized_credits / np.sum(normalized_credits)

    # ========== Select reference points based on processed credit probabilities ==========
    selected_indices = np.random.choice(
        np.arange(len(X_history)),
        size=n_biased,
        p=normalized_credits,
        replace=True
    )

    # ========== Improvement 2: Efficient Neighboring Point Generation ==========
    # Use vectorized operations for high dimensions, avoid per-point loops

    # Get all selected reference points
    reference_points = X_history[selected_indices]

    # Adapt sigma_scale based on dimension - use smaller sigma for higher dimensions
    adaptive_sigma_scale = sigma_scale / np.sqrt(max(1, dimension_x / 10.0)) # Use float division

    # Calculate standard deviation for each dimension
    # Use range (ub-lb) as scale, capped at 1 to avoid excessive spread if range is large
    sigma_vector = adaptive_sigma_scale * np.minimum(ub - lb, np.ones(dimension_x))


    # Generate Gaussian noise using vectorized operations
    if dimension_x > 50: # High-dimensional optimization: perturb only a subset of dimensions
        # Randomly select dimensions to perturb for each point
        mask = np.random.rand(n_biased, dimension_x) < 0.3 # 30% chance per dimension
        # Ensure at least one dimension is perturbed per point
        zero_rows = np.sum(mask, axis=1) == 0
        if np.any(zero_rows):
            rand_dims = np.random.randint(0, dimension_x, size=np.sum(zero_rows))
            mask[zero_rows, rand_dims] = True

        # Generate Gaussian noise only for selected dimensions
        noise = np.zeros((n_biased, dimension_x))
        noise[mask] = np.random.normal(0, 1, size=np.sum(mask))

        # Apply dimension-specific standard deviation
        noise = noise * sigma_vector # Broadcasting applies sigma_vector to each row
    else:
        # Standard approach: generate noise for all dimensions
        noise = np.random.normal(0, 1, size=(n_biased, dimension_x))
        # Apply dimension-specific standard deviation
        noise = noise * sigma_vector # Broadcasting

    # Generate new points = reference points + noise
    X_biased = reference_points + noise

    # Ensure points are within bounds
    X_biased = np.maximum(X_biased, lb)
    X_biased = np.minimum(X_biased, ub)

    # Combine uniform and credit-biased samples
    X_candidates = np.concatenate([X_uniform, X_biased], axis=0)

    return X_candidates


#############################################
# Update Data: Add new sample point x_new, corresponding objective y_new, and credit_new to history - From BO_HCA_multi_v3.py
#############################################
def update_data(x_new, y_new, credit_new, D_old, Y_old):
    """
    x_new: numpy array, shape (1, D) or (D,)
    y_new: scalar, objective function value
    credit_new: scalar, credit value
    D_old: numpy array, shape (N, D)
    Y_old: numpy array, shape (N, 2)
    """
    D_new = np.atleast_2d(x_new)
    y_new = np.atleast_2d(y_new)            # Shape (1, 1)
    credit_new = np.atleast_2d(credit_new)  # Shape (1, 1)
    new_Y = np.hstack([y_new, credit_new])  # Shape (1, 2)
    D_update = np.concatenate((D_old, D_new), axis=0)
    Y_update = np.concatenate((Y_old, new_Y), axis=0)
    return D_update, Y_update

#############################################
# Main Experiment Function - Integrated
#############################################
def Experiments(repet_time):
    print("Repetition:", repet_time + 1)
    np.random.seed(repet_time)
    start_all = time.time()
    date = args.time
    heter = args.heter # Use heteroscedastic noise / credit mechanism?
    dy = args.noise_std # Base noise level (used by test function)

    func = args.function
    select = args.select_method # (Potentially unused if method fixed)
    dimension_x = args.dimension_x
    n_sample = args.n_sample
    iteration = args.iteration
    algorithm = args.algorithm # EI, TS, UCB
    # Number_Z = args.n_Z # (Potentially unused if HCA fixed)
    print("Algorithm:", algorithm)
    print("Heteroscedastic:", heter)

    # ----------------- Select Test Function -----------------
    # (Function selection logic remains the same as original files)
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
        print(f"Warning: Function '{func}' not recognized. Defaulting to Hartmann6.")
        dimension_x = 6
        f = Hartmann6(dimension_x, dimension_x, dy)


    x_star = f.x_star
    f_star = f.f_star
    print(f"Function: {func}, Dimension: {dimension_x}, f_star: {f_star}")

    # Total number of samples
    N_total = args.n_sample + args.iteration

    # ----------------- Initial Candidate Points (Uniform) -----------------
    # These are only used before the first iteration or if heter=False
    n_candidate = 10000
    a = f.lb
    b = f.ub
    X_uniform_candidates = np.random.rand(n_candidate, dimension_x)
    for dim_i in range(dimension_x):
        X_uniform_candidates[:, dim_i] = X_uniform_candidates[:, dim_i] * (b[dim_i] - a[dim_i]) + a[dim_i]
    X_ = X_uniform_candidates # Initial candidates

    # ----------------- Initial Latin Hypercube Sampling -----------------
    lhd = lhs(dimension_x, samples=n_sample, criterion="maximin")
    D1 = np.zeros((n_sample, dimension_x))
    Y_obj_init = np.zeros((n_sample, 1)) # Objective values
    for dim_i in range(dimension_x):
        D1[:, dim_i] = lhd[:, dim_i] * (b[dim_i] - a[dim_i]) + a[dim_i]
    for i in range(n_sample):
        Y_obj_init[i, :] = f(D1[i, :])
    # Initial samples' credit are all set to 1
    credit_init = np.ones((n_sample, 1))
    Y1 = np.hstack([Y_obj_init, credit_init]) # Shape (n_sample, 2)

    # Convert to torch.tensor for gpytorch training (double precision)
    train_x = torch.tensor(D1, dtype=dtype, device=device)
    train_y = torch.tensor(Y1, dtype=dtype, device=device)

    # ----------------- Initialize Multi-output GP Model -----------------
    model, likelihood = train_gp_model(train_x, train_y, training_iter=50, lr=0.1)

    # Initial predictions on candidate points
    # Pass initial credits (all 1s) for prediction consistency if needed by likelihood, though effect might be minimal here
    initial_credits_for_predict = Y1[:, 1] if heter else None
    # Predict on uniform candidates initially
    mu_pred_tasks, sigma_pred_tasks = gp_predict(model, likelihood, X_, credits=None) # Predict without credits effect initially is fine
    mu_g = mu_pred_tasks[:, 0]      # Task 0: Objective Mean
    sigma_g = sigma_pred_tasks[:, 0]  # Task 0: Objective Std Dev

    # Initialize minimizer list
    if dy == 0: # Noiseless case
        current_best_idx = np.argmin(Y1[:, 0])
        minimizer = [D1[current_best_idx, :]]
        current_best_y = Y1[current_best_idx, 0]
    else: # Noisy case, use predicted mean on initial data
        mu_initial_tasks, _ = gp_predict(model, likelihood, D1, credits=initial_credits_for_predict)
        current_best_idx = np.argmin(mu_initial_tasks[:, 0])
        minimizer = [D1[current_best_idx, :]]
        # Estimate best y based on prediction at the best predicted point
        current_best_y = mu_initial_tasks[current_best_idx, 0]


    TIME_RE = []
    D_update = D1.copy()
    Y_update = Y1.copy() # Stores [objective, credit]

    for i in range(iteration):
        if i % 20 == 0:
            print("Iteration", i + 1)
            # Optionally print some model parameters for debugging
            # state_dict_small = {
            #     k: v.detach().cpu().numpy()
            #     for k, v in model.state_dict().items()
            #     if v.numel() < 10
            # }
            # print("GP model parameters snippet:", state_dict_small)
            print(f"Current best objective (estimated): {current_best_y:.4f}")


        start = time.time()

        # =============== (0) Generate Candidate Points ===============
        # Use credit-biased sampling if heter=True and not the first iteration
        if heter and i > 0:
            # Adaptive parameters for biased sampling (optional, could use fixed values)
            sigma_scale = max(0.05, 0.2 * (1 - i / iteration))
            uniform_ratio = max(0.2, 0.5 * (1 - i / (2 * iteration)))
            smooth_power = max(0.3, 0.5 * (1 - i / iteration))
            repulsion_strength = min(2.0, 1.0 + i / (iteration * 0.5))

            # Generate candidates biased by current credits
            X_ = generate_candidates_with_credit_bias(
                n_candidate=n_candidate,
                dimension_x=dimension_x,
                lb=f.lb,
                ub=f.ub,
                X_history=D_update,
                credits=Y_update[:, 1], # Pass current credits
                uniform_ratio=uniform_ratio,
                sigma_scale=sigma_scale,
                smooth_power=smooth_power,
                min_prob=0.05,
                use_repulsion=True,
                repulsion_strength=repulsion_strength
            )
            # Predict on the new non-uniform candidates
            # Pass current data's credits to gp_predict for heteroscedastic noise adjustment during prediction
            mu_pred_tasks, sigma_pred_tasks = gp_predict(model, likelihood, X_, credits=Y_update[:, 1])
            mu_g = mu_pred_tasks[:, 0]      # Task 0: Objective Mean
            sigma_g = sigma_pred_tasks[:, 0]  # Task 0: Objective Std Dev

        elif i == 0: # First iteration uses the initial uniform candidates and predictions
             pass # X_, mu_g, sigma_g already computed before loop
        else: # heter=False, always use uniform candidates
            X_ = X_uniform_candidates
            # Predict on uniform candidates (no credit effect)
            mu_pred_tasks, sigma_pred_tasks = gp_predict(model, likelihood, X_, credits=None)
            mu_g = mu_pred_tasks[:, 0]
            sigma_g = sigma_pred_tasks[:, 0]


        # =============== (1) Acquisition Function ===============
        # Use standard acquisition functions applied to Task 0 (objective) predictions
        # The predictions mu_g, sigma_g are already informed by credit via MTGP if heter=True
        if algorithm == "EI":
            y_best = current_best_y # Use the current best estimated objective value
            x_new = EI(mu_g, sigma_g, X_, y_best) # Standard EI
        elif algorithm == "TS":
             # Thompson Sampling based on objective task predictions
             # Reshape for sampling function if needed
             mu_ts = mu_g.reshape(-1, 1)
             sigma_ts = sigma_g.reshape(-1, 1)
             eps = 1e-9
             std_clipped = np.maximum(sigma_ts, eps)

             # Perform Thompson sampling
             n_points_ts = 5 # Number of samples for TS stability
             y_samples = []
             for _ in range(n_points_ts):
                 y_sample = np.random.normal(loc=mu_ts, scale=std_clipped)
                 y_samples.append(y_sample)
             all_samples = np.hstack(y_samples)

             # Select the point with the minimum average sampled value
             min_index = np.argmin(all_samples.mean(axis=1))
             x_new = X_[min_index] # Select from the potentially biased candidates X_
        elif algorithm == "UCB":
             # Standard UCB (minimization form)
             # Increase exploration constant for potentially noisy/heteroscedastic setting
             x_new = UCB(mu_g, sigma_g, X_, const=3.5)
        else:
            # Default to EI
            print(f"Warning: Algorithm '{algorithm}' not recognized. Defaulting to EI.")
            y_best = current_best_y
            x_new = EI(mu_g, sigma_g, X_, y_best)

        # =============== (2) Evaluate New Point, Update Data ===============
        y_new_obj = f(x_new)  # Evaluate objective function
        # New sample's credit is initially 1.0; will be updated by HCA if heter=True
        credit_new = 1.0
        D_update, Y_update = update_data(x_new, y_new_obj, credit_new, D_update, Y_update)

        # (Optional: Remove duplicates from D_update, Y_update if necessary, but usually handled by GP)
        # X_update = np.unique(D_update, axis=0) # If unique points needed elsewhere

        # =============== (3) Calculate Credit using HCA and Update GP Model ===============
        if heter:
            # Predict historical points' objective (Task 0) using current model
            # Pass current credits for consistent prediction variance
            mu_hist_tasks, sigma_hist_tasks = gp_predict(model, likelihood, D_update, credits=Y_update[:, 1])
            mu_hist_task0 = mu_hist_tasks[:, 0]
            sigma_hist_task0 = sigma_hist_tasks[:, 0]

            # HCA calculation based on current best observed value Z
            Z_val = np.min(Y_update[:, 0]) # Current best observed objective value
            eps = 1e-6
            # Probability density of observing Z at each historical point under the current GP model
            h_z_vals = norm.pdf(Z_val, loc=mu_hist_task0, scale=sigma_hist_task0 + eps)

            # Calculate importance ratios (credits)
            importance_ratios = np.ones_like(h_z_vals) # Initialize credits to 1

            # Calculate importance only for points added after initial sampling
            if D_update.shape[0] > n_sample:
                # Indices of points added during optimization iterations
                new_indices = np.arange(n_sample, D_update.shape[0])
                # h(z|x) values for the new points
                new_h_z = h_z_vals[new_indices]

                # Assume uniform prior proposal distribution for new points pi(x) ~ 1/num_new_points
                # Avoid division by zero if new_h_z is empty (shouldn't happen here)
                num_new_points = len(new_h_z)
                if num_new_points > 0:
                    new_pi_vals = np.full(new_h_z.shape, 1.0 / num_new_points)
                    new_importance = new_h_z / (new_pi_vals + eps) # Add eps for stability

                    # Clip importance ratios to a reasonable range to avoid extreme values
                    # Wider range [0.05, 20] compared to original [0.1, 10]
                    new_importance = np.clip(new_importance, 5e-2, 2e1)

                    # Update importance ratios for the new points
                    importance_ratios[new_indices] = new_importance

            # Update the credit column (Task 1) in Y_update
            Y_update[:, 1] = importance_ratios

            # Optional: Print credit statistics periodically
            if i % 25 == 0:
                 print(f"Credit Stats: Min={Y_update[:, 1].min():.4f}, Max={Y_update[:, 1].max():.4f}, Mean={Y_update[:, 1].mean():.4f}")
                 # print(f"Credit Quantiles: 25%={np.percentile(Y_update[:, 1], 25):.4f}, 50%={np.percentile(Y_update[:, 1], 50):.4f}, 75%={np.percentile(Y_update[:, 1], 75):.4f}")


        # Re-train the Multi-Task GP model with updated data (includes objective and credit)
        train_x = torch.tensor(D_update, dtype=dtype, device=device)
        train_y = torch.tensor(Y_update, dtype=dtype, device=device)
        model, likelihood = train_gp_model(train_x, train_y, training_iter=50, lr=0.1)


        # =============== (4) Update Minimizer List and Best Value ===============
        if dy == 0: # Noiseless: track minimum observed value
            current_best_idx = np.argmin(Y_update[:, 0])
            minimizer.append(D_update[current_best_idx, :])
            current_best_y = Y_update[current_best_idx, 0]
        else: # Noisy: track minimum predicted mean on observed data
            # Pass current credits for consistent prediction
            mu_train_tasks, _ = gp_predict(model, likelihood, D_update, credits=Y_update[:, 1])
            current_best_idx = np.argmin(mu_train_tasks[:, 0])
            minimizer.append(D_update[current_best_idx, :])
            current_best_y = mu_train_tasks[current_best_idx, 0] # Update estimated best y


        Training_time = time.time() - start
        # print(f"Iteration {i + 1} took {Training_time:.2f} seconds.") # Verbose timing
        TIME_RE.append(Training_time)

    end_all = time.time() - start_all
    print(f"******* Total Optimization Time: {end_all:.2f} seconds. *******")
    TIME_RE = np.array(TIME_RE).reshape(1, -1)

    # =============== Evaluate Convergence Performance ===============
    # Calculate sequence of minimum observed values up to each step
    minimum_observed_sequence = []
    minimizer_location_sequence = []
    for k in range(Y_update.shape[0]):
        best_idx_so_far = np.argmin(Y_update[: (k + 1), 0])
        minimum_observed_sequence.append(Y_update[best_idx_so_far, 0])
        minimizer_location_sequence.append(D_update[best_idx_so_far, :])

    minimizer_location_sequence = np.array(minimizer_location_sequence)
    minimum_observed_sequence = np.array(minimum_observed_sequence)

    # Construct the sequence of points believed to be the minimum at each step
    # Note: `minimizer` list stores the best point *after* each iteration (length = iteration+1)
    # The first n_sample points' best is derived from initial data
    minimizer_all = np.array(minimizer) # Contains best point after iter 0, 1, ...
    # Prepend the best points identified within the initial n_sample data
    initial_minimizers = minimizer_location_sequence[:n_sample]
    minimizer_all_full = np.concatenate((initial_minimizers, minimizer_all), axis=0)
    # Should have N_total + 1 points if minimizer includes initial state? Check length.
    # Let's stick to N_total points representing the best *after* each evaluation budget is spent.
    minimizer_all = minimizer_location_sequence # This seems more direct: best point found using budget 1..N_total

    print("Shape of sequence of minimizer locations:", minimizer_all.shape) # Should be (N_total, D)

    # Evaluate the true function value at the identified minimizers
    minimum_true_at_minimizers = np.zeros(N_total)
    for index in range(N_total):
        # Use the actual function f to get true value (even if noisy observations were used)
        minimum_true_at_minimizers[index] = f.evaluate_true(minimizer_all[index, :])

    minimum_true_at_minimizers = minimum_true_at_minimizers.reshape(-1, 1)

    # Calculate Absolute Gap to true optimum f_star
    GAP = np.abs(minimum_true_at_minimizers - f_star).reshape(-1, 1)

    # Calculate Simple Regret (minimum gap found up to budget i)
    SimRegret = []
    for i in range(len(GAP)):
        SimRegret.append(np.min(GAP[: (i + 1)]))
    SimRegret = np.array(SimRegret).reshape(-1, 1)

    # Calculate Instantaneous Regret for each evaluated point
    Regret = np.zeros(N_total)
    for index in range(N_total):
        Regret[index] = f.evaluate_true(D_update[index, :]) - f_star
    Regret = Regret.reshape(-1, 1)

    # Calculate Cumulative Regret
    CumRegret = np.cumsum(Regret).reshape(-1, 1)

    # Calculate Average Regret
    AvgRegret = CumRegret / np.arange(1, N_total + 1).reshape(-1, 1)

    # Calculate RMSE between true optimum and true values of evaluated points
    true_values_evaluated = np.array([f.evaluate_true(d) for d in D_update])
    # RMSE might not be the best metric here, maybe RMSE of minimum_true_at_minimizers vs f_star?
    # Let's calculate RMSE of the sequence of best found values vs f_star
    rmse = np.sqrt(np.mean((minimum_true_at_minimizers - f_star) ** 2))
    print(f"RMSE of best found values vs f_star: {rmse:.4f}")

    # Calculate distance to true minimizer x_star if available
    if x_star is not None:
        xGAP = []
        # Ensure x_star is a 2D array for consistent norm calculation
        x_star_arr = np.atleast_2d(x_star)
        for k in range(N_total):
             # Find distance between the best point found *up to budget k* and the closest true optimum
             dist = np.linalg.norm(minimizer_all[k, :] - x_star_arr, axis=1)
             xGAP.append(np.min(dist)) # Min distance if multiple x_star
        xGAP = np.array(xGAP).reshape(-1, 1)


    # =============== Save Results ===============
    # Create results directories if they don't exist
    if not os.path.exists("Results"):
        os.makedirs("Results")
    if not os.path.exists("Images"):
        os.makedirs("Images")

    # Get Number_Z from args for consistent file naming
    Number_Z = args.n_Z

    # Extract Objective and Credits separately
    Y_obj_only = Y_update[:, 0]
    all_credits = Y_update[:, 1]

    # Calculate unique points if needed for saving
    X_update = np.unique(D_update, axis=0)

    Budget = range(0, N_total) # Match BO_HCA_Sample range
    # Adapt filename to match BO_HCA_Sample.py format
    file_base = "_".join([
        date, str(Number_Z), select, func, algorithm,
        str(dy), str(n_sample), str(iteration), str(dimension_x)
    ])
    imagefile = os.path.join("Images", file_base)
    resultfile = os.path.join("Results", file_base) # Keep using os.path.join for robustness

    # Plotting (Keep the existing plotting logic)
    plt.figure(figsize=(18, 10)) # Wider figure
    plt.suptitle(f"{func} (D={dimension_x}), Algo={algorithm}, Heter={heter}, Noise={dy}, Init={n_sample}, Iter={iteration}", fontsize=14)

    plot_configs = [
        (1, "Simple Regret (Min Gap)", SimRegret),
        (2, "Cumulative Regret", CumRegret),
        (3, "Average Regret", AvgRegret),
        (4, "Best Found Value", minimum_true_at_minimizers),
        (5, "Gap |f(x_best) - f*|", GAP),
    ]
    plot_idx = 1
    for pos, ylabel, data in plot_configs:
        plt.subplot(2, 3, plot_idx)
        plt.plot(Budget, data) # Use Budget starting from 0 for plotting consistency if needed, or adjust range later
        plt.ylabel(ylabel)
        plt.xlabel("Budget (Evaluations)")
        plt.yscale('log') if ylabel in ["Simple Regret (Min Gap)", "Gap |f(x_best) - f*|"] else None # Log scale for regret/gap
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        if ylabel != "Best Found Value":
            plt.axhline(0, color="black", ls="--", linewidth=0.8)
        else:
            plt.axhline(f_star, color="red", ls=":", linewidth=1.0, label=f'f* = {f_star:.3f}')
            plt.legend()
        plot_idx += 1


    if x_star is not None:
        plt.subplot(2, 3, plot_idx)
        plt.plot(Budget, xGAP)
        plt.ylabel("Distance to x* ||x_best - x*||")
        plt.xlabel("Budget (Evaluations)")
        plt.yscale('log') # Log scale usually appropriate for distance
        plt.axhline(0, color="black", ls="--", linewidth=0.8)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plot_idx += 1

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    # Use resultfile base name for consistency, add repetition number
    plt.savefig(f"{imagefile}_{repet_time}.pdf", format="pdf")
    plt.close()

    # Saving data - Match BO_HCA_Sample.py format using 'ab' and transpose
    file_configs = [
        ("_X_update.csv", X_update, False),                 # Unique points
        ("_D_update.csv", D_update, False),                 # All evaluated points
        ("_Y_update.csv", Y_obj_only.reshape(-1, 1), True), # Objective values only, transpose needed
        ("_Credits.csv", all_credits.reshape(-1, 1), True), # Credit values, transpose needed
        ("_minimizer_all.csv", minimizer_all, False),       # Sequence of best points found
        ("_minimum.csv", minimum_true_at_minimizers, True), # True value at best points, transpose needed
        ("_Time.csv", TIME_RE, False),                      # Time per iteration (already 1xIter)
        ("_CumRegret.csv", CumRegret, True),                # Cumulative Regret, transpose needed
        ("_Regret.csv", Regret, True),                      # Instantaneous Regret, transpose needed
        ("_AvgRegret.csv", AvgRegret, True),                # Average Regret, transpose needed
        ("_SimRegret.csv", SimRegret, True),                # Simple Regret, transpose needed
        ("_GAP.csv", GAP, True),                            # Absolute Gap, transpose needed
        ("_RMSE.csv", np.array([[rmse]]), False)            # RMSE (already 1x1)
    ]
    if x_star is not None:
        file_configs.append(("_xGAP.csv", xGAP, True))      # Distance to x*, transpose needed

    for suffix, data, needs_transpose in file_configs:
        filepath = f"{resultfile}{suffix}" # Use the consistent resultfile base path
        with open(filepath, "ab") as f_out:
            # Reshape data if it's 1D and needs transpose, otherwise use as is or transpose if 2D+
            if data.ndim == 1 and needs_transpose:
                 output_data = data.reshape(1, -1) # Make it 1xN for saving as a row
            elif needs_transpose:
                 output_data = data.T # Transpose if already 2D+
            else:
                 output_data = data # Use as is if no transpose needed

            # Ensure output_data is at least 2D for savetxt when saving single rows from repetitions
            if output_data.ndim == 1:
                 output_data = output_data.reshape(1, -1) # Ensure single results like RMSE are saved as a row

            np.savetxt(f_out, output_data, delimiter=",")


#############################################
# Parameter Parsing and Main Execution Block
#############################################
parser = argparse.ArgumentParser(description='Integrated Bayesian Optimization with HCA-Credit, MTGP, and Biased Sampling')

# Dummy argument for compatibility with some environments (e.g., Jupyter)
parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")

# Experiment Identification
parser.add_argument('-t', '--time', type=str, help='Timestamp or ID for the experiment run', default='BO_HCA_ALL_v2')

# Function and Dimension
parser.add_argument('-func', '--function', type=str, help='Test function name', default='Hart6')
parser.add_argument('-dimensionx', '--dimension_x', type=int, help='Problem dimension (overridden by some functions)', default=6)

# BO Algorithm Settings
parser.add_argument('-algo', '--algorithm', type=str, help='Acquisition function: EI, TS, UCB', default="UCB")
parser.add_argument('-nsample', '--n_sample', type=int, help='Number of initial samples (LHS)', default=20)
parser.add_argument('-i', '--iteration', type=int, help='Number of optimization iterations', default=100) # Reduced default for quicker tests

# Noise and Heteroscedasticity Settings
parser.add_argument('-noise', '--noise_std', type=float, help='Base noise standard deviation for test function', default=0.1)
parser.add_argument('-heter', '--heter', help='Enable Heteroscedastic mechanism (MTGP + HCA + Biased Sampling)', action='store_true', default=True)

# Execution Control
parser.add_argument('-rep', '--repet_num', type=int, help='Total number of repetitions (macroreplications)', default=5) # Reduced default
parser.add_argument('-startrep', '--start_num', type=int, help='Starting repetition number (index)', default=0)
# parser.add_argument('-core', '--n_cores', type=int, help='Number of Cores (for parallel execution, not implemented here)', default=1)

# --- Arguments below might be less relevant with the integrated approach but kept for potential future use ---
parser.add_argument('-select', '--select_method', type=str, help='Selection method (historical, maybe unused)', default="HCA_MTGP_BiasedSample")
parser.add_argument('-Z', '--n_Z', type=int, help='Hyperparameter Z (historical, maybe unused)', default=1) # HCA uses Z=min(y) now
# parser.add_argument('-thre', '--threshold', type=float, help='Threshold (historical, maybe unused)', default=1e-4)
# parser.add_argument('-k', '--knn', type=int, help='K in KNN (historical, unused here)', default=5)


args = parser.parse_args()

# Function to run a single experiment repetition
def run_experiment(experiment_id):
    print(f"--- Starting Experiment Repetition {experiment_id} ---")
    start_rep_time = time.time()
    Experiments(experiment_id)
    end_rep_time = time.time()
    print(f"--- Experiment Repetition {experiment_id} completed in {end_rep_time - start_rep_time:.2f} seconds ---")

# Function to run multiple experiment repetitions
def run_experiments(start_rep, end_rep):
    print(f"Running experiments from repetition {start_rep} to {end_rep}...")
    for i in range(start_rep, end_rep): # end_rep is exclusive
        run_experiment(i)
    print("--- All experiment repetitions finished. ---")

if __name__ == "__main__":
    start_repetition = args.start_num
    end_repetition = args.start_num + args.repet_num # Calculate end based on start and number of reps
    run_experiments(start_repetition, end_repetition)

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