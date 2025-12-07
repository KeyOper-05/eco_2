"""
module_basic_v1.py
Basic module to setup the model.
Includes NNs, Sampling, and the Unified Prediction Adapter.
"""

import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import json
from torch.distributions.log_normal import LogNormal

# Load configuration from the JSON file
class Config:
    def __init__(self, config_file):
        with open(config_file, "r") as json_file:
            self.config = json.load(json_file)

        # extract all parameters
        for key, value in self.config.items():
            setattr(self, key, value)

    def get_z_bounds(self):
        z_bounds = self.bounds.get("z", {})
        return z_bounds.get("min"), z_bounds.get("max")


class MyModel(nn.Module):
    """
    define the neural networks:
    """
    class MyReLU(nn.Module):
        def forward(self, x):
            return nn.functional.relu(x) + 1

    def __init__(self, n_input, n_p_output, n_v_output, n1_p, n2_p, n1_v, n2_v):
        super(MyModel, self).__init__()

        self.my_relu = self.MyReLU()

        # Define Policy Function
        # [修改] 移除 Sigmoid，输出 Raw Logits，由外部(Adapter/Loss)决定激活函数
        self.policy_func = nn.Sequential(
            nn.Linear(n_input, n1_p),
            nn.ReLU(),
            nn.Linear(n1_p, n2_p),
            nn.ReLU(),
            nn.Linear(n2_p, n_p_output)
        )

        # Define Value Function
        self.value_func = nn.Sequential(
            nn.Linear(n_input, n1_v),
            nn.ReLU(),
            nn.Linear(n1_v, n2_v),
            nn.ReLU(),
            nn.Linear(n2_v, n_v_output)
        )

    def f_policy(self, x):
        return self.policy_func(x)

    def f_value(self, x):
        return self.value_func(x)


class DomainSampling:
    def __init__(self, ranges, device=None):
        self.ranges = ranges
        self.device = device
        self.config = Config("config_v1.json")

    def generate_samples(self, num_samples, num_k):
        '''num_k is the dim of the last columns needs to be normalized as they represent the distribution'''
        keys = ["z", "a"]
        ranges = [(self.config.bounds[key]["min"], self.config.bounds[key]["max"]) for key in keys]
        # Extend the ranges for dist_a_pdf
        for dist_a_pdf in self.config.dist_a_pdf:
            extended_min = dist_a_pdf * (1 - self.config.dist_a_band)
            extended_max = dist_a_pdf * (1 + self.config.dist_a_band)
            ranges.append((extended_min, extended_max))

        n_states = len(ranges)
        samples_tensor = torch.empty(num_samples, n_states, dtype=torch.float32, device=self.device)

        # Parameters for the lognormal distribution
        mu, sigma = self.config.mu_z, self.config.sigma_z
        z_min, z_max = ranges[0]
        log_normal = LogNormal(mu, sigma)

        # Efficient generation of the first column (z)
        z_samples = torch.zeros(num_samples, dtype=torch.float32, device=self.device)
        for i in range(num_samples):
            sample_accepted = False
            attempts = 0
            while not sample_accepted and attempts < 1000:
                sample = log_normal.sample()
                if z_min <= sample <= z_max:
                    z_samples[i] = sample
                    sample_accepted = True
                attempts += 1
            if not sample_accepted:
                z_samples[i] = torch.rand(1, device=self.device) * (z_max - z_min) + z_min

        samples_tensor[:, 0] = z_samples

        # Generate other columns
        for i, (lower, upper) in enumerate(ranges[1:], start=1):
            samples_tensor[:, i] = torch.rand(num_samples, dtype=torch.float32, device=self.device) * (
                        upper - lower) + lower

        # Normalize the last num_k columns
        sum_last_k = samples_tensor[:, -num_k:].sum(dim=1, keepdim=True)
        samples_tensor[:, -num_k:] /= sum_last_k

        return samples_tensor

    def generate_samples_a_pdf(self, n_batch, num_k):
        ranges = []
        for dist_a_pdf in self.config.dist_a_pdf[-num_k:]:
            extended_min = dist_a_pdf * (1 - self.config.dist_a_band_path)
            extended_max = dist_a_pdf * (1 + self.config.dist_a_band_path)
            ranges.append((extended_min, extended_max))

        samples_tensor = torch.empty(n_batch, num_k, dtype=torch.float32, device=self.device)
        for i, (lower, upper) in enumerate(ranges):
            samples_tensor[:, i] = torch.rand(n_batch, dtype=torch.float32, device=self.device) * (
                        upper - lower) + lower

        sum_last_k = samples_tensor.sum(dim=1, keepdim=True)
        normalized_samples = samples_tensor / sum_last_k
        return normalized_samples

    def dist_enforce_boundaries(self, x_dist1, a_pdf_penalty):
        n_batch, n_dim = x_dist1.shape
        dist_a_pdf_tensor = torch.tensor(self.config.dist_a_pdf, device=self.device)
        extended_min = dist_a_pdf_tensor * (1 - self.config.dist_a_band)
        extended_max = dist_a_pdf_tensor * (1 + self.config.dist_a_band)

        penalty_below = a_pdf_penalty * (extended_min[None, :] - x_dist1).clamp(min=0)
        penalty_above = a_pdf_penalty * (x_dist1 - extended_max[None, :]).clamp(min=0)
        total_penalty = a_pdf_penalty * (penalty_below + penalty_above).sum(dim=1).unsqueeze(-1)
        x_dist1_clamped = x_dist1.clamp(min=extended_min[None, :], max=extended_max[None, :])

        return x_dist1_clamped, total_penalty

class plot_equm_funcs:
    def __init__(self, num_samples, num_k, dist_a_mid, model, device=None):
        self.num_samples = num_samples
        self.num_k = num_k
        self.dist_a_mid = dist_a_mid
        self.model = model
        self.device = device
        self.config = Config("config_v1.json")
        self.initialize_ranges()

    def initialize_ranges(self):
        keys = ["z", "a"]
        self.ranges = [(self.config.bounds[key]["min"], self.config.bounds[key]["max"]) for key in keys]

    def generate_samples_fixed_k(self):
        config = self.config
        keys = ["z", "a"]
        ranges = [(config.bounds[key]["min"], config.bounds[key]["max"]) for key in keys]
        for dist_a_pdf in config.dist_a_pdf:
            extended_min = dist_a_pdf * (1 - config.dist_a_band)
            extended_max = dist_a_pdf * (1 + config.dist_a_band)
            ranges.append((extended_min, extended_max))

        samples_tensor = torch.empty(self.num_samples, len(ranges), dtype=torch.float32, device=self.device)
        mu, sigma = config.mu_z, config.sigma_z
        z_bounds = config.bounds["z"]
        log_normal = LogNormal(mu, sigma)

        for i in range(self.num_samples):
            z_sample = log_normal.sample()
            while not (z_bounds["min"] <= z_sample <= z_bounds["max"]):
                z_sample = log_normal.sample()
            samples_tensor[i, 0] = z_sample
            for j, (lower, upper) in enumerate(ranges[1:], start=1):
                samples_tensor[i, j] = torch.rand(1, device=self.device) * (upper - lower) + lower

        sum_last_k = samples_tensor[0, -self.num_k:].sum()
        samples_tensor[:, -self.num_k:] = samples_tensor[0, -self.num_k:] / sum_last_k
        return samples_tensor

    def extract_state_variables(self, x_sample):
        x_z = x_sample[:, 0].unsqueeze(1).to(self.device)
        x_a = x_sample[:, 1].unsqueeze(1).to(self.device)
        x_dist = x_sample[:, 2:].to(self.device)
        return x_z, x_a, x_dist

    def create_plot(self):
        config = self.config
        a_min, a_max = self.ranges[1]

        samples_tensor = self.generate_samples_fixed_k()
        x_z0, x_a0, x_dist0 = self.extract_state_variables(samples_tensor)

        x_i_tfp0 = torch.zeros_like(x_z0).long().to(self.device)
        tfp_grid = torch.tensor(config.tfp_grid).view(-1, 1).to(self.device)
        x_tfp0 = tfp_grid[x_i_tfp0.squeeze()].view(-1, 1)

        x_x0_policy = torch.cat([x_z0, x_a0, x_dist0], 1).to(self.device)
        x_x0_policy_sd = normalize_inputs(x_x0_policy, config.bounds)
        x_x0_policy_sd = torch.cat((x_tfp0, x_x0_policy_sd), dim=1)

        # [调用统一适配器]
        x_a1_norm = predict_model_adapter(self.model, x_x0_policy_sd, config, self.dist_a_mid, self.device)
        x_a1 = x_a1_norm * (a_max - a_min)

        # Calculate Consumption for plot
        x_int_z = torch.full_like(x_z0, np.exp(0.5 * (1 + 1/config.theta_l)**2 * config.sigma_z**2))
        dist_a_mid_tensor = self.dist_a_mid.view(1, -1) # Ensure correct shape for calc
        x_a0_total = (x_dist0 * dist_a_mid_tensor).sum(dim=1, keepdim=True)
        x_w0, x_l0, x_r0 = calculate_aggregates_static(x_tfp0, x_z0, x_a0_total, x_int_z, config)
        
        x_c0 = (1 + x_r0) * x_a0 + x_w0 * x_l0 * x_z0 - x_a1

        # Calculate Value
        if isinstance(self.model, torch.nn.DataParallel):
            x_v = self.model.module.f_value(x_x0_policy_sd)[:, 0].unsqueeze(1)
        else:
            x_v = self.model.f_value(x_x0_policy_sd)[:, 0].unsqueeze(1)

        x1 = x_z0.cpu().numpy()
        x2 = x_a0.cpu().numpy()
        z_c = x_c0.detach().cpu().numpy()
        z_a = x_a1.detach().cpu().numpy()
        z_v = x_v.detach().cpu().numpy()

        # Save plots
        if not os.path.exists('figures'):
            os.makedirs('figures')

        fig1 = plt.figure(figsize=(9, 9))
        ax1 = fig1.add_subplot(111, projection='3d')
        ax1.scatter(x1, x2, z_c, c='blue', marker='o')
        ax1.set_xlabel('z'); ax1.set_ylabel('a'); ax1.set_zlabel('c')
        plt.savefig(f'figures/scatter_policy_c.png'); plt.close()

        fig2 = plt.figure(figsize=(9, 9))
        ax2 = fig2.add_subplot(111, projection='3d')
        ax2.scatter(x1, x2, z_a, c='red', marker='o')
        ax2.set_xlabel('z'); ax2.set_ylabel('a'); ax2.set_zlabel('a+')
        plt.savefig(f'figures/scatter_policy_a1.png'); plt.close()

        fig3 = plt.figure(figsize=(9, 9))
        ax3 = fig3.add_subplot(111, projection='3d')
        ax3.scatter(x1, x2, z_v, c='green', marker='o')
        ax3.set_xlabel('z'); ax3.set_ylabel('a'); ax3.set_zlabel('V')
        plt.savefig(f'figures/scatter_value.png'); plt.close()


# --- Helper Functions (Shared) ---

def normalize_inputs(inputs, normalization_bounds):
    n, total_columns = inputs.shape
    outputs = torch.zeros_like(inputs)
    for i in range(2):
        key = list(normalization_bounds.keys())[i]
        min_val = normalization_bounds[key]["min"]
        max_val = normalization_bounds[key]["max"]
        outputs[:, i] = (inputs[:, i] - min_val) / (max_val - min_val)
    k_columns = inputs[:, 2:]
    row_sums = k_columns.sum(dim=1, keepdim=True)
    outputs[:, 2:] = k_columns / row_sums
    return outputs

def bounded_log_normal_samples(mu, sigma, lower_bound, upper_bound, num_samples):
    log_normal = LogNormal(mu, sigma)
    samples = torch.empty(num_samples)
    for i in range(num_samples):
        sample_accepted = False
        while not sample_accepted:
            sample = log_normal.sample().item()
            if lower_bound <= sample <= upper_bound:
                samples[i] = sample
                sample_accepted = True
    return samples

def fischer_burmeister(a, b):
    return a + b - torch.sqrt(a**2 + b**2 + 1e-8)

def calculate_aggregates_static(x_tfp, x_z, x_a_total, x_int_z, config):
    x_w_1 = (1 - config.alpha) * (x_a_total / x_int_z) ** config.alpha
    x_w = x_tfp * config.psi_l ** (config.alpha / config.theta_l) * x_w_1 ** (
                config.theta_l / (config.alpha + config.theta_l))
    x_l = (x_w * x_z / config.psi_l) ** (1 / config.theta_l)
    x_r = x_tfp * config.alpha * (x_w / (1 - config.alpha)) ** ((config.alpha - 1) / config.alpha) - config.delta
    return x_w, x_l, x_r

def predict_model_adapter(model, input_data, config, dist_a_mid_tensor, device):
    """
    UNIFIED ADAPTER.
    Converts model outputs (Logits) to Normalized Asset Level (a').
    """
    if isinstance(model, torch.nn.DataParallel):
        output = model.module.f_policy(input_data)
    else:
        output = model.f_policy(input_data)

    solver_method = getattr(config, 'solver_method', 'bellman')
    s = None

    # Case 1: Euler Method 2 (Output Dim=2 -> [s, h])
    if output.shape[1] == 2:
        s = torch.sigmoid(output[:, 0].unsqueeze(1))
        
    # Case 2: DEQN or Bellman (Output Dim=1)
    elif output.shape[1] == 1:
        if solver_method == 'deqn':
            s = torch.sigmoid(output)
        else:
            return output # Bellman is already a'

    # If we have Savings Rate 's', convert to a'
    if s is not None:
        # Reconstruct Wealth
        x_tfp = input_data[:, 0:1]
        z_min, z_max = config.bounds["z"]["min"], config.bounds["z"]["max"]
        x_z = input_data[:, 1:2] * (z_max - z_min) + z_min
        a_min, a_max = config.bounds["a"]["min"], config.bounds["a"]["max"]
        x_a = input_data[:, 2:3] * (a_max - a_min) + a_min
        x_dist = input_data[:, 3:]
        
        # [Fix] Ensure dist_a_mid_tensor is (1, K) for broadcasting with (N, K)
        dist_a_mid_tensor = dist_a_mid_tensor.view(1, -1)
        x_a_total = (x_dist * dist_a_mid_tensor).sum(dim=1, keepdim=True)
        
        x_term = 1 + 1 / config.theta_l
        int_z_val = np.exp(0.5 * (x_term * config.sigma_z) ** 2)
        x_int_z = torch.full_like(x_z, int_z_val)
        
        x_w, x_l, x_r = calculate_aggregates_static(x_tfp, x_z, x_a_total, x_int_z, config)
        wealth = (1 + x_r) * x_a + x_w * x_l * x_z
        
        # a' = s * wealth
        a_prime = s * wealth
        
        # Normalize a' back to [0, 1]
        a_prime_norm = (a_prime - a_min) / (a_max - a_min)
        return torch.clamp(a_prime_norm, 0.0, 1.0)

    return output