"""
New module for Euler Equation based objective functions.
Decoupled from Bellman module.
"""
import torch
import numpy as np
import module_basic_v1

# Load config independently
config = module_basic_v1.Config("config_v1.json")

class DefineEulerObjective:
    def __init__(self, model, device=None):
        self.device = device
        self.model = model.to(self.device)
        
        # Define ranges (Same logic as Bellman but independent implementation)
        keys = ["z", "a"]
        self.ranges = [(config.bounds[key]["min"], config.bounds[key]["max"]) for key in keys]
        # Extend the ranges for dist_a_mid
        for dist_a_pdf in config.dist_a_pdf:
            extended_min = dist_a_pdf * (1 - config.dist_a_band)
            extended_max = dist_a_pdf * (1 + config.dist_a_band)
            self.ranges.append((extended_min, extended_max))

    def extract_state_variables(self, x_batch):
        """
        Extract state variables from the input batch.
        """
        x_z = x_batch[:, 0].unsqueeze(1).to(self.device)
        x_a = x_batch[:, 1].unsqueeze(1).to(self.device)
        x_dist = x_batch[:, 2:].to(self.device)
        return x_z, x_a, x_dist

    def predict_policy(self, input_data):
        """
        Wrapper to call policy function from MyModel
        """
        if isinstance(self.model, torch.nn.DataParallel):
            return self.model.module.f_policy(input_data)
        else:
            return self.model.f_policy(input_data)

    def calculate_aggregates(self, x_tfp, x_z, x_a_total, x_int_z):
        """
        Calculate w, r, etc. Re-implemented here to avoid cross-import.
        """
        # Calculate wage component w0_1
        x_w_1 = (1 - config.alpha) * (x_a_total / x_int_z) ** config.alpha
        
        # Calculate wage w
        x_w = x_tfp * config.psi_l ** (config.alpha / config.theta_l) * x_w_1 ** (
                    config.theta_l / (config.alpha + config.theta_l))
        
        # Calculate labor l
        x_l = (x_w * x_z / config.psi_l) ** (1 / config.theta_l)
        
        # Calculate interest rate r
        x_r = x_tfp * config.alpha * (x_w / (1 - config.alpha)) ** ((config.alpha - 1) / config.alpha) - config.delta
        
        return x_w, x_l, x_r

    def get_euler_residuals(self, x_batch, n_mc_samples, dist_a_mid):
        """
        Core function for Method 2: Euler Equation Residual Minimization.
        Minimizes the error in the Euler equation using KKT conditions (Fischer-Burmeister).
        """
        # 1. Prepare Data
        n_batch = x_batch.size(0)
        x_z0, x_a0, x_dist0 = self.extract_state_variables(x_batch)
        
        # Prepare TFP (Current Step)
        tfp_grid = torch.tensor(config.tfp_grid).view(-1, 1).to(self.device)
        tfp_transition = torch.tensor(config.tfp_transition).view(config.n_tfp, config.n_tfp).to(self.device)
        
        # Randomly sample current TFP indices
        x_i_tfp0 = torch.randint(config.n_tfp, (n_batch, 1), device=self.device)
        x_tfp0 = tfp_grid[x_i_tfp0.squeeze()].view(-1, 1)

        # 2. Calculate Current Period Variables (t)
        # 2.1 Aggregates (K, w, r)
        x = 1 + 1 / config.theta_l
        x_int_z_const = torch.exp(torch.tensor(1 / 2 * x * x * config.sigma_z ** 2)).to(self.device)
        x_int_z = torch.full_like(x_z0, x_int_z_const)
        
        # Total Capital K = sum(dist * dist_mid)
        x_a0_total = (x_dist0 * dist_a_mid.T).sum(dim=1, keepdim=True)
        
        x_w0, x_l0, x_r0 = self.calculate_aggregates(x_tfp0, x_z0, x_a0_total, x_int_z)

        # 2.2 Predict Policy a' (k_prime)
        # Normalize Input
        x_x0_input = torch.cat([x_z0, x_a0, x_dist0], dim=1)
        x_x0_norm = module_basic_v1.normalize_inputs(x_x0_input, config.bounds)
        x_x0_norm_tfp = torch.cat((x_tfp0, x_x0_norm), dim=1)
        
        # Get raw policy output (0-1) and scale to range
        policy_out = self.predict_policy(x_x0_norm_tfp)
        a_min, a_max = self.ranges[1]
        x_a1 = policy_out[:, 0].unsqueeze(1) * (a_max - a_min)

        # 2.3 Calculate Consumption c
        # Budget Constraint: c + a' = (1+r)a + wl
        income = (1 + x_r0) * x_a0 + x_w0 * x_l0 * x_z0
        x_c0 = income - x_a1
        
        # Numerical stability: clamp c > 0
        x_c0_safe = torch.maximum(x_c0, torch.tensor(1e-6, device=self.device))
        u_prime_c0 = x_c0_safe ** (-config.sigma)

        # -------------------------------------------------------------------
        # 3. Monte Carlo Integration for Expectation (t+1)
        # We need to compute E[ beta * (1+r') * u'(c') ]
        # -------------------------------------------------------------------
        
        # 3.1 Expand tensors for MC sampling
        # Shape becomes: (n_batch * n_mc_samples, ...)
        x_z0_rep = x_z0.repeat_interleave(n_mc_samples, dim=0)
        x_a1_rep = x_a1.repeat_interleave(n_mc_samples, dim=0) # This is a0 for next period
        x_dist0_rep = x_dist0.repeat_interleave(n_mc_samples, dim=0) 
        # Note: We use x_dist0 as x_dist1 approximation for short-term Euler training 
        # (or one could implement G_batch logic here, but keeping it simple for stability first)
        x_dist1_rep = x_dist0_rep 

        # 3.2 Sample Next Shock z'
        z_min, z_max = self.ranges[0]
        x_z1 = module_basic_v1.bounded_log_normal_samples(
            config.mu_z, config.sigma_z, z_min, z_max, n_batch * n_mc_samples
        ).unsqueeze(1).to(self.device)

        # 3.3 Sample Next Shock TFP'
        x_i_tfp0_rep = x_i_tfp0.repeat_interleave(n_mc_samples, dim=0)
        transition_probs = tfp_transition[x_i_tfp0_rep.squeeze()]
        x_i_tfp1 = torch.multinomial(transition_probs, 1)
        x_tfp1 = tfp_grid[x_i_tfp1.squeeze()].view(-1, 1)

        # 3.4 Calculate Next Period Aggregates (t+1)
        x_int_z_rep = torch.full_like(x_z1, x_int_z_const)
        x_a1_total = (x_dist1_rep * dist_a_mid.T).sum(dim=1, keepdim=True)
        
        x_w1, x_l1, x_r1 = self.calculate_aggregates(x_tfp1, x_z1, x_a1_total, x_int_z_rep)

        # 3.5 Predict Policy a'' (k_double_prime)
        x_x1_input = torch.cat([x_z1, x_a1_rep, x_dist1_rep], dim=1)
        x_x1_norm = module_basic_v1.normalize_inputs(x_x1_input, config.bounds)
        x_x1_norm_tfp = torch.cat((x_tfp1, x_x1_norm), dim=1)
        
        policy_out_next = self.predict_policy(x_x1_norm_tfp)
        x_a2 = policy_out_next[:, 0].unsqueeze(1) * (a_max - a_min)

        # 3.6 Calculate Next Consumption c'
        income_next = (1 + x_r1) * x_a1_rep + x_w1 * x_l1 * x_z1
        x_c1 = income_next - x_a2
        x_c1_safe = torch.maximum(x_c1, torch.tensor(1e-6, device=self.device))

        # 3.7 Compute Euler RHS term inside Expectation
        # Term = beta * (1 + r') * u'(c')
        euler_term_next = config.beta * (1 + x_r1) * (x_c1_safe ** (-config.sigma))

        # -------------------------------------------------------------------
        # 4. Compute Loss
        # -------------------------------------------------------------------
        
        # 4.1 Average over MC samples to get Expectation
        # Reshape back to (n_batch, n_mc) and take mean dim=1
        euler_expected = euler_term_next.view(n_batch, n_mc_samples, 1).mean(dim=1)

        # 4.2 Calculate Difference
        # Euler Equation: u'(c) = beta * E[...]  => Diff = u'(c) - E[...]
        diff = u_prime_c0 - euler_expected

        # 4.3 Normalize Difference (Optional but recommended for scaling)
        # Using unit-free Euler error: 1 - RHS/LHS
        # diff_norm = 1.0 - euler_expected / u_prime_c0
        # However, for Fischer-Burmeister, raw diff is often more stable with scaling factors.
        # Let's use the standard difference form for KKT.

        # 4.4 Apply Fischer-Burmeister for Inequality Constraint a' >= 0
        # Condition: a' >= 0  AND  (u'(c) - E[...]) >= 0  AND  a' * (u'(c) - E[...]) = 0
        # FB(a, b) = a + b - sqrt(a^2 + b^2)
        # Here a = x_a1 (policy output), b = diff (Euler Residual)
        
        # Note: We scale diff to be comparable to a magnitude
        scaling = 1.0 
        fb_error = module_basic_v1.fischer_burmeister(x_a1, diff * scaling)

        # 4.5 Additional Penalty for c < 0 (Soft constraint)
        # x_c0 < 0 implies violation of budget feasibility
        penalty_c = torch.relu(-x_c0).pow(2).mean()

        # Final MSE Loss
        loss = fb_error.pow(2).mean() + 1000.0 * penalty_c

        return loss