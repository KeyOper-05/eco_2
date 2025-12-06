"""
module_obj_euler_v1.py
Rigorous implementation of Method 2 from 'euler_residual.pdf' (Page 43-44).
Structural Fix: Applies manual activations (Sigmoid/Softplus) on raw logits.
"""
import torch
import numpy as np
import module_basic_v1

config = module_basic_v1.Config("config_v1.json")

class DefineEulerObjective:
    def __init__(self, model, device=None):
        self.device = device
        self.model = model.to(self.device)
        
        # Define ranges
        keys = ["z", "a"]
        self.ranges = [(config.bounds[key]["min"], config.bounds[key]["max"]) for key in keys]
        for dist_a_pdf in config.dist_a_pdf:
            extended_min = dist_a_pdf * (1 - config.dist_a_band)
            extended_max = dist_a_pdf * (1 + config.dist_a_band)
            self.ranges.append((extended_min, extended_max))

    def fischer_burmeister(self, a, b):
        """
        FB Equation: Psi(a, b) = a + b - sqrt(a^2 + b^2)
        Satisfies: a>=0, b>=0, ab=0
        """
        return a + b - torch.sqrt(a**2 + b**2 + 1e-8)

    def extract_state_variables(self, x_batch):
        x_z = x_batch[:, 0].unsqueeze(1).to(self.device)
        x_a = x_batch[:, 1].unsqueeze(1).to(self.device)
        x_dist = x_batch[:, 2:].to(self.device)
        return x_z, x_a, x_dist

    def predict_raw_logits(self, input_data):
        """
        Returns raw logits from the model (No activation applied).
        """
        if isinstance(self.model, torch.nn.DataParallel):
            return self.model.module.f_policy(input_data)
        else:
            return self.model.f_policy(input_data)

    def calculate_aggregates(self, x_tfp, x_z, x_a_total, x_int_z):
        x_w_1 = (1 - config.alpha) * (x_a_total / x_int_z) ** config.alpha
        x_w = x_tfp * config.psi_l ** (config.alpha / config.theta_l) * x_w_1 ** (
                    config.theta_l / (config.alpha + config.theta_l))
        x_l = (x_w * x_z / config.psi_l) ** (1 / config.theta_l)
        x_r = x_tfp * config.alpha * (x_w / (1 - config.alpha)) ** ((config.alpha - 1) / config.alpha) - config.delta
        return x_w, x_l, x_r

    def get_euler_loss_method2(self, x_batch, n_mc_samples, dist_a_mid):
        """
        Loss = E [ FB(s, 1-h)^2 + v * (Ratio1 - h) * (Ratio2 - h) ]
        """
        # 1. Prepare Current State
        n_batch = x_batch.size(0)
        x_z0, x_a0, x_dist0 = self.extract_state_variables(x_batch)
        
        tfp_grid = torch.tensor(config.tfp_grid).view(-1, 1).to(self.device)
        tfp_transition = torch.tensor(config.tfp_transition).view(config.n_tfp, config.n_tfp).to(self.device)
        x_i_tfp0 = torch.randint(config.n_tfp, (n_batch, 1), device=self.device)
        x_tfp0 = tfp_grid[x_i_tfp0.squeeze()].view(-1, 1)

        # 2. Calculate Current Variables
        x_theta_param = 1 + 1 / config.theta_l
        x_int_z_const = torch.exp(torch.tensor(1 / 2 * x_theta_param**2 * config.sigma_z ** 2)).to(self.device)
        x_int_z = torch.full_like(x_z0, x_int_z_const)
        x_a0_total = (x_dist0 * dist_a_mid.T).sum(dim=1, keepdim=True)
        
        x_w0, x_l0, x_r0 = self.calculate_aggregates(x_tfp0, x_z0, x_a0_total, x_int_z)
        
        # Wealth (Cash on Hand)
        wealth_0 = (1 + x_r0) * x_a0 + x_w0 * x_l0 * x_z0

        # --- Neural Network Prediction (Raw Logits) ---
        x_x0_input = torch.cat([x_z0, x_a0, x_dist0], dim=1)
        x_x0_norm = module_basic_v1.normalize_inputs(x_x0_input, config.bounds)
        x_x0_norm_tfp = torch.cat((x_tfp0, x_x0_norm), dim=1)
        
        nn_logits = self.predict_raw_logits(x_x0_norm_tfp)
        
        # [STRUCTURAL FIX]: Apply Manual Activations
        # Dim 0: Savings Rate s in (0, 1) -> Sigmoid
        # 含义: a' = s * Wealth
        s_0 = torch.sigmoid(nn_logits[:, 0].unsqueeze(1))
        
        # Dim 1: Expectation h in (0, +inf) -> Softplus
        # 允许 h >= 1，从而允许 1-h <= 0，解除 KKT 死锁
        h_0 = torch.nn.functional.softplus(nn_logits[:, 1].unsqueeze(1))

        # Implied Consumption and Next Asset
        x_a1 = s_0 * wealth_0
        a_max = self.ranges[1][1]
        x_a1 = torch.min(x_a1, torch.tensor(a_max, device=self.device)) # Clamp for safety

        x_c0 = (1 - s_0) * wealth_0
        x_c0_safe = torch.maximum(x_c0, torch.tensor(1e-5, device=self.device))
        
        # 3. Double Sampling for Next State
        def get_euler_ratio_sample():
            # (a) Sample Next Shocks (n_mc_samples handled by broadcasting if needed, here keeping simple batch)
            z_min, z_max = self.ranges[0]
            x_z1 = module_basic_v1.bounded_log_normal_samples(
                config.mu_z, config.sigma_z, z_min, z_max, n_batch
            ).unsqueeze(1).to(self.device)
            
            transition_probs = tfp_transition[x_i_tfp0.squeeze()]
            x_i_tfp1 = torch.multinomial(transition_probs, 1)
            x_tfp1 = tfp_grid[x_i_tfp1.squeeze()].view(-1, 1)
            
            # (b) Next State Aggregates
            x_dist1 = x_dist0 # Short-term approx
            x_a1_total_next = (x_dist1 * dist_a_mid.T).sum(dim=1, keepdim=True)
            x_int_z_next = torch.full_like(x_z1, x_int_z_const)
            x_w1, x_l1, x_r1 = self.calculate_aggregates(x_tfp1, x_z1, x_a1_total_next, x_int_z_next)
            
            # (c) Next Wealth
            wealth_1 = (1 + x_r1) * x_a1 + x_w1 * x_l1 * x_z1
            
            # (d) Next Policy (Raw Logits -> Activation)
            x_x1_input = torch.cat([x_z1, x_a1, x_dist1], dim=1)
            x_x1_norm = module_basic_v1.normalize_inputs(x_x1_input, config.bounds)
            x_x1_norm_tfp = torch.cat((x_tfp1, x_x1_norm), dim=1)
            
            nn_logits_next = self.predict_raw_logits(x_x1_norm_tfp)
            s_1 = torch.sigmoid(nn_logits_next[:, 0].unsqueeze(1)) # Savings rate next
            
            # (e) Next Consumption
            x_c1 = (1 - s_1) * wealth_1
            x_c1_safe = torch.maximum(x_c1, torch.tensor(1e-5, device=self.device))
            
            # (f) Calculate Euler Ratio: beta * (1+r') * (c'/c)^(-sigma)
            # This is numerically stable
            c_ratio = x_c1_safe / x_c0_safe
            euler_ratio = config.beta * (1 + x_r1) * (c_ratio ** (-config.sigma))
            
            return euler_ratio

        ratio_1 = get_euler_ratio_sample()
        ratio_2 = get_euler_ratio_sample()

        # 4. Construct Loss (PDF Page 44)
        
        # Term 1: Fischer-Burmeister
        # a = s (savings rate) -> s >= 0
        # b = 1 - h -> h <= 1 (Euler Inequality)
        fb_a = s_0
        fb_b = 1.0 - h_0
        
        loss_fb = self.fischer_burmeister(fb_a, fb_b).pow(2)
        
        # Term 2: Variance Reduction
        v_weight = getattr(config, 'euler_weight_v', 1.0)
        loss_var = v_weight * (ratio_1 - h_0) * (ratio_2 - h_0)
        
        total_loss = torch.mean(loss_fb + loss_var)
        
        return total_loss