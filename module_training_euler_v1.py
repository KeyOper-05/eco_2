"""
New module for training using Euler Equation method.
Decoupled from Bellman Trainer.
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import os

import module_basic_v1
import module_obj_euler_v1

# Load config
config = module_basic_v1.Config("config_v1.json")

class EulerTrainer:
    def __init__(self, model, device=None, i_save=0):
        self.model = model
        self.device = device
        self.i_save = i_save
        self.pretrained_policy_path = f'models/trained_policy_nn_{config.model_number_output}.pth'

    def get_domain_sampler(self):
        # Helper to get sampler from basic module
        keys = ["z", "a"]
        ranges = [(config.bounds[key]["min"], config.bounds[key]["max"]) for key in keys]
        for dist_a_pdf in config.dist_a_pdf:
            extended_min = dist_a_pdf * (1 - config.dist_a_band)
            extended_max = dist_a_pdf * (1 + config.dist_a_band)
            ranges.append((extended_min, extended_max))
        return module_basic_v1.DomainSampling(ranges, device=self.device)

    def load_pretrained_policy(self):
        if self.pretrained_policy_path is not None and os.path.exists(self.pretrained_policy_path):
            print(f"Loading pretrained policy from {self.pretrained_policy_path}")
            # Logic to load state dict (simplified)
            state_dict = torch.load(self.pretrained_policy_path, map_location=self.device)
            # Remove 'module.' prefix if it exists (from DataParallel)
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            # Filter for policy_func only
            policy_dict = {k.replace('policy_func.', ''): v for k, v in new_state_dict.items() if 'policy_func' in k}
            if hasattr(self.model, 'module'):
                self.model.module.policy_func.load_state_dict(policy_dict)
            else:
                self.model.policy_func.load_state_dict(policy_dict)

    def train_policy(self, num_epochs, n_mc_samples, dist_a_mid):
        """
        Main training loop for Euler Method.
        Only trains the policy network.
        """
        # 1. Setup
        domain_sampler = self.get_domain_sampler()
        euler_objective = module_obj_euler_v1.DefineEulerObjective(self.model, self.device)
        
        # Optimizer only for Policy Function
        if isinstance(self.model, torch.nn.DataParallel):
            params = self.model.module.policy_func.parameters()
        else:
            params = self.model.policy_func.parameters()
            
        optimizer = optim.Adam(params, lr=config.lr_euler)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)

        losses = []
        
        print(f"Start Euler Training for {num_epochs} epochs...")

        # 2. Training Loop
        for epoch in range(num_epochs):
            # Generate Data
            # Use larger samples for Euler as it is cheaper than simulation
            x_data = domain_sampler.generate_samples(num_samples=config.batch_size_p * 50, num_k=config.k_dist)
            dataset = TensorDataset(x_data)
            data_loader = DataLoader(dataset, batch_size=config.batch_size_p, shuffle=True)
            
            epoch_loss_sum = 0.0
            total_batches = 0
            
            for batch_x, in data_loader:
                batch_x = batch_x.to(self.device)
                
                # Calculate Euler Loss
                loss = euler_objective.get_euler_residuals(batch_x, n_mc_samples, dist_a_mid)
                
                # Update
                optimizer.zero_grad()
                loss.backward()
                
                # Clip gradients to prevent explosion (common in Euler method)
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                
                optimizer.step()
                
                epoch_loss_sum += loss.item()
                total_batches += 1
            
            # End of Epoch
            avg_loss = epoch_loss_sum / total_batches
            losses.append(np.log(avg_loss + 1e-10)) # Log loss for plotting
            
            scheduler.step(avg_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Avg Euler Loss = {avg_loss:.6f}, LR = {optimizer.param_groups[0]['lr']}")

        # 3. Finish & Save
        self.plot_loss(losses)
        
        if self.i_save == 1:
            self.save_model()
            
        return self.model

    def plot_loss(self, losses):
        plt.figure()
        plt.plot(losses)
        plt.xlabel('Epochs')
        plt.ylabel('Log Loss')
        plt.title('Euler Equation Residual Minimization')
        if not os.path.exists('figures'):
            os.makedirs('figures')
        plt.savefig(f'figures/loss_euler_method.png')
        plt.close()

    def save_model(self):
        if not os.path.exists('models'):
            os.makedirs('models')
        # Helper to remove module prefix
        state_dict = self.model.state_dict()
        clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        torch.save(clean_state_dict, f'models/trained_policy_nn_{config.model_number_output}_euler.pth')
        print("Model saved.")