"""
module_training_euler_v1.py
Trainer specific for Method 2 (Euler Residuals).
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

config = module_basic_v1.Config("config_v1.json")

class EulerTrainer:
    def __init__(self, model, device=None, i_save=0):
        self.model = model
        self.device = device
        self.i_save = i_save
        self.pretrained_path = f'models/trained_policy_nn_{config.model_number_output}_euler.pth'

    def get_domain_sampler(self):
        keys = ["z", "a"]
        ranges = [(config.bounds[key]["min"], config.bounds[key]["max"]) for key in keys]
        for dist_a_pdf in config.dist_a_pdf:
            extended_min = dist_a_pdf * (1 - config.dist_a_band)
            extended_max = dist_a_pdf * (1 + config.dist_a_band)
            ranges.append((extended_min, extended_max))
        return module_basic_v1.DomainSampling(ranges, device=self.device)

    def train_policy(self, num_epochs, n_mc_samples, dist_a_mid):
        domain_sampler = self.get_domain_sampler()
        euler_objective = module_obj_euler_v1.DefineEulerObjective(self.model, self.device)
        
        # Get params as list
        if isinstance(self.model, torch.nn.DataParallel):
            params = list(self.model.module.policy_func.parameters())
        else:
            params = list(self.model.policy_func.parameters())
            
        l2_reg = getattr(config, 'l2_penalty', 1e-5)
        
        # Optimizer with Weight Decay
        optimizer = optim.Adam(params, lr=config.lr_euler, weight_decay=l2_reg)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)

        losses = []
        print(f"Start Euler Training (Method 2: FB+DoubleSampling) for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            x_data = domain_sampler.generate_samples(num_samples=config.batch_size_p * 20, num_k=config.k_dist)
            dataset = TensorDataset(x_data)
            data_loader = DataLoader(dataset, batch_size=config.batch_size_p, shuffle=True)
            
            epoch_loss_sum = 0.0
            total_batches = 0
            
            for batch_x, in data_loader:
                batch_x = batch_x.to(self.device)
                
                # Pass n_mc_samples (though logic uses double sampling, keep interface clean)
                loss = euler_objective.get_euler_loss_method2(batch_x, n_mc_samples, dist_a_mid)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()
                
                epoch_loss_sum += loss.item()
                total_batches += 1
            
            avg_loss = epoch_loss_sum / total_batches
            losses.append(np.log(avg_loss + 1e-12))
            scheduler.step(avg_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Avg Loss = {avg_loss:.6f}, LR = {optimizer.param_groups[0]['lr']}")

        self.plot_loss(losses)
        if self.i_save == 1:
            self.save_model()
            
        return self.model

    def plot_loss(self, losses):
        plt.figure()
        plt.plot(losses)
        plt.xlabel('Epochs')
        plt.ylabel('Log Loss')
        plt.title('Euler Method 2 Loss')
        if not os.path.exists('figures'):
            os.makedirs('figures')
        plt.savefig(f'figures/loss_euler_method2.png')
        plt.close()

    def save_model(self):
        if not os.path.exists('models'):
            os.makedirs('models')
        state_dict = self.model.state_dict()
        clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        torch.save(clean_state_dict, self.pretrained_path)
        print(f"Model saved to {self.pretrained_path}")