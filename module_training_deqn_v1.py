"""
module_training_deqn_v1.py
Trainer for DEQN Algorithm with Monitoring and Live Plotting.
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm  # [新增] 引入进度条

import module_basic_v1
import module_obj_deqn_v1

config = module_basic_v1.Config("config_v1.json")

class DEQNTrainer:
    def __init__(self, model, device=None, i_save=0):
        self.model = model
        self.device = device
        self.i_save = i_save
        self.pretrained_path = f'models/trained_policy_nn_{config.model_number_output}_deqn.pth'

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
        deqn_objective = module_obj_deqn_v1.DefineDEQNObjective(self.model, self.device)
        
        # Policy Params
        if isinstance(self.model, torch.nn.DataParallel):
            params = list(self.model.module.policy_func.parameters())
        else:
            params = list(self.model.policy_func.parameters())
            
        l2_reg = getattr(config, 'l2_penalty', 1e-5)
        
        optimizer = optim.Adam(params, lr=config.deqn_lr, weight_decay=l2_reg)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)

        losses = []
        print(f"Start DEQN Training for {num_epochs} epochs with {n_mc_samples} MC samples...")

        # [新增] 实例化绘图器，用于训练中途绘图
        # 我们只采样 2000 个点用于快速预览，避免拖慢训练
        plotter = module_basic_v1.plot_equm_funcs(num_samples=2000, num_k=config.k_dist, 
                                                  dist_a_mid=dist_a_mid, model=self.model, device=self.device)

        # [新增] 使用 tqdm 包装循环，显示进度条
        pbar = tqdm(range(num_epochs), desc="DEQN Training")
        
        for epoch in pbar:
            # Generate fresh data
            x_data = domain_sampler.generate_samples(num_samples=config.batch_size_p * 20, num_k=config.k_dist)
            dataset = TensorDataset(x_data)
            data_loader = DataLoader(dataset, batch_size=config.batch_size_p, shuffle=True)
            
            epoch_loss_sum = 0.0
            total_batches = 0
            
            for batch_x, in data_loader:
                batch_x = batch_x.to(self.device)
                
                # Calculate DEQN Residuals
                loss = deqn_objective.get_deqn_residuals(batch_x, n_mc_samples, dist_a_mid)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()
                
                epoch_loss_sum += loss.item()
                total_batches += 1
            
            avg_loss = epoch_loss_sum / total_batches
            losses.append(np.log(avg_loss + 1e-15))
            scheduler.step(avg_loss)
            
            # [新增] 更新进度条信息
            pbar.set_postfix({'Loss': f'{avg_loss:.6f}', 'LR': f'{optimizer.param_groups[0]["lr"]:.1e}'})

            # [新增] 每隔 10 个 epoch 绘图一次并保存模型
            if epoch % 10 == 0:
                # 绘图 (会覆盖 figures/scatter_...png，实现动态更新)
                plotter.create_plot()
                
                # 绘制 Loss 曲线
                self.plot_loss(losses)
                
                # 保存临时模型 (Checkpoint)
                if self.i_save == 1:
                    self.save_model(suffix=f"_ep{epoch}")

        # 训练结束后的最终保存
        self.plot_loss(losses)
        if self.i_save == 1:
            self.save_model()
            
        return self.model

    def plot_loss(self, losses):
        plt.figure()
        plt.plot(losses)
        plt.xlabel('Epochs')
        plt.ylabel('Log Loss')
        plt.title('DEQN Residual Loss')
        if not os.path.exists('figures'):
            os.makedirs('figures')
        plt.savefig(f'figures/loss_deqn.png')
        plt.close()

    def save_model(self, suffix=""):
        if not os.path.exists('models'):
            os.makedirs('models')
        state_dict = self.model.state_dict()
        clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        # 支持保存带有 epoch 后缀的文件，方便回溯
        path = self.pretrained_path.replace(".pth", f"{suffix}.pth") if suffix else self.pretrained_path
        torch.save(clean_state_dict, path)