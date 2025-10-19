#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深度MRAS-Net训练脚本
- 使用增强版深度模型
- EMA权重更新
- 更强数据增强
- 学习率调度
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

# 添加tools目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'tools'))

from enhanced_mra_net_deep import DeepMRANet, DeepMRANetLoss
from logger import get_logger

class DeepMRANetTrainer:
    """深度MRAS-Net训练器"""
    
    def __init__(self, 
                 model: DeepMRANet,
                 device: str = 'cuda',
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5,
                 use_amp: bool = True):
        
        self.model = model.to(device)
        self.device = device
        self.use_amp = use_amp
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        # 损失函数
        self.criterion = DeepMRANetLoss(
            mse_weight=1.0,
            ssim_weight=0.1,
            edge_weight=0.05,
            psf_weight=0.01
        )
        
        # AMP
        if use_amp and device == 'cuda':
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        self.logger = get_logger(__name__)
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, data in enumerate(pbar):
            if len(data) == 3:
                clean, blurred, psf = data
            else:
                clean, blurred = data
                psf = None
            
            clean = clean.to(self.device)
            blurred = blurred.to(self.device)
            
            self.optimizer.zero_grad()
            
            # 前向传播
            if self.scaler is not None:
                with torch.amp.autocast('cuda'):
                    restored, pred_psf = self.model(blurred)
                    loss = self.criterion(restored, clean, pred_psf)
                
                # 反向传播
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                restored, pred_psf = self.model(blurred)
                loss = self.criterion(restored, clean, pred_psf)
                
                loss.backward()
                self.optimizer.step()
            
            # 更新EMA权重
            self.model.update_ema_weights()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        return total_loss / num_batches
    
    def validate(self, val_loader: DataLoader) -> float:
        """验证"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for data in val_loader:
                if len(data) == 3:
                    clean, blurred, psf = data
                else:
                    clean, blurred = data
                    psf = None
                
                clean = clean.to(self.device)
                blurred = blurred.to(self.device)
                
                # 使用EMA权重进行验证
                restored, pred_psf = self.model.forward_ema(blurred)
                loss = self.criterion(restored, clean, pred_psf)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              epochs: int = 50,
              save_dir: str = "outputs/models") -> str:
        """训练模型"""
        
        os.makedirs(save_dir, exist_ok=True)
        
        self.logger.info(f"开始训练深度MRAS-Net，共{epochs}个epoch")
        
        for epoch in range(epochs):
            # 训练
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # 验证
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            # 学习率调度
            self.scheduler.step()
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_model_path = os.path.join(save_dir, "best_deep_mra_net.pth")
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'ema_weights': self.model.ema_weights,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss
                }, best_model_path)
                self.logger.info(f"保存最佳模型，验证损失: {val_loss:.6f}")
            
            # 日志
            current_lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(f"Epoch {epoch+1}/{epochs}, "
                           f"训练损失: {train_loss:.6f}, "
                           f"验证损失: {val_loss:.6f}, "
                           f"学习率: {current_lr:.2e}")
        
        # 保存最终模型
        final_model_path = os.path.join(save_dir, "final_deep_mra_net.pth")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'ema_weights': self.model.ema_weights,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': epochs,
            'val_loss': self.val_losses[-1]
        }, final_model_path)
        
        # 绘制训练历史
        self._plot_training_history(save_dir)
        
        self.logger.info(f"深度MRAS-Net训练完成，最佳模型保存到: {best_model_path}")
        return best_model_path
    
    def _plot_training_history(self, save_dir: str):
        """绘制训练历史"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='训练损失')
        plt.plot(self.val_losses, label='验证损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('训练历史')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.train_losses, label='训练损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('训练损失')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        history_path = os.path.join(save_dir, "deep_mra_net_training_history.png")
        plt.savefig(history_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"训练历史图表已保存到: {history_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="深度MRAS-Net训练")
    parser.add_argument("--train-path", default="outputs/datasets/train_dataset.h5")
    parser.add_argument("--val-path", default="outputs/datasets/val_dataset.h5")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-stages", type=int, default=16)
    parser.add_argument("--hidden-channels", type=int, default=192)
    parser.add_argument("--no-amp", action="store_true", help="禁用AMP")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--pin-memory", action="store_true")
    
    args = parser.parse_args()
    
    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 创建模型
    model = DeepMRANet(
        num_stages=args.num_stages,
        hidden_channels=args.hidden_channels,
        use_ema=True
    )
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 数据加载器
    train_dataset = h5py.File(args.train_path, 'r')
    val_dataset = h5py.File(args.val_path, 'r')
    
    train_loader = DataLoader(
        [(train_dataset['clean_images'][i], train_dataset['blurred_images'][i]) 
         for i in range(len(train_dataset['clean_images']))],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory and device == 'cuda'
    )
    
    val_loader = DataLoader(
        [(val_dataset['clean_images'][i], val_dataset['blurred_images'][i]) 
         for i in range(len(val_dataset['clean_images']))],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory and device == 'cuda'
    )
    
    # 训练器
    trainer = DeepMRANetTrainer(
        model=model,
        device=device,
        learning_rate=args.learning_rate,
        use_amp=not args.no_amp
    )
    
    # 开始训练
    best_model_path = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs
    )
    
    print(f"训练完成！最佳模型: {best_model_path}")


if __name__ == "__main__":
    main()
