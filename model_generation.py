#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型生成脚本

功能：
1. 训练U-Net模型
2. 训练MRA-Net模型
3. 模型评估和保存
python model_generation.py --mode both --epochs_unet 50 --epochs_mranet 30
使用方法：
    python model_generation.py
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import h5py
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import contextlib

# 添加tools目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'tools'))

from unet_model import UNet
from enhanced_mra_net_model import EnhancedMRANet, EnhancedMRANetLoss
from mras_net_model import MRANetLoss
from logger import get_logger

class ImageDataset(Dataset):
    """图像数据集类 - 优化版本"""
    
    def __init__(self, dataset_path: str, transform=None, preload=True, target_size=(256, 256)):
        """
        初始化图像数据集
        
        Args:
            dataset_path: HDF5数据集路径
            transform: 数据增强转换
            preload: 是否预加载所有数据到内存，多进程加载时必须为True
            target_size: 目标图像尺寸
        """
        self.dataset_path = dataset_path
        self.transform = transform
        self.target_size = target_size
        self.psf_target_size = (64, 64)  # PSF目标尺寸
        
        # 强制预加载模式，解决h5py对象不能被序列化的问题
        # 这对于多进程数据加载是必要的
        with h5py.File(dataset_path, 'r') as h5_file:
            self.clean_images = h5_file['clean_images'][:]
            self.blurred_images = h5_file['blurred_images'][:]
            
            # 检查是否有PSF数据
            self.has_psfs = 'psfs' in h5_file
            if self.has_psfs:
                self.psfs = []
                psfs_group = h5_file['psfs']
                # 预处理并缓存所有PSF
                for i in range(len(self.clean_images)):
                    psf_key = f'item_{i}'
                    if psf_key in psfs_group:
                        psf_img = psfs_group[psf_key][:]
                        # 预先调整PSF尺寸
                        if psf_img.shape != self.psf_target_size:
                            psf_img = cv2.resize(psf_img, self.psf_target_size, interpolation=cv2.INTER_LINEAR)
                        self.psfs.append(psf_img)
                    else:
                        self.psfs.append(None)
            else:
                self.psfs = None
    
    def __len__(self):
        return len(self.clean_images)
    
    def __getitem__(self, idx):
        clean_img = self.clean_images[idx]
        blurred_img = self.blurred_images[idx]
        
        # 确保图像尺寸一致 - 调整为相同大小
        if clean_img.shape != self.target_size:
            clean_img = cv2.resize(clean_img, self.target_size, interpolation=cv2.INTER_LINEAR)
        
        if blurred_img.shape != self.target_size:
            blurred_img = cv2.resize(blurred_img, self.target_size, interpolation=cv2.INTER_LINEAR)
        
        # 转换为张量并添加通道维度
        clean = torch.from_numpy(clean_img).float().unsqueeze(0)
        blurred = torch.from_numpy(blurred_img).float().unsqueeze(0)
        
        # 处理PSF数据
        if self.has_psfs and self.psfs is not None and idx < len(self.psfs):
            psf_img = self.psfs[idx]
            if psf_img is not None:
                psf = torch.from_numpy(psf_img).float().unsqueeze(0)
                return blurred, clean, psf
        
        return blurred, clean

class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, device: str = None, output_dir: str = "outputs/models",
                 amp_mode: str = 'auto', num_workers: int = 4, pin_memory_mode: str = 'auto'):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = output_dir
        self.logger = get_logger(__name__)
        # 运行时可配置项
        self.has_cuda = (self.device == 'cuda' and torch.cuda.is_available())
        self.amp_mode = amp_mode  # 'auto' | 'on' | 'off'
        self.num_workers = max(int(num_workers), 0)
        self.pin_memory_mode = pin_memory_mode  # 'auto' | 'on' | 'off'
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.logger.info(f"模型训练器初始化完成，使用设备: {self.device}")
    
    def train_unet(self, 
                   train_dataset_path: str,
                   val_dataset_path: str = None,
                   epochs: int = 100,
                   batch_size: int = 8,
                   learning_rate: float = 1e-4) -> str:
        """
        训练U-Net模型
        
        Args:
            train_dataset_path: 训练数据集路径
            val_dataset_path: 验证数据集路径
            epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            
        Returns:
            模型保存路径
        """
        self.logger.info("开始训练U-Net模型")
        
        # 创建数据加载器 - 高性能版本
        self.logger.info("创建训练数据加载器...")
        # 禁用预加载模式，避免内存问题
        train_dataset = ImageDataset(
            train_dataset_path,
            preload=False,  # 禁用预加载避免内存问题
            target_size=(256, 256)  # 确保所有图像尺寸一致
        )
        self.logger.info(f"训练数据集大小: {len(train_dataset)}")
        
        # 增加num_workers实现多线程加载，pin_memory加速CPU到GPU的数据传输
        if self.pin_memory_mode == 'on':
            pin_mem = True
        elif self.pin_memory_mode == 'off':
            pin_mem = False
        else:
            pin_mem = self.has_cuda
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=self.num_workers,  # 使用多个工作线程加载数据
            pin_memory=pin_mem,  # 仅在CUDA可用时启用
            prefetch_factor=2,  # 预加载因子
            persistent_workers=(self.num_workers > 0)  # 仅当有worker时启用
        )
        
        val_loader = None
        if val_dataset_path and os.path.exists(val_dataset_path):
            self.logger.info("创建验证数据加载器...")
            val_dataset = ImageDataset(
                val_dataset_path,
                preload=True,  # 预加载数据到内存
                target_size=(256, 256)
            )
            self.logger.info(f"验证数据集大小: {len(val_dataset)}")
            
            val_loader = DataLoader(
                val_dataset, 
                batch_size=batch_size*2,  # 验证时可以使用更大的批量
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=pin_mem,
                persistent_workers=(self.num_workers > 0)  # 仅当有worker时启用
            )
        
        # 创建模型
        model = UNet(n_channels=1, n_classes=1, bilinear=False)
        model = model.to(self.device)
        
        # 损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # 训练历史
        train_losses = []
        val_losses = []
        
        best_val_loss = float('inf')
        best_model_path = os.path.join(self.output_dir, "best_unet.pth")
        
        # 训练循环
        for epoch in range(epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            for batch_idx, batch_data in enumerate(train_pbar):
                # 处理数据批次，兼容有PSF和无PSF的情况
                if len(batch_data) == 3:  # 如果包含PSF
                    blurred, clean, _ = batch_data  # 忽略PSF数据
                else:  # 如果只有模糊图像和清晰图像
                    blurred, clean = batch_data
                blurred = blurred.to(self.device)
                clean = clean.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(blurred)
                loss = criterion(outputs, clean)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_pbar.set_postfix({'loss': loss.item()})
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # 验证阶段
            if val_loader:
                model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
                    for data in val_pbar:
                        try:
                            # 处理数据加载器返回的不同情况
                            if isinstance(data, (list, tuple)) and len(data) == 3:  # 如果返回三个值 (blurred, clean, psf)
                                blurred, clean, _ = data  # 忽略PSF数据
                            elif isinstance(data, (list, tuple)) and len(data) == 2:  # 如果返回两个值 (blurred, clean)
                                blurred, clean = data
                            else:
                                self.logger.warning(f"验证数据加载器返回了意外的数据格式: {type(data)}, {len(data) if isinstance(data, (list, tuple)) else '非列表/元组'}")
                                continue
                                
                            blurred = blurred.to(self.device, non_blocking=True)
                            clean = clean.to(self.device, non_blocking=True)
                            
                            outputs = model(blurred)
                            loss = criterion(outputs, clean)
                            val_loss += loss.item()
                            val_pbar.set_postfix({'loss': loss.item()})
                        except Exception as e:
                            self.logger.error(f"验证过程中出错: {str(e)}")
                            continue
                
                val_loss /= len(val_loader)
                val_losses.append(val_loss)
                
                # 保存最佳模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                    }, best_model_path)
                
                # 使用新的学习率调度器
                scheduler.step(val_loss)
                
                self.logger.info(f"Epoch {epoch+1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
            else:
                # 没有验证数据时，使用训练损失作为调度器的指标
                scheduler.step(train_loss)
                self.logger.info(f"Epoch {epoch+1}: Train Loss = {train_loss:.6f}")
        
        # 保存最终模型
        final_model_path = os.path.join(self.output_dir, "final_unet.pth")
        torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
        }, final_model_path)
        
        # 保存训练历史图表
        self._plot_training_history(train_losses, val_losses, "unet_training_history.png")
        
        self.logger.info(f"U-Net训练完成，最佳模型保存到: {best_model_path}")
        return best_model_path
    
    def train_mra_net(self, 
                      train_dataset_path: str,
                      val_dataset_path: str = None,
                      epochs: int = 100,  # 与U-Net保持一致，增加训练轮数
                      batch_size: int = 8,  # 与U-Net保持一致，提高训练稳定性
                      learning_rate: float = 1e-4,
                      loss_weights: Optional[Dict[str, float]] = None) -> str:
        """
        训练MRA-Net模型 (快速验证版本)
        
        Args:
            train_dataset_path: 训练数据集路径
            val_dataset_path: 验证数据集路径
            epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            
        Returns:
            模型保存路径
        """
        self.logger.info("开始训练MRA-Net模型 (快速验证版)")
        
        # 创建数据加载器 - 高性能版本
        self.logger.info("创建训练数据加载器...")
        # 统一数据加载策略，与U-Net保持一致
        train_dataset = ImageDataset(
            train_dataset_path,
            preload=False,  # 与U-Net保持一致，避免内存问题
            target_size=(256, 256)  # 确保所有图像尺寸一致
        )
        self.logger.info(f"训练数据集大小: {len(train_dataset)}")
        
        # 优化数据加载器配置，提高训练速度
        has_cuda = self.has_cuda
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=self.num_workers,  # 启用多线程加载
            pin_memory=(True if self.pin_memory_mode=='on' else False if self.pin_memory_mode=='off' else has_cuda),
            persistent_workers=(self.num_workers > 0)  # 仅当有worker时启用
        )
        
        val_loader = None
        if val_dataset_path and os.path.exists(val_dataset_path):
            self.logger.info("创建验证数据加载器...")
            val_dataset = ImageDataset(
                val_dataset_path,
                preload=False,  # 与U-Net保持一致
                target_size=(256, 256)  # 确保所有图像尺寸一致
            )
            self.logger.info(f"验证数据集大小: {len(val_dataset)}")
            
            val_loader = DataLoader(
                val_dataset, 
                batch_size=batch_size*2,  # 验证时使用更大的批次
                shuffle=False,
                num_workers=self.num_workers,  # 启用多线程加载
                pin_memory=(True if self.pin_memory_mode=='on' else False if self.pin_memory_mode=='off' else has_cuda),  # 仅在CUDA可用时启用
                persistent_workers=(self.num_workers > 0)  # 仅当有worker时启用
            )
        
        # 创建增强版MRA-Net模型 - 增加模型容量以提升性能
        model = EnhancedMRANet(num_stages=12, hidden_channels=128)  # 增加阶段数和通道数
        model = model.to(self.device)
        
        # 打印模型参数信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"MRA-Net模型参数: 总计 {total_params:,}, 可训练 {trainable_params:,}")
        self.logger.info(f"模型大小: {total_params * 4 / 1024 / 1024:.2f} MB")
        
        # 设置模型为训练模式
        model.train()
        
        # 优化损失函数配置，可由外部权重覆盖
        lw = loss_weights or {}
        lambda_physics = float(lw.get('physics', 0.5))
        lambda_perceptual = float(lw.get('perceptual', 0.2))
        lambda_edge = float(lw.get('edge', 0.3))
        lambda_ssim = float(lw.get('ssim', 0.4))
        criterion = EnhancedMRANetLoss(
            lambda_physics=lambda_physics,
            lambda_perceptual=lambda_perceptual,
            lambda_edge=lambda_edge,
            lambda_ssim=lambda_ssim
        )
        criterion = criterion.to(self.device)
        
        # 使用AdamW优化器，优化权重衰减和学习率
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5, betas=(0.9, 0.999), eps=1e-8)
        
        # 改进的学习率调度策略
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2, eta_min=1e-6  # 增加重启周期，提高收敛稳定性
        )
        
        # 添加梯度裁剪
        max_grad_norm = 1.0
        
        # 启用混合精度训练，提高训练速度
        # AMP 开关按参数覆盖
        if self.amp_mode == 'on':
            use_amp = has_cuda
        elif self.amp_mode == 'off':
            use_amp = False
        else:
            use_amp = has_cuda  # auto
        # 使用新API：torch.amp.GradScaler('cuda', ...)
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp) if has_cuda else torch.amp.GradScaler('cuda', enabled=False)
        autocast_ctx = torch.amp.autocast('cuda') if use_amp else contextlib.nullcontext()
        
        # 训练历史
        train_losses = []
        val_losses = []
        
        best_val_loss = float('inf')
        best_model_path = os.path.join(self.output_dir, "best_mra_net_fast.pth")
        
        # 打印GPU信息
        if torch.cuda.is_available():
            self.logger.info(f"使用GPU: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            
        # 设置CUDA相关优化，仅在GPU可用时启用
        if has_cuda:
            torch.backends.cudnn.benchmark = True  # 启用cuDNN自动调优
            torch.backends.cudnn.fastest = True  # 使用最快的算法
            torch.backends.cudnn.deterministic = False  # 禁用确定性以提高性能
            self.logger.info("已启用cuDNN benchmark和fastest模式以优化性能")
            # 清理GPU缓存，确保有足够的显存
            torch.cuda.empty_cache()
        else:
            self.logger.info("CUDA不可用：已禁用AMP、pin_memory与cuDNN优化，采用CPU训练路径")
        
        # 添加早停机制，避免过度训练
        patience = 10
        no_improve_epochs = 0
        
        # 训练循环
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # 训练阶段
            model.train()
            train_loss = 0.0
            
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            for batch_idx, data in enumerate(train_pbar):
                try:
                    if len(data) == 3:  # 有PSF数据
                        blurred, clean, psf = data
                        psf = psf.to(self.device, non_blocking=True)
                    else:  # 没有PSF数据
                        blurred, clean = data
                        psf = None
                    
                    blurred = blurred.to(self.device, non_blocking=True)
                    clean = clean.to(self.device, non_blocking=True)
                    
                    optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零
                    
                    # 使用混合精度训练（或空上下文）
                    with autocast_ctx:
                        # 确保输入数据类型一致
                        blurred = blurred.float()
                        # MRANet.forward()只接受一个参数
                        outputs, model_psf = model(blurred)
                        
                        # 将模型对象作为第三个参数传递给criterion
                        loss_dict = criterion(outputs, clean, model)
                        # 从损失字典中提取总损失
                        total_loss = loss_dict['total_loss']
                    
                    # 使用梯度缩放器进行反向传播
                    scaler.scale(total_loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    
                    train_loss += total_loss.item()
                    
                    # 简化进度条信息，避免复杂的GPU查询
                    if batch_idx % 10 == 0:  # 每10个批次更新一次
                        train_pbar.set_postfix({
                            'loss': f"{total_loss.item():.4f}"
                        })
                    
                except RuntimeError as e:
                    # 捕获并处理可能的CUDA错误
                    self.logger.error(f"批次 {batch_idx} 发生CUDA错误: {str(e)}")
                    torch.cuda.empty_cache()  # 清理GPU内存
                    continue
                except Exception as e:
                    self.logger.error(f"批次 {batch_idx} 发生未预期错误: {str(e)}")
                    torch.cuda.empty_cache()
                    continue
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # 验证阶段
            if val_loader:
                model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
                    for data in val_pbar:
                        if len(data) == 3:
                            blurred, clean, psf = data
                            psf = psf.to(self.device, non_blocking=True)
                        else:
                            blurred, clean = data
                            psf = None
                        
                        blurred = blurred.to(self.device, non_blocking=True)
                        clean = clean.to(self.device, non_blocking=True)
                        
                        # 使用混合精度进行推理（或空上下文）
                        with autocast_ctx:
                            # MRANet.forward()只接受一个参数
                            outputs, model_psf = model(blurred)
                            # 将模型对象作为第三个参数传递给criterion
                            loss_dict = criterion(outputs, clean, model)
                            # 从损失字典中提取总损失
                            total_loss = loss_dict['total_loss']
                        
                        val_loss += total_loss.item()
                        val_pbar.set_postfix({'loss': total_loss.item()})
                
                val_loss /= len(val_loader)
                val_losses.append(val_loss)
                
                # 保存最佳模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                    }, best_model_path)
                    self.logger.info(f"保存最佳模型，验证损失: {val_loss:.6f}")
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1
                    if no_improve_epochs >= patience:
                        self.logger.info(f"早停：{patience}轮未改善，停止训练")
                        break
                
                scheduler.step(val_loss)
                
                # 计算并记录每个epoch的时间
                epoch_time = time.time() - epoch_start_time
                self.logger.info(
                    f"Epoch {epoch+1}/{epochs}, 训练损失: {train_loss:.6f}, 验证损失: {val_loss:.6f}, "
                    f"用时: {epoch_time:.2f}秒"
                )
            else:
                # 计算并记录每个epoch的时间
                epoch_time = time.time() - epoch_start_time
                self.logger.info(
                    f"Epoch {epoch+1}/{epochs}, 训练损失: {train_loss:.6f}, "
                    f"用时: {epoch_time:.2f}秒"
                )
                
                # 没有验证集时，每5个epoch保存一次模型（减少保存频率）
                if (epoch + 1) % 5 == 0:
                    checkpoint_path = os.path.join(self.output_dir, f"mra_net_epoch_{epoch+1}.pth")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': train_loss,
                    }, checkpoint_path)
        
        # 保存最终模型
        final_model_path = os.path.join(self.output_dir, "final_mra_net.pth")
        torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
        }, final_model_path)
        
        # 保存训练历史图表
        self._plot_training_history(train_losses, val_losses, "mra_net_training_history.png")
        
        self.logger.info(f"MRA-Net训练完成，最佳模型保存到: {best_model_path}")
        return best_model_path
    
    def _plot_training_history(self, train_losses: List[float], val_losses: List[float], filename: str):
        """绘制训练历史图表"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss', color='blue')
        if val_losses:
            plt.plot(val_losses, label='Validation Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)
        
        chart_path = os.path.join(self.output_dir, filename)
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"训练历史图表已保存到: {chart_path}")
    
    def evaluate_model(self, model_path: str, test_dataset_path: str, model_type: str = "unet") -> Dict:
        """评估模型性能"""
        self.logger.info(f"开始评估{model_type}模型")
        
        # 加载模型
        if model_type.lower() == "unet":
            model = UNet(n_channels=1, n_classes=1, bilinear=False)
        else:  # mra_net
            model = EnhancedMRANet(num_stages=12, hidden_channels=128)  # 使用增强配置
        
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        # 加载测试数据
        test_dataset = ImageDataset(test_dataset_path)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        # 评估指标
        total_loss = 0.0
        criterion = nn.MSELoss()
        
        with torch.no_grad():
            for data in tqdm(test_loader, desc="Evaluating"):
                if len(data) == 3 and model_type.lower() == "mra_net":
                    blurred, clean, psf = data
                    psf = psf.to(self.device)
                    outputs = model(blurred.to(self.device), psf)
                else:
                    blurred, clean = data[:2]
                    outputs = model(blurred.to(self.device))
                
                loss = criterion(outputs, clean.to(self.device))
                total_loss += loss.item()
        
        avg_loss = total_loss / len(test_loader)
        
        results = {
            'model_type': model_type,
            'model_path': model_path,
            'test_dataset_path': test_dataset_path,
            'average_loss': avg_loss,
            'num_test_samples': len(test_dataset)
        }
        
        self.logger.info(f"{model_type}模型评估完成，平均损失: {avg_loss:.6f}")
        return results


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='模型训练系统')
    parser.add_argument('--mode', choices=['unet', 'mranet', 'both', 'interactive'], 
                       default='interactive', help='训练模式')
    parser.add_argument('--train_path', default='outputs/datasets/train_dataset.h5',
                       help='训练数据集路径')
    parser.add_argument('--val_path', default='outputs/datasets/val_dataset.h5',
                       help='验证数据集路径')
    parser.add_argument('--epochs_unet', type=int, default=100, help='U-Net训练轮数')
    parser.add_argument('--epochs_mranet', type=int, default=100, help='MRA-Net训练轮数')
    parser.add_argument('--batch_size_unet', type=int, default=8, help='U-Net批次大小')
    parser.add_argument('--batch_size_mranet', type=int, default=8, help='MRA-Net批次大小')
    # 新增运行时控制参数
    parser.add_argument('--amp', dest='amp', action='store_true', help='强制开启AMP混合精度')
    parser.add_argument('--no-amp', dest='no_amp', action='store_true', help='强制关闭AMP混合精度')
    parser.add_argument('--num-workers', dest='num_workers', type=int, default=4, help='DataLoader工作线程数')
    parser.add_argument('--pin-memory', dest='pin_memory', action='store_true', help='强制开启pin_memory')
    parser.add_argument('--no-pin-memory', dest='no_pin_memory', action='store_true', help='强制关闭pin_memory')
    # 新增：损失权重
    parser.add_argument('--lw-physics', type=float, default=None, help='物理损失权重')
    parser.add_argument('--lw-perceptual', type=float, default=None, help='感知损失权重')
    parser.add_argument('--lw-edge', type=float, default=None, help='边缘损失权重')
    parser.add_argument('--lw-ssim', type=float, default=None, help='SSIM损失权重')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'interactive':
            print("\n=== 模型训练系统 ===")
            print("1. 训练U-Net模型")
            print("2. 训练MRA-Net模型")
            print("3. 评估模型性能")
            print("4. 退出")
            
            choice = input("\n请选择操作 (1-4): ").strip()
        elif args.mode == 'unet':
            choice = '1'
        elif args.mode == 'mranet':
            choice = '2'
        elif args.mode == 'both':
            choice = 'both'
        
        if choice == '1':
            print("\n🚀 开始训练U-Net模型...")
            
            if args.mode == 'interactive':
                # 获取用户输入
                train_path = input("训练数据集路径 [outputs/datasets/train_dataset.h5]: ").strip()
                if not train_path:
                    train_path = "outputs/datasets/train_dataset.h5"
                
                val_path = input("验证数据集路径 [outputs/datasets/val_dataset.h5]: ").strip()
                if not val_path:
                    val_path = "outputs/datasets/val_dataset.h5"
                
                epochs = input("训练轮数 [100]: ").strip()
                epochs = int(epochs) if epochs else 100
                
                batch_size = input("批次大小 [8]: ").strip()
                batch_size = int(batch_size) if batch_size else 8
            else:
                # 使用命令行参数
                train_path = args.train_path
                val_path = args.val_path
                epochs = args.epochs_unet
                batch_size = args.batch_size_unet
            
            # 检查数据集
            if not os.path.exists(train_path):
                print(f"❌ 训练数据集不存在: {train_path}")
                return
            
            # 创建训练器并开始训练
            amp_mode = 'on' if args.amp else ('off' if args.no_amp else 'auto')
            pin_memory_mode = 'on' if args.pin_memory else ('off' if args.no_pin_memory else 'auto')
            trainer = ModelTrainer(amp_mode=amp_mode, num_workers=args.num_workers, pin_memory_mode=pin_memory_mode)
            model_path = trainer.train_unet(
                train_dataset_path=train_path,
                val_dataset_path=val_path if os.path.exists(val_path) else None,
                epochs=epochs,
                batch_size=batch_size
            )
            
            print(f"\n✅ U-Net训练完成: {model_path}")
            
        elif choice == '2':
            print("\n🚀 开始训练MRA-Net模型...")
            
            if args.mode == 'interactive':
                # 获取用户输入
                train_path = input("训练数据集路径 [outputs/datasets/train_dataset.h5]: ").strip()
                if not train_path:
                    train_path = "outputs/datasets/train_dataset.h5"
                
                val_path = input("验证数据集路径 [outputs/datasets/val_dataset.h5]: ").strip()
                if not val_path:
                    val_path = "outputs/datasets/val_dataset.h5"
                
                epochs = input("训练轮数 [100]: ").strip()
                epochs = int(epochs) if epochs else 100
                
                batch_size = input("批次大小 [8]: ").strip()
                batch_size = int(batch_size) if batch_size else 8
            else:
                # 使用命令行参数
                train_path = args.train_path
                val_path = args.val_path
                epochs = args.epochs_mranet
                batch_size = args.batch_size_mranet
            
            # 检查数据集
            if not os.path.exists(train_path):
                print(f"❌ 训练数据集不存在: {train_path}")
                return
            
            # 创建训练器并开始训练
            amp_mode = 'on' if args.amp else ('off' if args.no_amp else 'auto')
            pin_memory_mode = 'on' if args.pin_memory else ('off' if args.no_pin_memory else 'auto')
            trainer = ModelTrainer(amp_mode=amp_mode, num_workers=args.num_workers, pin_memory_mode=pin_memory_mode)
            # 组装损失权重（仅当提供时覆盖默认）
            loss_weights = {}
            if args.lw_physics is not None: loss_weights['physics'] = args.lw_physics
            if args.lw_perceptual is not None: loss_weights['perceptual'] = args.lw_perceptual
            if args.lw_edge is not None: loss_weights['edge'] = args.lw_edge
            if args.lw_ssim is not None: loss_weights['ssim'] = args.lw_ssim

            model_path = trainer.train_mra_net(
                train_dataset_path=train_path,
                val_dataset_path=val_path if os.path.exists(val_path) else None,
                epochs=epochs,
                batch_size=batch_size,
                loss_weights=loss_weights if loss_weights else None
            )
            
            print(f"\n✅ MRA-Net训练完成: {model_path}")
            
        elif choice == 'both':
            print("\n🚀 开始训练两个模型...")
            
            # 检查数据集
            if not os.path.exists(args.train_path):
                print(f"❌ 训练数据集不存在: {args.train_path}")
                return
            
            amp_mode = 'on' if args.amp else ('off' if args.no_amp else 'auto')
            pin_memory_mode = 'on' if args.pin_memory else ('off' if args.no_pin_memory else 'auto')
            trainer = ModelTrainer(amp_mode=amp_mode, num_workers=args.num_workers, pin_memory_mode=pin_memory_mode)
            
            # 训练U-Net
            print("\n🔥 第1步: 训练U-Net模型...")
            unet_path = trainer.train_unet(
                train_dataset_path=args.train_path,
                val_dataset_path=args.val_path if os.path.exists(args.val_path) else None,
                epochs=args.epochs_unet,
                batch_size=args.batch_size_unet
            )
            print(f"✅ U-Net训练完成: {unet_path}")
            
            # 训练MRA-Net
            print("\n🔥 第2步: 训练MRA-Net模型...")
            # 组装损失权重（仅当提供时覆盖默认）
            loss_weights = {}
            if args.lw_physics is not None: loss_weights['physics'] = args.lw_physics
            if args.lw_perceptual is not None: loss_weights['perceptual'] = args.lw_perceptual
            if args.lw_edge is not None: loss_weights['edge'] = args.lw_edge
            if args.lw_ssim is not None: loss_weights['ssim'] = args.lw_ssim

            mranet_path = trainer.train_mra_net(
                train_dataset_path=args.train_path,
                val_dataset_path=args.val_path if os.path.exists(args.val_path) else None,
                epochs=args.epochs_mranet,
                batch_size=args.batch_size_mranet,
                loss_weights=loss_weights if loss_weights else None
            )
            print(f"✅ MRA-Net训练完成: {mranet_path}")
            
            print(f"\n🎉 所有模型训练完成!")
            print(f"U-Net模型: {unet_path}")
            print(f"MRA-Net模型: {mranet_path}")
            
        elif choice == '3':
            print("\n📊 模型评估功能开发中...")
            
        elif choice == '4':
            print("\n👋 退出程序")
            
        else:
            print("\n❌ 无效选择，请重新运行程序")
            
    except KeyboardInterrupt:
        print("\n⚠️ 操作被用户中断")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()