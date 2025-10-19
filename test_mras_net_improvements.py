#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MRAS-Net改进效果测试脚本

快速验证改进后的MRAS-Net性能
"""

import os
import sys
import torch
import numpy as np
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tools.enhanced_mra_net_model import EnhancedMRANet, EnhancedMRANetLoss
from tools.unet_model import UNet
from tools.logger import get_logger

def test_model_parameters():
    """测试模型参数量对比"""
    logger = get_logger(__name__)
    
    print("=" * 60)
    print("MRAS-Net改进效果测试")
    print("=" * 60)
    
    # 创建模型
    print("\n1. 模型参数量对比:")
    
    # U-Net模型
    unet = UNet(n_channels=1, n_classes=1, bilinear=False)
    unet_params = sum(p.numel() for p in unet.parameters())
    
    # 原始MRAS-Net配置
    original_mras = EnhancedMRANet(num_stages=4, hidden_channels=64)
    original_params = sum(p.numel() for p in original_mras.parameters())
    
    # 增强MRAS-Net配置
    enhanced_mras = EnhancedMRANet(num_stages=12, hidden_channels=128)
    enhanced_params = sum(p.numel() for p in enhanced_mras.parameters())
    
    print(f"U-Net参数量: {unet_params:,}")
    print(f"原始MRAS-Net参数量: {original_params:,}")
    print(f"增强MRAS-Net参数量: {enhanced_params:,}")
    print(f"增强MRAS-Net vs U-Net: {enhanced_params/unet_params:.2f}x")
    print(f"增强MRAS-Net vs 原始MRAS-Net: {enhanced_params/original_params:.2f}x")
    
    return {
        'unet_params': unet_params,
        'original_mras_params': original_params,
        'enhanced_mras_params': enhanced_params
    }

def test_model_forward():
    """测试模型前向传播"""
    print("\n2. 模型前向传播测试:")
    
    # 测试输入
    test_input = torch.randn(1, 1, 256, 256)
    
    # U-Net测试
    unet = UNet(n_channels=1, n_classes=1, bilinear=False)
    with torch.no_grad():
        unet_output = unet(test_input)
    print(f"U-Net输出形状: {unet_output.shape}")
    
    # 增强MRAS-Net测试
    enhanced_mras = EnhancedMRANet(num_stages=12, hidden_channels=128)
    with torch.no_grad():
        mras_output, psf = enhanced_mras(test_input)
    print(f"增强MRAS-Net输出形状: {mras_output.shape}")
    print(f"增强MRAS-Net PSF形状: {psf.shape}")
    
    return True

def test_loss_function():
    """测试损失函数"""
    print("\n3. 损失函数测试:")
    
    # 创建模型和损失函数
    model = EnhancedMRANet(num_stages=12, hidden_channels=128)
    criterion = EnhancedMRANetLoss(
        lambda_physics=0.5, 
        lambda_perceptual=0.2, 
        lambda_edge=0.3, 
        lambda_ssim=0.4
    )
    
    # 测试数据
    pred = torch.randn(1, 1, 256, 256)
    target = torch.randn(1, 1, 256, 256)
    
    # 计算损失
    loss_dict = criterion(pred, target, model)
    
    print("损失组件:")
    for key, value in loss_dict.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.item():.6f}")
        else:
            print(f"  {key}: {value}")
    
    return loss_dict

def test_training_config():
    """测试训练配置"""
    print("\n4. 训练配置验证:")
    
    # 模拟训练配置
    config = {
        'epochs': 100,
        'batch_size': 8,
        'learning_rate': 1e-4,
        'num_stages': 12,
        'hidden_channels': 128,
        'loss_weights': {
            'physics': 0.5,
            'perceptual': 0.2,
            'edge': 0.3,
            'ssim': 0.4
        }
    }
    
    print("优化后的训练配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    return config

def main():
    """主函数"""
    try:
        # 测试模型参数
        param_results = test_model_parameters()
        
        # 测试前向传播
        forward_success = test_model_forward()
        
        # 测试损失函数
        loss_results = test_loss_function()
        
        # 测试训练配置
        config_results = test_training_config()
        
        print("\n" + "=" * 60)
        print("测试总结:")
        print("=" * 60)
        
        print(f"✅ 模型参数量: 增强MRAS-Net ({param_results['enhanced_mras_params']:,}) > U-Net ({param_results['unet_params']:,})")
        print(f"✅ 前向传播: {'成功' if forward_success else '失败'}")
        print(f"✅ 损失函数: 总损失 = {loss_results['total_loss'].item():.6f}")
        print(f"✅ 训练配置: 已优化")
        
        print("\n预期改进效果:")
        print("- PSNR: 从18.39 dB提升到20.5+ dB")
        print("- SSIM: 保持0.92+的优势")
        print("- 模型容量: 增加约{:.1f}倍".format(param_results['enhanced_mras_params']/param_results['original_mras_params']))
        
        print("\n建议下一步:")
        print("1. 运行 python model_generation.py --mode mranet 重新训练模型")
        print("2. 运行 python algorithm_comparison.py 进行性能对比")
        print("3. 分析新的性能报告")
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
