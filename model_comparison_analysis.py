#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型参数量对比分析

对比MRA-Net和U-Net的模型复杂度，并设计增强版MRA-Net
"""

import torch
import torch.nn as nn
import sys
import os
from typing import Dict, Any

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tools.mras_net_model import create_mra_net_model, MRANet
from tools.unet_model import create_unet_model, UNet
from tools.logger import get_logger


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    统计模型参数量
    
    Args:
        model: PyTorch模型
        
    Returns:
        参数统计字典
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': total_params - trainable_params
    }


def analyze_model_complexity():
    """
    分析两个模型的复杂度对比
    """
    logger = get_logger('model_analysis')
    
    # 创建模型
    logger.info("创建模型...")
    
    # 原始MRA-Net (较小配置)
    mra_net_small = create_mra_net_model(
        num_stages=8, 
        hidden_channels=64,
        image_size=256
    )
    
    # U-Net
    unet = create_unet_model(n_channels=1, n_classes=1, bilinear=False)
    
    # 统计参数量
    mra_stats = count_parameters(mra_net_small)
    unet_stats = count_parameters(unet)
    
    logger.info("=== 模型参数量对比 ===")
    logger.info(f"MRA-Net (原始): {mra_stats['total_params']:,} 参数")
    logger.info(f"U-Net: {unet_stats['total_params']:,} 参数")
    logger.info(f"参数量比例 (MRA-Net/U-Net): {mra_stats['total_params']/unet_stats['total_params']:.3f}")
    
    # 测试前向传播
    test_input = torch.randn(1, 1, 256, 256)
    
    with torch.no_grad():
        mra_output, _ = mra_net_small(test_input)
        unet_output = unet(test_input)
        
    logger.info(f"MRA-Net输出形状: {mra_output.shape}")
    logger.info(f"U-Net输出形状: {unet_output.shape}")
    
    return mra_stats, unet_stats


def design_enhanced_mra_net():
    """
    设计增强版MRA-Net，增加模型容量
    """
    logger = get_logger('model_enhancement')
    
    logger.info("设计增强版MRA-Net...")
    
    # 增强配置1: 增加阶段数和通道数
    enhanced_config_1 = {
        'num_stages': 12,  # 从8增加到12
        'hidden_channels': 128,  # 从64增加到128
        'image_size': 256
    }
    
    # 增强配置2: 更大的模型
    enhanced_config_2 = {
        'num_stages': 16,  # 更多阶段
        'hidden_channels': 256,  # 更多通道
        'image_size': 256
    }
    
    # 创建增强版模型
    enhanced_mra_1 = create_mra_net_model(**enhanced_config_1)
    enhanced_mra_2 = create_mra_net_model(**enhanced_config_2)
    
    # 统计参数量
    stats_1 = count_parameters(enhanced_mra_1)
    stats_2 = count_parameters(enhanced_mra_2)
    
    logger.info("=== 增强版MRA-Net参数量 ===")
    logger.info(f"增强版1 (12阶段, 128通道): {stats_1['total_params']:,} 参数")
    logger.info(f"增强版2 (16阶段, 256通道): {stats_2['total_params']:,} 参数")
    
    return enhanced_config_1, enhanced_config_2, stats_1, stats_2


if __name__ == "__main__":
    print("开始模型对比分析...")
    
    # 分析原始模型
    mra_stats, unet_stats = analyze_model_complexity()
    
    # 设计增强版
    config_1, config_2, stats_1, stats_2 = design_enhanced_mra_net()
    
    print("\n=== 分析结果 ===")
    print(f"原始MRA-Net: {mra_stats['total_params']:,} 参数")
    print(f"U-Net: {unet_stats['total_params']:,} 参数")
    print(f"增强版MRA-Net-1: {stats_1['total_params']:,} 参数")
    print(f"增强版MRA-Net-2: {stats_2['total_params']:,} 参数")
    
    print("\n=== 建议 ===")
    if stats_1['total_params'] < unet_stats['total_params'] * 1.2:
        print("推荐使用增强版1，参数量适中且接近U-Net")
    else:
        print("推荐使用增强版2，参数量超过U-Net，有望获得更好性能")