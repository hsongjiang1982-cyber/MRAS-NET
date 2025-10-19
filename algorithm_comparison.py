#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
四算法比较脚本

功能：
1. 对比维纳滤波、Richardson-Lucy、U-Net和MRAS-Net的性能
2. 生成定量评估指标
3. 创建可视化图表和报告

使用方法：
    python algorithm_comparison.py
"""

import os
import sys
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import h5py
import json
import time
import psutil
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage
from scipy.signal import wiener
from skimage import restoration
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
from tqdm import tqdm
import yaml
from pathlib import Path

# 添加tools目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'tools'))

from unet_model import UNet
from enhanced_mra_net_model import EnhancedMRANet
from logger import get_logger
from report_generator import PerformanceReportGenerator

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class TestDataset(Dataset):
    """测试数据集类"""
    
    def __init__(self, dataset_path: str):
        with h5py.File(dataset_path, 'r') as f:
            self.clean_images = f['clean_images'][:]
            self.blurred_images = f['blurred_images'][:]
            if 'psfs' in f:
                try:
                    self.psfs = f['psfs'][:]
                except (TypeError, KeyError):
                    # 如果PSF数据格式不正确或不存在，设为None
                    self.psfs = None
            else:
                self.psfs = None
    
    def __len__(self):
        return len(self.clean_images)
    
    def __getitem__(self, idx):
        clean = self.clean_images[idx].astype(np.float32)
        blurred = self.blurred_images[idx].astype(np.float32)
        
        if self.psfs is not None:
            psf = self.psfs[idx].astype(np.float32)
            return clean, blurred, psf
        else:
            return clean, blurred

class AlgorithmComparison:
    """算法比较主类"""
    
    def _load_config(self) -> Dict:
        """加载统一配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            self.logger.warning(f"配置文件 {self.config_path} 不存在，使用默认配置")
            return {
                'random_seed': 42,
                'comparison': {
                    'min_samples': 10,
                    'algorithms': {
                        'wiener': {'enabled': True},
                        'richardson_lucy': {'enabled': True, 'iterations': 30},
                        'unet': {'enabled': True},
                        'mra_net': {'enabled': True}
                    }
                }
            }
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {e}")
            raise
    
    def __init__(self, output_dir: str = None, config_path: str = "config/unified_experiment_config.yaml"):
        """
        初始化算法比较器
        
        Args:
            output_dir: 输出目录，默认为outputs/comparison_YYYYMMDD_HHMMSS
            config_path: 统一配置文件路径
        """
        # 加载统一配置
        self.config_path = config_path
        self.config = self._load_config()
        
        # 如果没有指定输出目录，则使用带时间戳的默认目录
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"outputs/comparison_{timestamp}"
        self.output_dir = output_dir
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger = get_logger(__name__)
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "charts"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "results"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "reports"), exist_ok=True)
        
        # 设置随机种子确保可重复性
        if 'random_seed' in self.config:
            torch.manual_seed(self.config['random_seed'])
            np.random.seed(self.config['random_seed'])
        
        self.logger.info(f"算法比较工具初始化完成，使用设备: {self.device}")
        # 缓存模型避免重复加载
        self._mranet_model = None
        self._mranet_loaded_path = None
        # 评测增强默认参数（可由CLI覆盖）
        self.tta_tile_size = 256
        self.tta_overlap = 32
        self.tta_scales = [1.0]
    
    def wiener_filter(self, blurred_image: np.ndarray, psf: np.ndarray = None, noise_var: float = None) -> np.ndarray:
        """维纳滤波算法"""
        if noise_var is None:
            noise_var = self._estimate_noise_variance(blurred_image)
        
        # 使用scipy的维纳滤波
        if psf is not None:
            # 如果有PSF，使用反卷积
            restored = restoration.wiener(blurred_image, psf, balance=noise_var)
        else:
            # 否则使用简单的维纳滤波
            restored = wiener(blurred_image, noise=noise_var)
        
        return np.clip(restored, 0, 1)
    
    def richardson_lucy(self, blurred_image: np.ndarray, psf: np.ndarray = None, iterations: int = 30) -> np.ndarray:
        """Richardson-Lucy算法"""
        if psf is None:
            # 如果没有PSF，创建一个简单的模糊核
            psf = np.ones((5, 5)) / 25
        
        # 使用skimage的Richardson-Lucy算法
        restored = restoration.richardson_lucy(blurred_image, psf, num_iter=iterations)
        return np.clip(restored, 0, 1)
    
    def _estimate_noise_variance(self, image: np.ndarray) -> float:
        """估计图像噪声方差"""
        # 使用Laplacian算子估计噪声
        laplacian = cv2.Laplacian(image.astype(np.float32), cv2.CV_32F)
        noise_var = np.var(laplacian) * 0.5
        return max(noise_var, 1e-6)
    
    def _test_unet(self, test_loader: DataLoader, model_path: str) -> List[np.ndarray]:
        """测试U-Net模型"""
        # 加载模型
        model = UNet(n_channels=1, n_classes=1, bilinear=False)
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        results = []
        with torch.no_grad():
            for data in test_loader:
                if len(data) == 3:
                    clean, blurred, psf = data
                else:
                    clean, blurred = data
                
                # 检查数据类型并转换为tensor
                if isinstance(blurred, torch.Tensor):
                    blurred_tensor = blurred.float().unsqueeze(0).to(self.device)
                    if blurred_tensor.dim() == 3:  # [H, W] -> [1, 1, H, W]
                        blurred_tensor = blurred_tensor.unsqueeze(0)
                else:
                    blurred_tensor = torch.from_numpy(blurred).float().unsqueeze(0).unsqueeze(0).to(self.device)
                
                # 预测（评测阶段关闭AMP，减少量化误差）
                with torch.no_grad():
                    outputs = model(blurred_tensor)
                
                # 转换回numpy
                restored = outputs.squeeze().cpu().numpy()
                if restored.ndim == 3:  # batch dimension
                    restored = restored[0]
                
                results.append(np.clip(restored, 0, 1))
        
        return results
    
    def _test_mra_net(self, test_loader: DataLoader, model_path: str) -> List[np.ndarray]:
        """测试MRA-Net模型（TTA+滑窗重叠推理）"""
        # 从统一配置加载模型参数，使用增强配置
        model_config = self.config.get('model', {}).get('mra_net', {})
        num_stages = model_config.get('num_stages', 12)  # 使用增强配置
        hidden_channels = model_config.get('hidden_channels', 128)  # 使用增强配置
        
        # 仅当未加载或路径变更时加载一次
        if self._mranet_model is None or self._mranet_loaded_path != model_path:
            print(f"加载MRAS-Net模型: stages={num_stages}, channels={hidden_channels}")
            model = EnhancedMRANet(num_stages=num_stages, hidden_channels=hidden_channels)
            checkpoint = torch.load(model_path, map_location=self.device)
            try:
                to_load = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
                model.load_state_dict(to_load, strict=False)
                print("成功加载MRAS-Net模型参数（使用strict=False）")
            except Exception as e:
                print(f"模型加载失败: {e}")
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    print("尝试使用checkpoint中的state_dict加载")
                    model.load_state_dict(checkpoint['state_dict'], strict=False)
                else:
                    raise RuntimeError("无法加载模型参数，请检查模型结构或权重文件")
            self._mranet_model = model.to(self.device).eval()
            self._mranet_loaded_path = model_path
        model = self._mranet_model
        
        results = []
        with torch.no_grad():
            for data in test_loader:
                if len(data) == 3:
                    clean, blurred, psf = data
                else:
                    clean, blurred = data
                
                # 检查数据类型并转换为tensor
                if isinstance(blurred, torch.Tensor):
                    blurred_tensor = blurred.float().unsqueeze(0).to(self.device)
                    if blurred_tensor.dim() == 3:  # [H, W] -> [1, 1, H, W]
                        blurred_tensor = blurred_tensor.unsqueeze(0)
                else:
                    blurred_tensor = torch.from_numpy(blurred).float().unsqueeze(0).unsqueeze(0).to(self.device)
                
                # 若输入为0-255，归一化到[0,1]
                if blurred_tensor.max() > 1.5:
                    blurred_tensor = blurred_tensor / 255.0
                
                # 使用TTA+滑窗增强预测
                restored = self._tta_predict(model, blurred_tensor)
                
                results.append(np.clip(restored, 0, 1))
        
        return results
    
    def _tta_predict(self, model, input_tensor: torch.Tensor) -> np.ndarray:
        """TTA测试时增强预测（多尺度 + 滑窗重叠）"""
        # 定义8种变换：原图 + 3种90度旋转 + 4种翻转
        transforms = [
            lambda x: x,  # 原图
            lambda x: torch.rot90(x, 1, dims=[2, 3]),  # 90度
            lambda x: torch.rot90(x, 2, dims=[2, 3]),  # 180度
            lambda x: torch.rot90(x, 3, dims=[2, 3]),  # 270度
            lambda x: torch.flip(x, dims=[2]),  # 水平翻转
            lambda x: torch.flip(x, dims=[3]),  # 垂直翻转
            lambda x: torch.flip(torch.rot90(x, 1, dims=[2, 3]), dims=[2]),  # 90度+水平翻转
            lambda x: torch.flip(torch.rot90(x, 1, dims=[2, 3]), dims=[3]),  # 90度+垂直翻转
        ]
        
        # 对应的逆变换
        inverse_transforms = [
            lambda x: x,  # 原图
            lambda x: torch.rot90(x, 3, dims=[2, 3]),  # 90度的逆
            lambda x: torch.rot90(x, 2, dims=[2, 3]),  # 180度的逆
            lambda x: torch.rot90(x, 1, dims=[2, 3]),  # 270度的逆
            lambda x: torch.flip(x, dims=[2]),  # 水平翻转的逆
            lambda x: torch.flip(x, dims=[3]),  # 垂直翻转的逆
            lambda x: torch.rot90(torch.flip(x, dims=[2]), 3, dims=[2, 3]),  # 90度+水平翻转的逆
            lambda x: torch.rot90(torch.flip(x, dims=[3]), 3, dims=[2, 3]),  # 90度+垂直翻转的逆
        ]
        
        predictions = []
        
        # 对每种变换进行预测（评测阶段关闭AMP）
        for scale in (self.tta_scales if hasattr(self, 'tta_scales') and self.tta_scales else [1.0]):
            for transform, inverse_transform in zip(transforms, inverse_transforms):
                # 应用变换
                transformed_input = transform(input_tensor)
                # 按比例缩放
                if abs(scale - 1.0) > 1e-6:
                    h, w = transformed_input.shape[2], transformed_input.shape[3]
                    new_h = max(64, int(h * scale))
                    new_w = max(64, int(w * scale))
                    transformed_input = torch.nn.functional.interpolate(
                        transformed_input, size=(new_h, new_w), mode='bilinear', align_corners=False
                    )
                # 预测（滑窗重叠推理）
                tile = getattr(self, 'tta_tile_size', 256)
                ovl = getattr(self, 'tta_overlap', 32)
                outputs = self._sliding_window_predict(model, transformed_input, tile_size=int(tile), overlap=int(ovl))
                # 缩回原尺寸
                if abs(scale - 1.0) > 1e-6:
                    outputs = torch.nn.functional.interpolate(
                        outputs, size=(input_tensor.shape[2], input_tensor.shape[3]), mode='bilinear', align_corners=False
                    )
                # 应用逆变换
                restored = inverse_transform(outputs)
                predictions.append(restored)
        
        # 平均所有预测结果
        final_prediction = torch.mean(torch.stack(predictions), dim=0)
        
        # 转换回numpy
        restored_np = final_prediction.squeeze().cpu().numpy()
        if restored_np.ndim == 3:  # batch dimension
            restored_np = restored_np[0]
        
        return restored_np

    def _sliding_window_predict(self, model, input_tensor: torch.Tensor, tile_size: int = 256, overlap: int = 32) -> torch.Tensor:
        """滑窗重叠推理，减少边界伪影

        参数:
            model: 已处于eval()的模型
            input_tensor: [B=1, C=1, H, W]
            tile_size: 滑窗尺寸
            overlap: 重叠像素
        返回:
            输出张量，尺寸与输入一致
        """
        assert input_tensor.dim() == 4 and input_tensor.size(0) == 1, "滑窗推理仅支持batch=1"

        _, _, H, W = input_tensor.shape
        # 限制tile不超过图像尺寸，限制overlap小于tile
        effective_tile = max(32, min(tile_size, int(H), int(W)))
        effective_overlap = int(max(0, min(overlap, effective_tile - 1)))
        step = max(1, effective_tile - effective_overlap)
        # 累计输出与权重，用于重叠区域加权平均
        output_acc = torch.zeros((1, 1, H, W), device=input_tensor.device, dtype=input_tensor.dtype)
        weight_acc = torch.zeros((1, 1, H, W), device=input_tensor.device, dtype=input_tensor.dtype)

        # 生成平滑权重窗口（二维余弦窗），降低边界拼接痕迹
        def _cosine_window(sz: int) -> torch.Tensor:
            x = torch.hann_window(sz, device=input_tensor.device, dtype=input_tensor.dtype)
            w2d = torch.ger(x, x)
            return w2d

        window = _cosine_window(effective_tile)
        window = window / (window.max() + 1e-8)
        window = window.unsqueeze(0).unsqueeze(0)  # [1,1,t,t]

        with torch.no_grad():
            for y in range(0, H, step):
                for x in range(0, W, step):
                    y0 = y
                    x0 = x
                    y1 = min(y0 + effective_tile, H)
                    x1 = min(x0 + effective_tile, W)

                    # 调整起点以保证patch大小为tile_size（靠右/下时回退）
                    y0 = max(0, y1 - effective_tile)
                    x0 = max(0, x1 - effective_tile)

                    patch = input_tensor[:, :, y0:y1, x0:x1]
                    # 若边缘导致patch小于tile_size，进行padding到tile_size
                    pad_h = effective_tile - patch.shape[2]
                    pad_w = effective_tile - patch.shape[3]
                    if pad_h > 0 or pad_w > 0:
                        # 使用replicate避免reflect对大padding的限制
                        patch = torch.nn.functional.pad(patch, (0, pad_w, 0, pad_h), mode='replicate')

                    # 模型推理（评测阶段关闭AMP）
                    out_patch, _ = model(patch)

                    # 裁剪回原始区域大小
                    out_patch = out_patch[:, :, : (y1 - y0), : (x1 - x0)]
                    win = window[:, :, : (y1 - y0), : (x1 - x0)]

                    output_acc[:, :, y0:y1, x0:x1] += out_patch * win
                    weight_acc[:, :, y0:y1, x0:x1] += win

        output = output_acc / (weight_acc + 1e-8)
        return output
    
    def calculate_metrics(self, original: np.ndarray, restored: np.ndarray) -> Dict[str, float]:
        """计算评估指标"""
        # 确保图像在[0,1]范围内
        original = np.clip(original, 0, 1)
        restored = np.clip(restored, 0, 1)
        
        # PSNR
        psnr = peak_signal_noise_ratio(original, restored, data_range=1.0)
        
        # SSIM
        ssim = structural_similarity(original, restored, data_range=1.0)
        
        # MSE
        mse = mean_squared_error(original, restored)
        
        # MAE
        mae = np.mean(np.abs(original - restored))
        
        # 边缘保持指数
        epi = self.calculate_edge_preservation(original, restored)
        
        return {
            'psnr': float(psnr),
            'ssim': float(ssim),
            'mse': float(mse),
            'mae': float(mae),
            'edge_preservation': float(epi)
        }
    
    def calculate_edge_preservation(self, original: np.ndarray, restored: np.ndarray) -> float:
        """计算边缘保持指数"""
        try:
            # 计算梯度
            grad_orig_x = cv2.Sobel(original.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
            grad_orig_y = cv2.Sobel(original.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
            grad_rest_x = cv2.Sobel(restored.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
            grad_rest_y = cv2.Sobel(restored.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
            
            # 计算梯度幅值
            grad_orig = np.sqrt(grad_orig_x**2 + grad_orig_y**2)
            grad_rest = np.sqrt(grad_rest_x**2 + grad_rest_y**2)
            
            # 裁剪到相同尺寸
            min_h = min(grad_orig.shape[0], grad_rest.shape[0])
            min_w = min(grad_orig.shape[1], grad_rest.shape[1])
            grad_orig = grad_orig[:min_h, :min_w]
            grad_rest = grad_rest[:min_h, :min_w]
            
            # 计算边缘保持指数
            numerator = np.sum(grad_orig * grad_rest)
            denominator = np.sum(grad_orig**2)
            
            if denominator > 0:
                epi = numerator / denominator
            else:
                epi = 0.0
            
            return max(0.0, min(1.0, epi))
        
        except Exception as e:
            self.logger.warning(f"边缘保持指数计算失败: {e}")
            return 0.0
    
    def measure_performance(self, func, *args, **kwargs) -> Tuple[Any, Dict[str, float]]:
        """测量算法性能"""
        # 记录开始时间和内存
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # 执行函数
        result = func(*args, **kwargs)
        
        # 记录结束时间和内存
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        performance = {
            'execution_time': end_time - start_time,
            'memory_usage': end_memory - start_memory,
            'peak_memory': end_memory
        }
        
        return result, performance
    
    def run_comparison(self, 
                      test_dataset_path: str,
                      unet_model_path: str = None,
                      mra_net_model_path: str = None,
                      num_samples: int = None) -> Dict[str, Any]:
        """运行算法比较"""
        self.logger.info("开始运行算法比较")
        
        # 加载测试数据集
        test_dataset = TestDataset(test_dataset_path)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        # 确保有足够的测试样本
        min_samples = self.config.get('comparison', {}).get('min_samples', 10)
        if num_samples:
            # 限制测试样本数量，但确保不少于最小样本数
            actual_samples = max(min(num_samples, len(test_dataset)), min_samples)
            test_data = [(test_dataset[i]) for i in range(min(actual_samples, len(test_dataset)))]
        else:
            # 使用全部样本，但至少要有最小样本数
            if len(test_dataset) < min_samples:
                self.logger.warning(f"测试数据集样本数({len(test_dataset)})少于建议的最小样本数({min_samples})")
            test_data = [test_dataset[i] for i in range(len(test_dataset))]
        
        # 存储所有结果
        all_results = []
        
        # 根据配置确定要测试的算法
        algorithms = []
        alg_list = self.config.get('comparison', {}).get('algorithms', ['Wiener', 'Richardson-Lucy', 'Unet', 'MRA-Net'])
        
        # 检查每个算法是否可用
        if 'Wiener' in alg_list:
            algorithms.append('Wiener')
        if 'Richardson-Lucy' in alg_list:
            algorithms.append('Richardson-Lucy')
        if 'Unet' in alg_list and unet_model_path and os.path.exists(unet_model_path):
            algorithms.append('Unet')
        if 'MRA-Net' in alg_list and mra_net_model_path and os.path.exists(mra_net_model_path):
            algorithms.append('MRAS-Net')  # 使用MRAS-Net作为名称，与论文一致
        
        print(f"将测试以下算法: {algorithms}")
        
        # 对每个测试样本运行所有算法
        for i, data in enumerate(tqdm(test_data, desc="Processing samples")):
            if len(data) == 3:
                clean, blurred, psf = data
            else:
                clean, blurred = data
                psf = None
            
            sample_results = {
                'sample_id': i,
                'algorithms': {}
            }
            
            # 1. 维纳滤波
            restored, perf = self.measure_performance(
                self.wiener_filter, blurred, psf
            )
            metrics = self.calculate_metrics(clean, restored)
            sample_results['algorithms']['Wiener'] = {
                'metrics': metrics,
                'performance': perf
            }
            
            # 2. Richardson-Lucy
            rl_iterations = self.config.get('comparison', {}).get('richardson_lucy_iterations', 30)
            restored, perf = self.measure_performance(
                self.richardson_lucy, blurred, psf, rl_iterations
            )
            metrics = self.calculate_metrics(clean, restored)
            sample_results['algorithms']['Richardson-Lucy'] = {
                'metrics': metrics,
                'performance': perf
            }
            
            # 3. U-Net (如果模型存在)
            if 'Unet' in algorithms:
                try:
                    single_loader = DataLoader([data], batch_size=1, shuffle=False)
                    restored_list, perf = self.measure_performance(
                        self._test_unet, single_loader, unet_model_path
                    )
                    restored = restored_list[0]
                    metrics = self.calculate_metrics(clean, restored)
                    sample_results['algorithms']['Unet'] = {
                        'metrics': metrics,
                        'performance': perf
                    }
                except Exception as e:
                    self.logger.warning(f"U-Net测试失败: {e}")
            
            # 4. MRAS-Net (如果模型存在)
            if 'MRAS-Net' in algorithms:
                try:
                    single_loader = DataLoader([data], batch_size=1, shuffle=False)
                    restored_list, perf = self.measure_performance(
                        self._test_mra_net, single_loader, mra_net_model_path
                    )
                    restored = restored_list[0]
                    metrics = self.calculate_metrics(clean, restored)
                    sample_results['algorithms']['MRAS-Net'] = {
                        'metrics': metrics,
                        'performance': perf
                    }
                except Exception as e:
                    print(f"警告: MRAS-Net测试失败: {e}")
            
            all_results.append(sample_results)
        
        # 分析结果
        comparison_stats = self._analyze_results(all_results)
        
        # 保存结果
        self._save_results(all_results, comparison_stats)
        
        # 生成图表
        self._generate_charts(comparison_stats)
        
        # 生成报告
        self._generate_report(comparison_stats)
        
        self.logger.info("算法比较完成")
        return comparison_stats
    
    def _analyze_results(self, all_results: List[Dict]) -> Dict[str, Any]:
        """分析比较结果"""
        algorithms = set()
        for result in all_results:
            algorithms.update(result['algorithms'].keys())
        algorithms = list(algorithms)
        
        # 计算统计数据
        stats = {
            'algorithms': algorithms,
            'num_samples': len(all_results),
            'metrics': {},
            'performance': {},
            'summary': {}
        }
        
        # 对每个算法计算统计指标
        for alg in algorithms:
            # 收集指标
            metrics_data = {
                'psnr': [],
                'ssim': [],
                'mse': [],
                'mae': [],
                'edge_preservation': []
            }
            
            perf_data = {
                'execution_time': [],
                'memory_usage': [],
                'peak_memory': []
            }
            
            for result in all_results:
                if alg in result['algorithms']:
                    alg_result = result['algorithms'][alg]
                    
                    # 收集指标数据
                    for metric in metrics_data.keys():
                        if metric in alg_result['metrics']:
                            metrics_data[metric].append(alg_result['metrics'][metric])
                    
                    # 收集性能数据
                    for perf in perf_data.keys():
                        if perf in alg_result['performance']:
                            perf_data[perf].append(alg_result['performance'][perf])
            
            # 计算统计值
            stats['metrics'][alg] = {}
            for metric, values in metrics_data.items():
                if values:
                    # 确保有足够样本计算有意义的统计量
                    min_samples = self.config.get('comparison', {}).get('min_samples', 10)
                    if len(values) >= min_samples:
                        std_val = float(np.std(values, ddof=1))  # 使用样本标准差
                    elif len(values) >= 2:
                        std_val = float(np.std(values, ddof=1))  # 使用样本标准差但标记为不可靠
                    else:
                        std_val = 0.0  # 单样本时标准差为0
                    
                    stats['metrics'][alg][metric] = {
                        'mean': float(np.mean(values)),
                        'std': std_val,
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'median': float(np.median(values)),
                        'count': len(values)  # 添加样本数量信息
                    }

            stats['performance'][alg] = {}
            for perf, values in perf_data.items():
                if values:
                    # 确保有足够样本计算有意义的统计量
                    min_samples = self.config.get('comparison', {}).get('min_samples', 10)
                    if len(values) >= min_samples:
                        std_val = float(np.std(values, ddof=1))  # 使用样本标准差
                    elif len(values) >= 2:
                        std_val = float(np.std(values, ddof=1))  # 使用样本标准差但标记为不可靠
                    else:
                        std_val = 0.0  # 单样本时标准差为0
                    
                    stats['performance'][alg][perf] = {
                        'mean': float(np.mean(values)),
                        'std': std_val,
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'median': float(np.median(values)),
                        'count': len(values)  # 添加样本数量信息
                    }
        
        # 生成排名
        stats['rankings'] = self._calculate_rankings(stats['metrics'])
        
        return stats
    
    def _calculate_rankings(self, metrics: Dict) -> Dict:
        """计算算法排名"""
        rankings = {}
        
        for metric in ['psnr', 'ssim', 'edge_preservation']:
            # 这些指标越高越好
            metric_values = []
            for alg in metrics.keys():
                if metric in metrics[alg]:
                    metric_values.append((alg, metrics[alg][metric]['mean']))
            
            metric_values.sort(key=lambda x: x[1], reverse=True)
            rankings[metric] = [alg for alg, _ in metric_values]
        
        for metric in ['mse', 'mae']:
            # 这些指标越低越好
            metric_values = []
            for alg in metrics.keys():
                if metric in metrics[alg]:
                    metric_values.append((alg, metrics[alg][metric]['mean']))
            
            metric_values.sort(key=lambda x: x[1])
            rankings[metric] = [alg for alg, _ in metric_values]
        
        return rankings
    
    def _save_results(self, all_results: List[Dict], comparison_stats: Dict):
        """保存结果"""
        # 保存详细结果（原始格式）
        detailed_results_file = os.path.join(self.output_dir, "results", "detailed_comparison_results.json")
        os.makedirs(os.path.dirname(detailed_results_file), exist_ok=True)
        
        with open(detailed_results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'detailed_results': all_results,
                'statistics': comparison_stats,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2, ensure_ascii=False)
        
        # 保存转换后的结果（用于兼容性）
        converted_results = {}
        for result in all_results:
            sample_id = result['sample_id']
            for alg_name, alg_data in result['algorithms'].items():
                alg_key = alg_name.lower().replace('-', '_')
                if alg_key not in converted_results:
                    converted_results[alg_key] = {'metrics': [], 'times': []}
                
                converted_results[alg_key]['metrics'].append(alg_data['metrics'])
                converted_results[alg_key]['times'].append(alg_data['performance'].get('execution_time', 0))
        
        results_file = os.path.join(self.output_dir, "results", "comparison_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(converted_results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"详细结果已保存到: {detailed_results_file}")
        self.logger.info(f"转换后结果已保存到: {results_file}")
    
    def _generate_charts(self, stats: Dict):
        """生成可视化图表"""
        algorithms = stats['algorithms']
        
        # 1. PSNR比较
        plt.figure(figsize=(10, 6))
        psnr_means = [stats['metrics'][alg]['psnr']['mean'] for alg in algorithms if 'psnr' in stats['metrics'][alg]]
        psnr_stds = [stats['metrics'][alg]['psnr']['std'] for alg in algorithms if 'psnr' in stats['metrics'][alg]]
        psnr_algs = [alg for alg in algorithms if 'psnr' in stats['metrics'][alg]]
        
        plt.bar(psnr_algs, psnr_means, yerr=psnr_stds, capsize=5)
        plt.title('PSNR Comparison')
        plt.ylabel('PSNR (dB)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "charts", "psnr_comparison.png"), dpi=300)
        plt.close()
        
        # 2. SSIM比较
        plt.figure(figsize=(10, 6))
        ssim_means = [stats['metrics'][alg]['ssim']['mean'] for alg in algorithms if 'ssim' in stats['metrics'][alg]]
        ssim_stds = [stats['metrics'][alg]['ssim']['std'] for alg in algorithms if 'ssim' in stats['metrics'][alg]]
        ssim_algs = [alg for alg in algorithms if 'ssim' in stats['metrics'][alg]]
        
        plt.bar(ssim_algs, ssim_means, yerr=ssim_stds, capsize=5)
        plt.title('SSIM Comparison')
        plt.ylabel('SSIM')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "charts", "ssim_comparison.png"), dpi=300)
        plt.close()
        
        # 3. 处理时间比较
        plt.figure(figsize=(10, 6))
        time_means = [stats['performance'][alg]['execution_time']['mean'] for alg in algorithms if 'execution_time' in stats['performance'][alg]]
        time_stds = [stats['performance'][alg]['execution_time']['std'] for alg in algorithms if 'execution_time' in stats['performance'][alg]]
        time_algs = [alg for alg in algorithms if 'execution_time' in stats['performance'][alg]]
        
        plt.bar(time_algs, time_means, yerr=time_stds, capsize=5)
        plt.title('Processing Time Comparison')
        plt.ylabel('Time (seconds)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "charts", "time_comparison.png"), dpi=300)
        plt.close()
        
        # 4. 雷达图
        self._create_radar_chart(stats)
        
        # 5. 热力图
        self._create_heatmap(stats)
        
        self.logger.info("可视化图表已生成")
    
    def _create_radar_chart(self, stats: Dict):
        """创建雷达图"""
        algorithms = stats['algorithms']
        metrics = ['psnr', 'ssim', 'edge_preservation']
        
        # 归一化数据
        normalized_data = {}
        for alg in algorithms:
            normalized_data[alg] = []
            for metric in metrics:
                if metric in stats['metrics'][alg]:
                    value = stats['metrics'][alg][metric]['mean']
                    # 简单归一化到0-1
                    if metric == 'psnr':
                        normalized_value = min(value / 40.0, 1.0)  # 假设40dB为满分
                    else:
                        normalized_value = value  # SSIM和EPI已经在0-1范围
                    normalized_data[alg].append(normalized_value)
                else:
                    normalized_data[alg].append(0)
        
        # 创建雷达图
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        colors = ['red', 'blue', 'green', 'orange']
        for i, alg in enumerate(algorithms):
            values = normalized_data[alg] + normalized_data[alg][:1]  # 闭合数据
            ax.plot(angles, values, 'o-', linewidth=2, label=alg, color=colors[i % len(colors)])
            ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title('Algorithm Performance Radar Chart')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "charts", "radar_chart.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_heatmap(self, stats: Dict):
        """创建热力图"""
        algorithms = stats['algorithms']
        metrics = ['psnr', 'ssim', 'mse', 'mae', 'edge_preservation']
        
        # 准备数据
        data = []
        for alg in algorithms:
            row = []
            for metric in metrics:
                if metric in stats['metrics'][alg]:
                    value = stats['metrics'][alg][metric]['mean']
                    row.append(value)
                else:
                    row.append(0)
            data.append(row)
        
        # 创建热力图
        plt.figure(figsize=(10, 6))
        sns.heatmap(data, 
                   xticklabels=metrics, 
                   yticklabels=algorithms, 
                   annot=True, 
                   fmt='.3f', 
                   cmap='viridis')
        plt.title('Algorithm Performance Heatmap')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "charts", "metrics_heatmap.png"), dpi=300)
        plt.close()
    
    def _generate_report(self, stats: Dict):
        """生成性能报告"""
        try:
            # 生成Markdown格式的性能分析报告
            report_content = self._create_markdown_report(stats)
            
            # 保存报告
            report_file = os.path.join(self.output_dir, "reports", "performance_analysis_report.md")
            os.makedirs(os.path.dirname(report_file), exist_ok=True)
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            self.logger.info(f"性能分析报告已生成: {report_file}")
        
        except Exception as e:
            self.logger.warning(f"报告生成失败: {e}")
    
    def _create_markdown_report(self, stats: Dict) -> str:
        """创建Markdown格式的性能分析报告"""
        report = []
        report.append("# Algorithm Performance Analysis Report")
        report.append(f"\n**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"\n**Number of test samples:** {stats['num_samples']}")
        report.append(f"\n**Algorithms tested:** {', '.join(stats['algorithms'])}")
        
        # Executive Summary
        report.append("\n## Executive Summary")
        report.append("\nThis report presents a comprehensive comparison of different image deblurring algorithms.")
        
        # Key Findings
        report.append("\n## Key Findings")
        for alg in stats['algorithms']:
            if alg in stats['metrics']:
                metrics = stats['metrics'][alg]
                report.append(f"\n### {alg}")
                if 'psnr' in metrics:
                    report.append(f"- **PSNR:** {metrics['psnr']['mean']:.3f} ± {metrics['psnr']['std']:.3f} dB")
                if 'ssim' in metrics:
                    report.append(f"- **SSIM:** {metrics['ssim']['mean']:.3f} ± {metrics['ssim']['std']:.3f}")
                if alg in stats['performance'] and 'execution_time' in stats['performance'][alg]:
                    perf = stats['performance'][alg]['execution_time']
                    report.append(f"- **Processing Time:** {perf['mean']:.3f} ± {perf['std']:.3f} seconds")
        
        # Detailed Results
        report.append("\n## Detailed Results")
        report.append("\n### Summary Statistics")
        report.append("\n| Algorithm | PSNR (dB) | SSIM | MSE | MAE | Processing Time (s) |")
        report.append("|-----------|-----------|------|-----|-----|---------------------|")
        
        for alg in stats['algorithms']:
            if alg in stats['metrics']:
                metrics = stats['metrics'][alg]
                perf = stats['performance'].get(alg, {})
                
                psnr = f"{metrics.get('psnr', {}).get('mean', 0):.3f}" if 'psnr' in metrics else "N/A"
                ssim = f"{metrics.get('ssim', {}).get('mean', 0):.3f}" if 'ssim' in metrics else "N/A"
                mse = f"{metrics.get('mse', {}).get('mean', 0):.6f}" if 'mse' in metrics else "N/A"
                mae = f"{metrics.get('mae', {}).get('mean', 0):.6f}" if 'mae' in metrics else "N/A"
                time_val = f"{perf.get('execution_time', {}).get('mean', 0):.3f}" if 'execution_time' in perf else "N/A"
                
                report.append(f"| {alg} | {psnr} | {ssim} | {mse} | {mae} | {time_val} |")
        
        # Algorithm Descriptions
        report.append("\n## Algorithm Descriptions")
        
        for alg in stats['algorithms']:
            if alg in stats['metrics']:
                metrics = stats['metrics'][alg]
                report.append(f"\n### {alg}")
                report.append(f"\n**Descriptive Statistics:**")
                
                for metric_name in ['psnr', 'ssim', 'mse', 'mae', 'edge_preservation']:
                    if metric_name in metrics:
                        metric_data = metrics[metric_name]
                        report.append(f"\n- **{metric_name.upper()}:**")
                        report.append(f"  - Mean: {metric_data['mean']:.6f}")
                        report.append(f"  - Std: {metric_data['std']:.6f}")
                        report.append(f"  - Min: {metric_data['min']:.6f}")
                        report.append(f"  - Max: {metric_data['max']:.6f}")
                        report.append(f"  - Median: {metric_data['median']:.6f}")
                        report.append(f"  - Count: {metric_data['count']}")
        
        return "\n".join(report)


def main():
    """主函数"""
    import argparse
    print("=" * 60)
    print("MRA-NetV2 四算法比较工具")
    print("=" * 60)

    parser = argparse.ArgumentParser(description="四算法比较")
    parser.add_argument("--test-path", dest="test_path", default="outputs/datasets/test_dataset.h5")
    parser.add_argument("--unet-path", dest="unet_path", default="outputs/models/best_unet.pth")
    parser.add_argument("--mranet-path", dest="mranet_path", default="outputs/models/best_mra_net_fast.pth")
    parser.add_argument("--num-samples", dest="num_samples", type=int, default=None)
    parser.add_argument("--no-interactive", dest="no_interactive", action="store_true", help="非交互模式，直接使用提供/默认路径")
    # TTA/滑窗参数
    parser.add_argument("--tta-tile-size", dest="tta_tile_size", type=int, default=384, help="滑窗tile大小，默认384")
    parser.add_argument("--tta-overlap", dest="tta_overlap", type=int, default=64, help="滑窗重叠，默认64")
    parser.add_argument("--tta-scales", dest="tta_scales", type=str, default="1.0,0.75", help="多尺度TTA比例，逗号分隔，例如: 1.0,0.75")
    args, unknown = parser.parse_known_args()

    # 创建比较实例
    comparison = AlgorithmComparison()
    # 注入评测增强参数
    try:
        tta_scales = [float(s.strip()) for s in (args.tta_scales.split(',') if args.tta_scales else ["1.0"])]
    except Exception:
        tta_scales = [1.0]
    comparison.tta_tile_size = max(64, int(args.tta_tile_size))
    comparison.tta_overlap = max(0, int(args.tta_overlap))
    comparison.tta_scales = [s for s in tta_scales if s > 0]

    # 默认路径
    default_test_path = args.test_path
    default_unet_path = args.unet_path
    default_mra_net_path = args.mranet_path

    if args.no_interactive:
        test_dataset_path = default_test_path
        unet_model_path = default_unet_path if os.path.exists(default_unet_path) else None
        mra_net_model_path = default_mra_net_path if os.path.exists(default_mra_net_path) else None
        if not os.path.exists(test_dataset_path):
            print("❌ 测试数据集文件不存在！")
            return
        num_samples = args.num_samples
    else:
        # 交互式输入
        test_dataset_path = input(f"请输入测试数据集路径 (默认: {default_test_path}): ").strip()
        if not test_dataset_path:
            test_dataset_path = default_test_path
            print(f"使用默认测试数据集: {default_test_path}")
        if not os.path.exists(test_dataset_path):
            print("❌ 测试数据集文件不存在！")
            return

        # U-Net
        if os.path.exists(default_unet_path):
            print(f"✅ 发现默认U-Net模型: {default_unet_path}")
            unet_model_path = input(f"请输入U-Net模型路径 (默认: {default_unet_path}, 直接回车使用默认): ").strip()
            if not unet_model_path:
                unet_model_path = default_unet_path
                print(f"使用默认U-Net模型: {default_unet_path}")
        else:
            unet_model_path = input("请输入U-Net模型路径 (可选，直接回车跳过): ").strip()
        if unet_model_path and not os.path.exists(unet_model_path):
            print("⚠️ U-Net模型文件不存在，将跳过U-Net测试")
            unet_model_path = None

        # MRAS-Net
        if os.path.exists(default_mra_net_path):
            print(f"✅ 发现默认MRA-Net模型: {default_mra_net_path}")
            mra_net_model_path = input(f"请输入MRA-Net模型路径 (默认: {default_mra_net_path}, 直接回车使用默认): ").strip()
            if not mra_net_model_path:
                mra_net_model_path = default_mra_net_path
                print(f"使用默认MRA-Net模型: {default_mra_net_path}")
        else:
            mra_net_model_path = input("请输入MRA-Net模型路径 (可选，直接回车跳过): ").strip()
        if mra_net_model_path and not os.path.exists(mra_net_model_path):
            print("⚠️ MRA-Net模型文件不存在，将跳过MRA-Net测试")
            mra_net_model_path = None

        num_samples_in = input("请输入测试样本数量 (可选，直接回车使用全部): ").strip()
        if num_samples_in:
            try:
                num_samples = int(num_samples_in)
            except ValueError:
                print("⚠️ 无效的样本数量，将使用全部样本")
                num_samples = None
        else:
            num_samples = None
    
    try:
        print("\n🚀 开始算法比较...")
        
        # 运行比较
        results = comparison.run_comparison(
            test_dataset_path=test_dataset_path,
            unet_model_path=unet_model_path,
            mra_net_model_path=mra_net_model_path,
            num_samples=num_samples
        )
        
        print("\n✅ 算法比较完成！")
        print(f"测试了 {len(results['algorithms'])} 个算法")
        print(f"处理了 {results['num_samples']} 个样本")
        
        # 显示简要结果
        print("\n📊 简要结果:")
        for metric in ['psnr', 'ssim']:
            if metric in results['rankings']:
                print(f"{metric.upper()} 排名: {' > '.join(results['rankings'][metric])}")
        
        print(f"\n📁 详细结果已保存到: {comparison.output_dir}")
        print("   - 图表: charts/")
        print("   - 数据: results/")
        print("   - 报告: reports/")
    
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断操作")
    except Exception as e:
        print(f"\n❌ 发生错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()