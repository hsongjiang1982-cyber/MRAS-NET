#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wiener反卷积处理脚本

功能：
- 对指定文件夹中的图片进行Wiener反卷积处理
- 使用config.yaml中的物理参数
- 生成处理结果和分析报告

作者：显微成像研究助手
日期：2025-08-31
"""

import os
import sys
import numpy as np
import cv2
import yaml
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import logging
from datetime import datetime
from pathlib import Path

# 添加tools目录到路径
sys.path.append('tools')
from logger import get_logger
from psf_calculator import AngularSpectrumPSF

class WienerDeconvolution:
    """
    Wiener反卷积处理类
    
    实现基于物理参数的Wiener反卷积算法，用于图像去模糊处理
    """
    
    def __init__(self, config_path: str):
        """
        初始化Wiener反卷积处理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = get_logger('wiener_deconv')
        
        # 初始化PSF计算器
        self.psf_calculator = AngularSpectrumPSF(config_path)
        
        # 创建输出目录
        self.output_dir = Path('outputs/Wiener')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'images').mkdir(exist_ok=True)
        (self.output_dir / 'charts').mkdir(exist_ok=True)
        (self.output_dir / 'reports').mkdir(exist_ok=True)
        
        self.logger.info(f"Wiener反卷积处理器初始化完成")
    
    def _load_config(self) -> dict:
        """
        加载配置文件
        
        Returns:
            配置字典
        """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            print(f"配置文件加载失败: {e}")
            sys.exit(1)
    
    def generate_psf(self, image_shape: tuple, defocus: float = 0.0) -> np.ndarray:
        """
        生成点扩散函数(PSF)
        
        Args:
            image_shape: 图像尺寸
            defocus: 离焦量(微米)
            
        Returns:
            PSF数组
        """
        try:
            # 使用配置参数生成PSF
            # 使用图像的最小尺寸作为PSF尺寸
            psf_size = min(image_shape)
            psf = self.psf_calculator.calculate_psf(
                psf_size=psf_size,
                defocus=defocus
            )
            
            # 如果PSF尺寸与图像尺寸不匹配，需要调整
            if psf.shape != image_shape:
                # 将PSF调整到图像尺寸
                psf_resized = np.zeros(image_shape)
                h, w = psf.shape
                start_h = (image_shape[0] - h) // 2
                start_w = (image_shape[1] - w) // 2
                psf_resized[start_h:start_h+h, start_w:start_w+w] = psf
                psf = psf_resized
            
            # 归一化PSF
            psf = psf / np.sum(psf)
            
            return psf
            
        except Exception as e:
            self.logger.error(f"PSF生成失败: {e}")
            # 使用简单高斯PSF作为备选
            sigma = 2.0
            x, y = np.meshgrid(np.arange(image_shape[1]), np.arange(image_shape[0]))
            cx, cy = image_shape[1]//2, image_shape[0]//2
            psf = np.exp(-((x-cx)**2 + (y-cy)**2) / (2*sigma**2))
            psf = psf / np.sum(psf)
            return psf
    
    def wiener_filter(self, blurred_image: np.ndarray, psf: np.ndarray, 
                     noise_variance: float = 0.01) -> np.ndarray:
        """
        执行Wiener反卷积
        
        Args:
            blurred_image: 模糊图像
            psf: 点扩散函数
            noise_variance: 噪声方差
            
        Returns:
            反卷积后的图像
        """
        try:
            # 转换到频域，使用fftshift确保频域中心正确
            blurred_fft = fftshift(fft2(blurred_image))
            psf_fft = fftshift(fft2(psf, s=blurred_image.shape))
            
            # Wiener滤波器
            psf_conj = np.conj(psf_fft)
            psf_abs_sq = np.abs(psf_fft)**2
            
            # 避免除零
            wiener_filter = psf_conj / (psf_abs_sq + noise_variance)
            
            # 应用滤波器
            restored_fft = blurred_fft * wiener_filter
            
            # 转换回空域，使用ifftshift确保正确的逆变换
            restored_image = np.real(ifft2(ifftshift(restored_fft)))
            
            # 确保值在合理范围内
            restored_image = np.clip(restored_image, 0, 1)
            
            return restored_image
            
        except Exception as e:
            self.logger.error(f"Wiener反卷积失败: {e}")
            return blurred_image
    
    def load_and_preprocess_image(self, image_path: str) -> np.ndarray:
        """
        加载和预处理图像
        
        Args:
            image_path: 图像路径
            
        Returns:
            预处理后的图像数组
        """
        try:
            # 读取图像
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"无法读取图像: {image_path}")
            
            # 调整尺寸到配置指定的大小
            target_size = tuple(self.config['image']['size'])
            if image.shape != target_size:
                image = cv2.resize(image, target_size)
            
            # 归一化到[0,1]
            image = image.astype(np.float32) / 255.0
            
            return image
            
        except Exception as e:
            self.logger.error(f"图像预处理失败: {e}")
            return None
    
    def calculate_image_quality_metrics(self, original: np.ndarray, 
                                      restored: np.ndarray) -> dict:
        """
        计算图像质量指标
        
        Args:
            original: 原始图像
            restored: 恢复图像
            
        Returns:
            质量指标字典
        """
        try:
            # PSNR
            mse = np.mean((original - restored) ** 2)
            if mse == 0:
                psnr = float('inf')
            else:
                psnr = 20 * np.log10(1.0 / np.sqrt(mse))
            
            # SSIM (简化版本)
            mu1 = np.mean(original)
            mu2 = np.mean(restored)
            sigma1 = np.var(original)
            sigma2 = np.var(restored)
            sigma12 = np.mean((original - mu1) * (restored - mu2))
            
            c1 = 0.01 ** 2
            c2 = 0.03 ** 2
            
            ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
                   ((mu1**2 + mu2**2 + c1) * (sigma1 + sigma2 + c2))
            
            # 梯度相似性
            grad_orig = np.gradient(original)
            grad_rest = np.gradient(restored)
            grad_similarity = np.corrcoef(grad_orig[0].flatten(), 
                                        grad_rest[0].flatten())[0, 1]
            
            return {
                'psnr': psnr,
                'ssim': ssim,
                'gradient_similarity': grad_similarity,
                'mse': mse
            }
            
        except Exception as e:
            self.logger.error(f"质量指标计算失败: {e}")
            return {'psnr': 0, 'ssim': 0, 'gradient_similarity': 0, 'mse': 1}
    
    def simulate_blur(self, image: np.ndarray, psf: np.ndarray, noise_var: float = 0.01) -> np.ndarray:
        """
        模拟图像模糊过程
        
        Args:
            image: 原始清晰图像
            psf: 点扩散函数
            noise_var: 噪声方差
            
        Returns:
            模糊图像
        """
        try:
            # 在频域进行卷积，使用fftshift确保频域中心正确
            image_fft = fftshift(fft2(image))
            psf_fft = fftshift(fft2(psf, s=image.shape))
            
            # 卷积操作
            blurred_fft = image_fft * psf_fft
            blurred_image = np.real(ifft2(ifftshift(blurred_fft)))
            
            # 添加噪声
            noise = np.random.normal(0, np.sqrt(noise_var), image.shape)
            blurred_image += noise
            
            # 确保值在合理范围内
            blurred_image = np.clip(blurred_image, 0, 1)
            
            return blurred_image
            
        except Exception as e:
            self.logger.error(f"模糊模拟失败: {e}")
            return image
    
    def process_single_image(self, image_path: str, defocus: float = 1.0, 
                           noise_var: float = 0.01) -> dict:
        """
        处理单张图像
        
        Args:
            image_path: 图像路径
            defocus: 离焦量
            noise_var: 噪声方差
            
        Returns:
            处理结果字典
        """
        try:
            self.logger.info(f"开始处理图像: {image_path}")
            
            # 加载图像
            original_image = self.load_and_preprocess_image(image_path)
            if original_image is None:
                return None
            
            # 生成PSF
            psf = self.generate_psf(original_image.shape, defocus)
            
            # 直接对原始图像执行Wiener反卷积处理
            restored_image = self.wiener_filter(original_image, psf, noise_var)
            
            # 计算质量指标（原始图像 vs 处理后图像）
            metrics = self.calculate_image_quality_metrics(original_image, restored_image)
            
            # 保存结果图像
            image_name = Path(image_path).stem
            output_path = self.output_dir / 'images' / f'{image_name}_wiener_restored.png'
            
            # 保存为0-255范围的图像
            restored_uint8 = (restored_image * 255).astype(np.uint8)
            cv2.imwrite(str(output_path), restored_uint8)
            
            result = {
                'image_name': image_name,
                'original_path': image_path,
                'restored_path': str(output_path),
                'defocus': defocus,
                'noise_variance': noise_var,
                'metrics': metrics,
                'original_image': original_image,
                'restored_image': restored_image,
                'psf': psf
            }
            
            self.logger.info(f"图像处理完成: {image_name}, PSNR: {metrics['psnr']:.2f}")
            return result
            
        except Exception as e:
            self.logger.error(f"图像处理失败: {e}")
            return None
    
    def create_comparison_plot(self, results: list):
        """
        创建对比图表
        
        Args:
            results: 处理结果列表
        """
        try:
            n_images = len(results)
            fig, axes = plt.subplots(3, n_images, figsize=(4*n_images, 12))
            
            if n_images == 1:
                axes = axes.reshape(-1, 1)
            
            for i, result in enumerate(results):
                if result is None:
                    continue
                
                # 原始图像
                axes[0, i].imshow(result['original_image'], cmap='gray')
                axes[0, i].set_title(f"Original: {result['image_name']}")
                axes[0, i].axis('off')
                
                # 恢复图像
                axes[1, i].imshow(result['restored_image'], cmap='gray')
                axes[1, i].set_title(f"Wiener Restored\nPSNR: {result['metrics']['psnr']:.2f}")
                axes[1, i].axis('off')
                
                # PSF
                axes[2, i].imshow(result['psf'], cmap='hot')
                axes[2, i].set_title(f"PSF (defocus: {result['defocus']:.1f}μm)")
                axes[2, i].axis('off')
            
            plt.tight_layout()
            chart_path = self.output_dir / 'charts' / 'wiener_comparison.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"对比图表已保存: {chart_path}")
            
        except Exception as e:
            self.logger.error(f"图表创建失败: {e}")
    
    def create_metrics_plot(self, results: list):
        """
        创建质量指标图表
        
        Args:
            results: 处理结果列表
        """
        try:
            valid_results = [r for r in results if r is not None]
            if not valid_results:
                return
            
            image_names = [r['image_name'] for r in valid_results]
            psnr_values = [r['metrics']['psnr'] for r in valid_results]
            ssim_values = [r['metrics']['ssim'] for r in valid_results]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # PSNR图表
            bars1 = ax1.bar(image_names, psnr_values, color='skyblue', alpha=0.7)
            ax1.set_title('PSNR Values After Wiener Deconvolution')
            ax1.set_ylabel('PSNR (dB)')
            ax1.tick_params(axis='x', rotation=45)
            
            # 添加数值标签
            for bar, value in zip(bars1, psnr_values):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{value:.2f}', ha='center', va='bottom')
            
            # SSIM图表
            bars2 = ax2.bar(image_names, ssim_values, color='lightcoral', alpha=0.7)
            ax2.set_title('SSIM Values After Wiener Deconvolution')
            ax2.set_ylabel('SSIM')
            ax2.tick_params(axis='x', rotation=45)
            
            # 添加数值标签
            for bar, value in zip(bars2, ssim_values):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            chart_path = self.output_dir / 'charts' / 'wiener_metrics.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"质量指标图表已保存: {chart_path}")
            
        except Exception as e:
            self.logger.error(f"指标图表创建失败: {e}")
    
    def generate_report(self, results: list):
        """
        生成分析报告
        
        Args:
            results: 处理结果列表
        """
        try:
            valid_results = [r for r in results if r is not None]
            
            report_content = f"""# Wiener反卷积处理报告

## 实验概述
- 处理时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 处理图像数量：{len(valid_results)}
- 算法：Wiener反卷积
- 配置文件：{self.config_path}

## 物理参数设置
- 波长：{self.config['image']['wavelength']} μm
- 像素尺寸：{self.config['image']['pixel_size']} μm
- 数值孔径：{self.config['psf']['numerical_aperture']}
- PSF方法：{self.config['psf']['method']}

## 处理结果统计

| 图像名称 | PSNR (dB) | SSIM | 梯度相似性 | MSE |
|----------|-----------|------|------------|-----|
"""
            
            total_psnr = 0
            total_ssim = 0
            
            for result in valid_results:
                metrics = result['metrics']
                report_content += f"| {result['image_name']} | {metrics['psnr']:.2f} | {metrics['ssim']:.3f} | {metrics['gradient_similarity']:.3f} | {metrics['mse']:.6f} |\n"
                total_psnr += metrics['psnr']
                total_ssim += metrics['ssim']
            
            if valid_results:
                avg_psnr = total_psnr / len(valid_results)
                avg_ssim = total_ssim / len(valid_results)
                
                report_content += f"""

## 平均性能指标
- 平均PSNR：{avg_psnr:.2f} dB
- 平均SSIM：{avg_ssim:.3f}

## 算法分析

### Wiener反卷积原理
Wiener反卷积是一种基于最小均方误差准则的图像复原方法，在频域中通过以下公式实现：

```
H_wiener(u,v) = H*(u,v) / [|H(u,v)|² + Sn(u,v)/Sf(u,v)]
```

其中：
- H*(u,v)是PSF的共轭
- |H(u,v)|²是PSF的功率谱
- Sn(u,v)/Sf(u,v)是噪声与信号的功率谱比

### 处理效果评估
1. **PSNR (峰值信噪比)**：值越高表示图像质量越好
2. **SSIM (结构相似性)**：值越接近1表示结构保持越好
3. **梯度相似性**：评估边缘和细节的保持程度

### 参数影响分析
- **离焦量**：影响PSF的形状和大小
- **噪声方差**：控制反卷积的正则化强度
- **数值孔径**：决定系统的分辨能力

## 结论
基于配置参数的Wiener反卷积处理已完成，平均PSNR为{avg_psnr:.2f}dB，表明算法在图像复原方面取得了{'良好' if avg_psnr > 25 else '一般' if avg_psnr > 20 else '有限'}的效果。

## 文件输出
- 处理后图像：`outputs/Wiener/images/`
- 分析图表：`outputs/Wiener/charts/`
- 本报告：`outputs/Wiener/reports/`
"""
            
            # 保存报告
            report_path = self.output_dir / 'reports' / 'wiener_deconvolution_report.md'
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            self.logger.info(f"分析报告已生成: {report_path}")
            
        except Exception as e:
            self.logger.error(f"报告生成失败: {e}")
    
    def process_image_folder(self, folder_path: str, defocus_range: list = None, 
                           noise_var: float = 0.01):
        """
        处理文件夹中的所有图像
        
        Args:
            folder_path: 图像文件夹路径
            defocus_range: 离焦范围列表
            noise_var: 噪声方差
        """
        try:
            self.logger.info(f"开始处理文件夹: {folder_path}")
            
            # 获取图像文件列表
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(Path(folder_path).glob(f'*{ext}'))
                image_files.extend(Path(folder_path).glob(f'*{ext.upper()}'))
            
            if not image_files:
                self.logger.warning(f"文件夹中未找到图像文件: {folder_path}")
                return
            
            self.logger.info(f"找到 {len(image_files)} 个图像文件")
            
            # 设置默认离焦范围
            if defocus_range is None:
                defocus_range = [1.0] * len(image_files)  # 为每个图像使用1.0μm离焦
            elif len(defocus_range) == 1:
                defocus_range = defocus_range * len(image_files)
            
            # 处理每个图像
            results = []
            for i, image_file in enumerate(image_files):
                defocus = defocus_range[i] if i < len(defocus_range) else 1.0
                result = self.process_single_image(str(image_file), defocus, noise_var)
                results.append(result)
            
            # 生成图表和报告
            self.create_comparison_plot(results)
            self.create_metrics_plot(results)
            self.generate_report(results)
            
            self.logger.info("所有图像处理完成")
            
        except Exception as e:
            self.logger.error(f"文件夹处理失败: {e}")

def main():
    """
    主函数
    """
    try:
        # 配置路径
        config_path = 'config/config.yaml'
        image_folder = 'picture'
        
        # 检查路径是否存在
        if not os.path.exists(config_path):
            print(f"配置文件不存在: {config_path}")
            return
        
        if not os.path.exists(image_folder):
            print(f"图像文件夹不存在: {image_folder}")
            return
        
        # 创建Wiener反卷积处理器
        wiener_processor = WienerDeconvolution(config_path)
        
        # 处理图像文件夹
        # 可以为不同图像设置不同的离焦量
        defocus_values = [0.5, 1.0, 1.5, 2.0, 1.0]  # 对应5张图片的离焦量
        noise_variance = 0.02  # 噪声方差
        
        wiener_processor.process_image_folder(
            image_folder, 
            defocus_range=defocus_values,
            noise_var=noise_variance
        )
        
        print("\n=== Wiener反卷积处理完成 ===")
        print(f"结果保存在: outputs/Wiener/")
        print(f"- 处理后图像: outputs/Wiener/images/")
        print(f"- 分析图表: outputs/Wiener/charts/")
        print(f"- 详细报告: outputs/Wiener/reports/")
        
    except Exception as e:
        print(f"程序执行失败: {e}")
        logging.error(f"主程序执行失败: {e}")

if __name__ == "__main__":
    main()