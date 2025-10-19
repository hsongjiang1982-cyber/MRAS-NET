#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据准备脚本

功能：
1. 生成微生物数据集
2. 模拟运动模糊和噪声
3. 创建训练和测试数据

使用方法：
    python data_preparation.py
"""

import os
import sys
import numpy as np
import cv2
import h5py
import json
import yaml
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# 添加tools目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'tools'))

from dataset_generator import DatasetGenerator
from blur_simulator import BlurSimulator
from logger import get_logger

class DataPreparation:
    """
    数据准备主类
    """
    
    def __init__(self, output_dir: str = "outputs/datasets", config_path: str = "config/config.yaml"):
        self.output_dir = output_dir
        self.config_path = config_path
        self.logger = get_logger(__name__)
        
        # 加载配置文件
        self.config = self._load_config()
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化组件
        self.dataset_generator = DatasetGenerator()
        self.blur_simulator = BlurSimulator()
        
        self.logger.info("数据准备工具初始化完成")
    
    def _load_config(self) -> Dict:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            self.logger.info(f"配置文件加载成功: {self.config_path}")
            return config
        except FileNotFoundError:
            self.logger.warning(f"配置文件 {self.config_path} 不存在，使用默认配置")
            return self._get_default_config()
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'dataset': {
                'num_samples': 1000,
                'train_ratio': 0.8,
                'val_ratio': 0.1,
                'test_ratio': 0.1
            },
            'image': {
                'size': [256, 256]
            }
        }
    
    def _convert_numpy_types(self, obj):
        """递归转换numpy类型为Python原生类型，以便JSON序列化"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_numpy_types(item) for item in obj)
        else:
            return obj
    
    def generate_microbe_dataset(self, 
                               num_samples: int = 1000,
                               image_size: Tuple[int, int] = (256, 256),
                               microbe_types: List[str] = None) -> str:
        """
        生成微生物数据集
        
        Args:
            num_samples: 样本数量
            image_size: 图像尺寸
            microbe_types: 微生物类型列表
            
        Returns:
            数据集保存路径
        """
        if microbe_types is None:
            microbe_types = ['bacteria', 'virus', 'fungi', 'protozoa']
        
        self.logger.info(f"开始生成微生物数据集，样本数量: {num_samples}")
        
        # 生成数据集
        dataset_path = self.dataset_generator.generate_dataset(
            total_pairs=num_samples,
            output_dir=self.output_dir
        )
        
        self.logger.info(f"微生物数据集已保存到: {dataset_path}")
        return dataset_path
    
    def simulate_motion_blur(self, 
                           dataset_path: str,
                           blur_types: List[str] = None,
                           noise_levels: List[float] = None) -> str:
        """
        模拟运动模糊和噪声
        
        Args:
            dataset_path: 原始数据集路径
            blur_types: 模糊类型列表
            noise_levels: 噪声水平列表
            
        Returns:
            模糊数据集保存路径
        """
        if blur_types is None:
            blur_types = ['defocus', 'motion', 'gaussian']
        if noise_levels is None:
            noise_levels = [0.01, 0.02, 0.05]
        
        self.logger.info("开始模拟运动模糊和噪声")
        
        # 加载原始数据集
        dataset = self._load_dataset(dataset_path)
        
        # 创建模糊数据集
        blurred_dataset = {
            'clean_images': dataset['images'],
            'blurred_images': [],
            'psfs': [],
            'blur_params': [],
            'metadata': dataset.get('metadata', {})
        }
        
        for i, clean_image in enumerate(dataset['images']):
            # 随机选择模糊类型和噪声水平
            blur_type = np.random.choice(blur_types)
            noise_level = np.random.choice(noise_levels)
            
            # 生成模糊图像和PSF
            blurred_image, psf, blur_params = self.blur_simulator.simulate_blur(
                clean_image, 
                blur_type=blur_type,
                noise_level=noise_level
            )
            
            blurred_dataset['blurred_images'].append(blurred_image)
            blurred_dataset['psfs'].append(psf)
            blurred_dataset['blur_params'].append(blur_params)
            
            if (i + 1) % 100 == 0:
                self.logger.info(f"已处理 {i + 1}/{len(dataset['images'])} 张图像")
        
        # 保存模糊数据集
        blurred_dataset_path = os.path.join(self.output_dir, "blurred_dataset.h5")
        self._save_dataset(blurred_dataset, blurred_dataset_path)
        
        self.logger.info(f"模糊数据集已保存到: {blurred_dataset_path}")
        return blurred_dataset_path
    
    def create_train_test_split(self, 
                              dataset_path: str,
                              train_ratio: float = None,
                              val_ratio: float = None) -> Dict[str, str]:
        """
        创建训练/验证/测试数据集分割
        
        Args:
            dataset_path: 原始数据集路径
            train_ratio: 训练集比例（如果为None，从配置文件读取）
            val_ratio: 验证集比例（如果为None，从配置文件读取）
            
        Returns:
            包含各数据集路径的字典
        """
        # 从配置文件获取默认比例
        if train_ratio is None:
            train_ratio = self.config.get('dataset', {}).get('train_ratio', 0.8)
        if val_ratio is None:
            val_ratio = self.config.get('dataset', {}).get('val_ratio', 0.1)
        self.logger.info("开始创建训练/验证/测试数据集分割")
        
        # 加载数据集
        dataset = self._load_dataset(dataset_path)
        
        # 计算分割索引
        total_samples = len(dataset['clean_images'])
        train_end = int(total_samples * train_ratio)
        val_end = train_end + int(total_samples * val_ratio)
        
        # 随机打乱索引
        indices = np.random.permutation(total_samples)
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        # 创建分割数据集
        splits = {
            'train': self._create_subset(dataset, train_indices),
            'val': self._create_subset(dataset, val_indices),
            'test': self._create_subset(dataset, test_indices)
        }
        
        # 保存分割数据集
        split_paths = {}
        for split_name, split_data in splits.items():
            split_path = os.path.join(self.output_dir, f"{split_name}_dataset.h5")
            self._save_dataset(split_data, split_path)
            split_paths[split_name] = split_path
            self.logger.info(f"{split_name}数据集已保存到: {split_path}")
        
        return split_paths
    
    def _save_dataset(self, dataset: Dict, filepath: str):
        """保存数据集到HDF5文件"""
        with h5py.File(filepath, 'w') as f:
            for key, value in dataset.items():
                if key == 'metadata':
                    # 保存元数据为JSON字符串，先转换numpy类型
                    converted_metadata = self._convert_numpy_types(value)
                    f.attrs['metadata'] = json.dumps(converted_metadata)
                elif isinstance(value, list):
                    # 检查列表中的元素是否为数组且形状一致
                    if len(value) > 0 and isinstance(value[0], np.ndarray):
                        # 检查所有数组的形状是否一致
                        shapes = [arr.shape for arr in value]
                        if len(set(shapes)) == 1:
                            # 形状一致，可以直接转换为numpy数组
                            f.create_dataset(key, data=np.array(value))
                        else:
                            # 形状不一致，分别保存每个数组
                            group = f.create_group(key)
                            for i, arr in enumerate(value):
                                group.create_dataset(f'item_{i}', data=arr)
                    else:
                         # 非数组列表，尝试直接转换
                         try:
                             arr = np.array(value)
                             # 检查数据类型是否为object
                             if arr.dtype == np.object_:
                                 # 保存为字符串
                                 f.attrs[f'{key}_str'] = json.dumps([str(v) for v in value])
                             else:
                                 f.create_dataset(key, data=arr)
                         except (ValueError, TypeError):
                             # 转换失败，保存为字符串
                             f.attrs[f'{key}_str'] = json.dumps([str(v) for v in value])
                else:
                    f.create_dataset(key, data=value)
    
    def _load_dataset(self, filepath: str) -> Dict:
        """从HDF5文件加载数据集"""
        dataset = {}
        with h5py.File(filepath, 'r') as f:
            # 检查文件结构并加载数据
            if 'train' in f:
                # 新格式：有train/validation/test分组
                dataset['images'] = []
                dataset['metadata'] = {}
                
                # 加载训练数据
                if 'clean' in f['train']:
                    dataset['images'] = f['train/clean'][:]
                
                # 加载属性
                for attr_name in f.attrs:
                    dataset['metadata'][attr_name] = f.attrs[attr_name]
            else:
                # 旧格式：直接在根目录
                for key in f.keys():
                    if isinstance(f[key], h5py.Dataset):
                        dataset[key] = f[key][:]
                    elif isinstance(f[key], h5py.Group):
                        # 检查是否为分组保存的数组列表
                        if all(subkey.startswith('item_') for subkey in f[key].keys()):
                            # 重新组装为列表
                            items = []
                            for i in range(len(f[key].keys())):
                                if f'item_{i}' in f[key]:
                                    items.append(f[key][f'item_{i}'][:])
                            dataset[key] = items
                        else:
                            # 普通分组
                            dataset[key] = {}
                            for subkey in f[key].keys():
                                dataset[key][subkey] = f[key][subkey][:]
                
                # 加载元数据和字符串属性
                for attr_name in f.attrs:
                    if attr_name == 'metadata':
                        dataset['metadata'] = json.loads(f.attrs['metadata'])
                    elif attr_name.endswith('_str'):
                        # 恢复字符串列表
                        key = attr_name[:-4]  # 移除'_str'后缀
                        dataset[key] = json.loads(f.attrs[attr_name])
        
        return dataset
    
    def _create_subset(self, dataset: Dict, indices: np.ndarray) -> Dict:
        """创建数据集子集"""
        subset = {}
        for key, value in dataset.items():
            if key == 'metadata':
                subset[key] = value.copy()
            elif isinstance(value, (list, np.ndarray)):
                subset[key] = [value[i] for i in indices]
            else:
                subset[key] = value
        
        return subset
    
    def generate_complete_dataset(self, 
                                num_samples: int = None,
                                image_size: Tuple[int, int] = None) -> Dict[str, str]:
        """
        生成完整的数据集（包括清晰图像、模糊图像和数据集分割）
        
        Args:
            num_samples: 样本数量（如果为None，从配置文件读取）
            image_size: 图像尺寸（如果为None，从配置文件读取）
            
        Returns:
            所有数据集路径的字典
        """
        # 从配置文件获取默认参数
        if num_samples is None:
            num_samples = self.config.get('dataset', {}).get('num_samples', 1000)
        if image_size is None:
            image_size = tuple(self.config.get('image', {}).get('size', [256, 256]))
        self.logger.info("开始生成完整数据集")
        
        # 1. 生成微生物数据集
        clean_dataset_path = self.generate_microbe_dataset(
            num_samples=num_samples,
            image_size=image_size
        )
        
        # 2. 模拟运动模糊
        blurred_dataset_path = self.simulate_motion_blur(clean_dataset_path)
        
        # 3. 创建训练/验证/测试分割
        split_paths = self.create_train_test_split(blurred_dataset_path)
        
        # 汇总所有路径
        all_paths = {
            'clean_dataset': clean_dataset_path,
            'blurred_dataset': blurred_dataset_path,
            **split_paths
        }
        
        # 保存路径信息
        paths_file = os.path.join(self.output_dir, "dataset_paths.json")
        with open(paths_file, 'w', encoding='utf-8') as f:
            json.dump(all_paths, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"完整数据集生成完成，路径信息已保存到: {paths_file}")
        return all_paths


def main():
    """主函数"""
    print("=" * 60)
    print("MRA-NetV2 数据准备工具")
    print("=" * 60)
    
    # 创建数据准备实例
    data_prep = DataPreparation()
    
    # 用户选择
    print("\n请选择操作：")
    print("1. 生成完整数据集（推荐）")
    print("2. 仅生成微生物数据集")
    print("3. 仅模拟运动模糊")
    print("4. 仅创建数据集分割")
    
    choice = input("\n请输入选择 (1-4): ").strip()
    
    try:
        if choice == '1':
            # 获取参数（默认使用配置文件中的值）
            config_samples = data_prep.config.get('dataset', {}).get('num_samples', 1000)
            user_input = input(f"请输入样本数量 (默认{config_samples}，直接回车使用配置文件值): ").strip()
            num_samples = int(user_input) if user_input else None  # None表示使用配置文件值
            
            # 生成完整数据集
            paths = data_prep.generate_complete_dataset(num_samples=num_samples)
            
            print("\n✅ 完整数据集生成成功！")
            print("生成的文件：")
            for name, path in paths.items():
                print(f"  {name}: {path}")
        
        elif choice == '2':
            num_samples = int(input("请输入样本数量 (默认1000): ") or "1000")
            path = data_prep.generate_microbe_dataset(num_samples=num_samples)
            print(f"\n✅ 微生物数据集生成成功: {path}")
        
        elif choice == '3':
            dataset_path = input("请输入原始数据集路径: ").strip()
            if not os.path.exists(dataset_path):
                print("❌ 数据集文件不存在！")
                return
            
            path = data_prep.simulate_motion_blur(dataset_path)
            print(f"\n✅ 运动模糊模拟完成: {path}")
        
        elif choice == '4':
            dataset_path = input("请输入数据集路径: ").strip()
            if not os.path.exists(dataset_path):
                print("❌ 数据集文件不存在！")
                return
            
            paths = data_prep.create_train_test_split(dataset_path)
            print("\n✅ 数据集分割完成！")
            for name, path in paths.items():
                print(f"  {name}: {path}")
        
        else:
            print("❌ 无效选择！")
    
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断操作")
    except Exception as e:
        print(f"\n❌ 发生错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()