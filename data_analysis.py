#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MRA-Net 数据集分析工具
独立的数据分析模块，从main.py中提取

功能：
- 数据集统计分析
- 可视化生成
- 分析报告生成
- 自动选择最新数据集

使用示例：
    python data_analysis.py
    python data_analysis.py --dataset-path outputs/datasets/test_dataset.h5
    python data_analysis.py --no-visualizations
"""

import os
import sys
import argparse
import glob
from datetime import datetime
from typing import Dict, Any, Optional

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tools.logger import get_logger
from tools.config_manager import ConfigManager
from tools.visualization import DatasetAnalyzer


class DataAnalysisTool:
    """
    数据分析工具类
    
    提供数据集分析的完整功能
    """
    
    def __init__(self, config_path: str = None):
        """
        初始化数据分析工具
        
        Args:
            config_path: 配置文件路径
        """
        self.logger = get_logger()
        self.config_manager = ConfigManager(config_path)
        self.analyzer = DatasetAnalyzer(config_path)
        
        self.logger.info("数据分析工具初始化完成")
    
    def find_latest_microbe_dataset(self) -> Optional[str]:
        """
        查找最新的microbe_dataset数据集
        
        Returns:
            最新数据集的路径，如果没有找到则返回None
        """
        datasets_dir = os.path.join(
            self.config_manager.get_config('output.base_dir'),
            self.config_manager.get_config('output.datasets_dir')
        )
        
        # 查找所有microbe_dataset目录
        pattern = os.path.join(datasets_dir, 'microbe_dataset_*')
        microbe_dirs = glob.glob(pattern)
        
        if not microbe_dirs:
            self.logger.warning("未找到任何microbe_dataset目录")
            return None
        
        # 按时间戳排序，获取最新的
        microbe_dirs.sort(reverse=True)
        latest_dir = microbe_dirs[0]
        
        # 检查dataset.h5文件是否存在
        dataset_path = os.path.join(latest_dir, 'dataset.h5')
        if os.path.exists(dataset_path):
            self.logger.info(f"找到最新数据集: {dataset_path}")
            return dataset_path
        else:
            self.logger.warning(f"最新目录中未找到dataset.h5文件: {latest_dir}")
            return None
    
    def analyze_dataset(self, dataset_path: str, 
                       generate_visualizations: bool = True,
                       generate_report: bool = True) -> Dict[str, Any]:
        """
        分析数据集
        
        Args:
            dataset_path: 数据集路径
            generate_visualizations: 是否生成可视化
            generate_report: 是否生成报告
            
        Returns:
            分析结果
        """
        self.logger.info(f"开始分析数据集: {dataset_path}")
        
        # 检查数据集文件是否存在
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"数据集文件不存在: {dataset_path}")
        
        # 统计分析
        stats = self.analyzer.analyze_dataset_statistics(dataset_path)
        
        # 创建带时间戳的分析目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        analysis_dir = os.path.join(
            self.config_manager.get_config('output.base_dir'),
            self.config_manager.get_config('output.datasets_dir'),
            f'dataset_analysis_{timestamp}'
        )
        
        # 生成可视化
        if generate_visualizations:
            self.logger.info("生成可视化图表...")
            self.analyzer.visualize_dataset_overview(dataset_path, analysis_dir)
            self.analyzer.visualize_sample_pairs(dataset_path, save_dir=analysis_dir)
            self.analyzer.visualize_quality_analysis(dataset_path, analysis_dir)
            self.analyzer.create_interactive_dashboard(dataset_path, analysis_dir)
        
        # 生成报告
        if generate_report:
            self.logger.info("生成分析报告...")
            self.analyzer.generate_analysis_report(dataset_path, analysis_dir)
        
        self.logger.info(f"数据集分析完成，结果保存到: {analysis_dir}")
        return stats, analysis_dir
    
    def get_dataset_summary(self, dataset_path: str) -> Dict[str, Any]:
        """
        获取数据集简要信息
        
        Args:
            dataset_path: 数据集路径
            
        Returns:
            数据集简要信息
        """
        stats = self.analyzer.analyze_dataset_statistics(dataset_path)
        
        summary = {
            'dataset_path': dataset_path,
            'total_samples': stats['basic_info']['total_samples'],
            'image_shape': stats['basic_info']['image_shape'],
            'psnr_mean': stats['quality_metrics']['psnr_mean'],
            'snr_mean': stats['quality_metrics']['snr_mean']
        }
        
        return summary


def create_argument_parser() -> argparse.ArgumentParser:
    """
    创建命令行参数解析器
    
    Returns:
        参数解析器
    """
    parser = argparse.ArgumentParser(
        description='MRA-Net数据集分析工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  分析默认数据集（自动选择最新的microbe_dataset）:
    python data_analysis.py
  
  分析指定数据集:
    python data_analysis.py --dataset-path outputs/datasets/test_dataset.h5
  
  只生成统计信息，不生成可视化:
    python data_analysis.py --no-visualizations
  
  只生成可视化，不生成报告:
    python data_analysis.py --no-report
  
  获取数据集简要信息:
    python data_analysis.py --summary-only
"""
    )
    
    # 数据集路径参数
    parser.add_argument('--dataset-path', type=str, 
                       help='数据集路径（如果不指定，将自动选择最新的microbe_dataset）')
    
    # 分析选项
    parser.add_argument('--no-visualizations', action='store_true', 
                       help='不生成可视化图表')
    parser.add_argument('--no-report', action='store_true', 
                       help='不生成分析报告')
    parser.add_argument('--summary-only', action='store_true', 
                       help='只显示数据集简要信息')
    
    # 配置选项
    parser.add_argument('--config', type=str, 
                       help='配置文件路径')
    
    # 日志选项
    parser.add_argument('--verbose', action='store_true', 
                       help='详细输出')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       help='日志级别')
    
    return parser


def prompt_for_dataset_selection(tool: DataAnalysisTool) -> str:
    """
    提示用户选择数据集
    
    Args:
        tool: 数据分析工具实例
        
    Returns:
        选择的数据集路径
    """
    print("\n=== MRA-Net 数据集分析工具 ===")
    print("\n请选择要分析的数据集:")
    
    # 尝试找到最新的microbe_dataset
    latest_dataset = tool.find_latest_microbe_dataset()
    
    if latest_dataset:
        print(f"\n1. 使用最新的microbe_dataset (推荐)")
        print(f"   路径: {latest_dataset}")
        
        # 显示数据集简要信息
        try:
            summary = tool.get_dataset_summary(latest_dataset)
            print(f"   样本数: {summary['total_samples']}")
            print(f"   图像尺寸: {summary['image_shape']}")
            print(f"   平均PSNR: {summary['psnr_mean']:.2f} dB")
            print(f"   平均SNR: {summary['snr_mean']:.2f} dB")
        except Exception as e:
            print(f"   (无法获取详细信息: {e})")
    
    print("\n2. 手动输入数据集路径")
    print("\n3. 退出")
    
    while True:
        try:
            choice = input("\n请输入选择 (1/2/3): ").strip()
            
            if choice == '1' and latest_dataset:
                return latest_dataset
            elif choice == '2':
                dataset_path = input("请输入数据集路径: ").strip()
                if os.path.exists(dataset_path):
                    return dataset_path
                else:
                    print(f"错误: 文件不存在 - {dataset_path}")
                    continue
            elif choice == '3':
                print("退出程序")
                sys.exit(0)
            else:
                print("无效选择，请重新输入")
                continue
        except KeyboardInterrupt:
            print("\n\n程序被用户中断")
            sys.exit(0)


def main():
    """
    主函数
    """
    parser = create_argument_parser()
    args = parser.parse_args()
    
    try:
        # 初始化工具
        config_path = getattr(args, 'config', None)
        tool = DataAnalysisTool(config_path)
        
        # 设置日志级别
        if args.verbose:
            import logging
            logging.getLogger().setLevel(logging.DEBUG)
        
        # 确定数据集路径
        dataset_path = args.dataset_path
        
        if not dataset_path:
            # 如果没有指定数据集路径，提示用户选择
            dataset_path = prompt_for_dataset_selection(tool)
        
        # 检查数据集文件是否存在
        if not os.path.exists(dataset_path):
            print(f"\n错误: 数据集文件不存在 - {dataset_path}")
            sys.exit(1)
        
        print(f"\n开始分析数据集: {dataset_path}")
        
        # 如果只需要简要信息
        if args.summary_only:
            summary = tool.get_dataset_summary(dataset_path)
            print("\n=== 数据集简要信息 ===")
            print(f"数据集路径: {summary['dataset_path']}")
            print(f"总样本数: {summary['total_samples']}")
            print(f"图像尺寸: {summary['image_shape']}")
            print(f"平均PSNR: {summary['psnr_mean']:.2f} dB")
            print(f"平均SNR: {summary['snr_mean']:.2f} dB")
            return
        
        # 执行完整分析
        stats, analysis_dir = tool.analyze_dataset(
            dataset_path,
            generate_visualizations=not args.no_visualizations,
            generate_report=not args.no_report
        )
        
        # 显示分析结果
        print("\n=== 分析完成 ===")
        print(f"数据集路径: {dataset_path}")
        print(f"总样本数: {stats['basic_info']['total_samples']}")
        print(f"图像尺寸: {stats['basic_info']['image_shape']}")
        print(f"平均PSNR: {stats['quality_metrics']['psnr_mean']:.2f} dB")
        print(f"平均SNR: {stats['quality_metrics']['snr_mean']:.2f} dB")
        print(f"\n分析结果保存到: {analysis_dir}")
        
        if not args.no_visualizations:
            print("\n生成的可视化文件:")
            print(f"  - 数据集概览: {os.path.join(analysis_dir, 'dataset_overview.png')}")
            print(f"  - 样本对比: {os.path.join(analysis_dir, 'sample_pairs.png')}")
            print(f"  - 质量分析: {os.path.join(analysis_dir, 'quality_analysis.png')}")
            print(f"  - 交互式仪表板: {os.path.join(analysis_dir, 'interactive_dashboard.html')}")
        
        if not args.no_report:
            print(f"\n分析报告: {os.path.join(analysis_dir, 'analysis_report.md')}")
        
        print("\n分析完成！")
    
    except Exception as e:
        print(f"\n错误: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()