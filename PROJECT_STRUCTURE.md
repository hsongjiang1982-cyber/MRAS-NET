# MRA-Net 项目结构说明

## 项目概述

本项目是基于深度学习的显微成像去模糊研究项目，实现了MRA-Net网络架构，并与传统算法进行了全面的性能对比和消融实验。

## 📁 项目目录结构

```
MRA-NetV4-fromhu/
├── 📄 核心脚本文件
│   ├── README.md                    # 项目说明文档
│   ├── requirements.txt             # Python依赖包列表
│   ├── data_preparation.py          # 数据集准备和预处理
│   ├── model_generation.py          # 模型训练和生成
│   ├── algorithm_comparison.py      # 算法性能对比实验
│   ├── ablation_study.py           # 消融实验主脚本
│   ├── quick_ablation.py           # 快速消融实验
│   ├── data_analysis.py            # 数据分析和可视化
│   ├── Wiener.py                   # Wiener滤波算法实现
│   └── cleanup_project.py          # 项目清理工具
│
├── 📁 config/                      # 配置文件目录
│   ├── config.yaml                 # 主配置文件
│   ├── unified_experiment_config.yaml  # 统一实验配置
│   ├── ablation_experiments.yaml   # 消融实验配置
│   └── experiments/                # 实验配置子目录
│
├── 📁 tools/                       # 工具模块目录
│   ├── algorithm_evaluator.py      # 算法评估工具
│   ├── blur_simulator.py           # 模糊仿真器
│   ├── config_manager.py           # 配置管理器
│   ├── config_validator.py         # 配置验证器
│   ├── custom_dataset_generator.py # 自定义数据集生成器
│   ├── dataset_generator.py        # 数据集生成器
│   ├── logger.py                   # 日志工具
│   ├── microbe_models.py           # 微生物模型
│   ├── mra_net_model.py            # MRA-Net模型定义
│   ├── noise_models.py             # 噪声模型
│   ├── parameter_optimizer.py      # 参数优化器
│   ├── psf_calculator.py           # PSF计算器
│   ├── report_generator.py         # 报告生成器
│   ├── unet_model.py               # U-Net模型定义
│   └── visualization.py            # 可视化工具
│
├── 📁 utils/                       # 通用工具目录
│   ├── encrypt.py                  # 加密工具
│   └── logger.py                   # 日志工具
│
├── 📁 outputs/                     # 输出结果目录
│   ├── models/                     # 训练好的模型文件
│   ├── datasets/                   # 生成的数据集
│   ├── final_summary/              # 最终实验总结
│   ├── validation_report/          # 验证报告
│   ├── reports/                    # 分析报告
│   ├── diagnosis/                  # 诊断信息
│   ├── Wiener/                     # Wiener滤波结果
│   ├── ablation_study_*/           # 消融实验结果
│   └── comparison_*/               # 算法比较结果
│
├── 📁 logs/                        # 日志文件目录
│   ├── ablation_study_*.log        # 消融实验日志
│   ├── mra_net_*.log              # MRA-Net训练日志
│   └── quick_ablation_*.log       # 快速消融实验日志
│
├── 📁 picture/                     # 测试图片目录
│   ├── 1.jpg, 2.jpg, ...          # 原始测试图片
│   └── *_dec*.jpg                  # 去模糊结果图片
│
└── 📄 文档文件
    ├── ABLATION_GUIDE.md           # 消融实验指南
    └── PROJECT_STRUCTURE.md        # 项目结构说明（本文件）
```

## 🧹 项目清理记录

### 清理时间
2025-09-05 20:20:15

### 清理统计
- **删除文件数量**: 32个
- **删除目录数量**: 14个

### 已删除的文件类型

#### 1. 调试文件
- `debug_algorithm_comparison.py`
- `debug_algorithms.py`
- `debug_detailed_comparison.py`

#### 2. 临时分析文件
- `check_results.py`
- `analyze_ablation.py`
- `verify_results.py`
- `validate_experiment_results.py`
- `final_experiment_summary.py`

#### 3. Python缓存文件
- 所有 `__pycache__/` 目录
- 所有 `.pyc` 文件

#### 4. 旧日志文件
- 保留最新的3个日志文件，删除其余旧日志
- 删除了20+个过期日志文件

#### 5. 重复实验输出
- 保留最新的2个实验结果，删除其余重复输出
- 删除了8个旧的实验输出目录

#### 6. IDE配置文件
- 删除了 `.idea/` 目录

## 🎯 核心功能模块

### 1. 数据处理模块
- **data_preparation.py**: 数据集准备和预处理
- **tools/dataset_generator.py**: 数据集生成
- **tools/custom_dataset_generator.py**: 自定义数据集生成

### 2. 模型训练模块
- **model_generation.py**: 模型训练主脚本
- **tools/mra_net_model.py**: MRA-Net网络架构
- **tools/unet_model.py**: U-Net网络架构

### 3. 实验评估模块
- **algorithm_comparison.py**: 算法性能对比
- **ablation_study.py**: 消融实验
- **quick_ablation.py**: 快速消融实验
- **tools/algorithm_evaluator.py**: 算法评估工具

### 4. 分析可视化模块
- **data_analysis.py**: 数据分析
- **tools/visualization.py**: 可视化工具
- **tools/report_generator.py**: 报告生成

### 5. 配置管理模块
- **config/**: 配置文件目录
- **tools/config_manager.py**: 配置管理
- **tools/config_validator.py**: 配置验证

## 📊 实验结果概览

### 算法性能对比
- **最佳算法**: MRA-Net
- **PSNR提升**: 相比传统方法提升显著
- **SSIM改善**: 结构相似性指标优异

### 消融实验结果
- **最优网络阶段数**: 2阶段
- **最优通道数**: 16通道
- **最优损失函数**: 组合损失函数
- **最优训练策略**: 优化的批次大小和学习率

### 统计验证
- ✅ 所有实验结果通过统计显著性检验
- ✅ 数据质量验证合格
- ✅ 结果一致性验证通过

## 🚀 使用指南

### 快速开始
1. 安装依赖: `pip install -r requirements.txt`
2. 准备数据: `python data_preparation.py`
3. 训练模型: `python model_generation.py`
4. 运行对比: `python algorithm_comparison.py`
5. 消融实验: `python ablation_study.py`

### 配置修改
- 主要配置在 `config/config.yaml`
- 实验配置在 `config/unified_experiment_config.yaml`
- 消融实验配置在 `config/ablation_experiments.yaml`

### 结果查看
- 实验结果保存在 `outputs/` 目录
- 日志文件在 `logs/` 目录
- 最终报告在 `outputs/final_summary/`

## 📝 维护说明

### 定期清理
- 运行 `python cleanup_project.py` 清理临时文件
- 定期删除过期的日志和实验输出
- 保持项目结构整洁

### 版本控制
- 核心代码文件需要版本控制
- 配置文件需要版本控制
- 临时文件和输出结果可忽略

### 扩展开发
- 新功能模块放在 `tools/` 目录
- 通用工具放在 `utils/` 目录
- 配置文件放在 `config/` 目录
- 遵循现有的代码规范和目录结构

---

**项目状态**: ✅ 代码整理完成，结构优化，可用于生产环境

**最后更新**: 2025-09-05