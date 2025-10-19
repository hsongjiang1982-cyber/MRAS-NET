# MRA-Net消融实验使用指南

本指南将帮助您快速开始MRA-Net的消融实验，为论文撰写提供充分的实验支撑。

## 📋 目录

1. [快速开始](#快速开始)
2. [实验类型](#实验类型)
3. [使用方法](#使用方法)
4. [结果分析](#结果分析)
5. [常见问题](#常见问题)
6. [进阶使用](#进阶使用)

## 🚀 快速开始

### 环境准备

确保您已经安装了必要的依赖：

```bash
# 安装基础依赖
pip install torch torchvision
pip install numpy matplotlib seaborn
pip install scikit-image scikit-learn
pip install pyyaml h5py

# 可选：安装用于报告生成的依赖
pip install markdown2 weasyprint
```

### 第一次运行

1. **查看可用实验**：
   ```bash
   python quick_ablation.py --list
   ```

2. **运行单个实验**（推荐新手）：
   ```bash
   python quick_ablation.py --experiment stages
   ```

3. **查看结果**：
   实验完成后，结果将保存在 `outputs/quick_ablation_YYYYMMDD/` 目录中。

## 🔬 实验类型

### 单个实验

| 实验名称 | 描述 | 预计时间 | 重要性 |
|----------|------|----------|--------|
| `stages` | 深度展开阶段数消融 | 2-4小时 | ⭐⭐⭐⭐⭐ |
| `channels` | 隐藏通道数消融 | 3-5小时 | ⭐⭐⭐⭐ |
| `loss_components` | 损失函数组件消融 | 4-6小时 | ⭐⭐⭐⭐⭐ |
| `loss_weights` | 损失函数权重消融 | 2-3小时 | ⭐⭐⭐ |
| `batch_size` | 批次大小消融 | 1-2小时 | ⭐⭐ |
| `learning_rate` | 学习率消融 | 2-3小时 | ⭐⭐⭐ |

### 实验组

| 实验组名称 | 包含实验 | 预计时间 | 适用场景 |
|------------|----------|----------|----------|
| `architecture` | stages + channels | 5-9小时 | 网络设计验证 |
| `loss` | loss_components + loss_weights | 6-9小时 | 损失函数优化 |
| `training` | batch_size + learning_rate | 3-5小时 | 训练策略优化 |
| `all` | 所有实验 | 12-20小时 | 完整消融研究 |

## 💻 使用方法

### 基础用法

```bash
# 查看帮助
python quick_ablation.py --help

# 列出所有可用实验
python quick_ablation.py --list

# 运行单个实验
python quick_ablation.py --experiment <实验名称>

# 使用自定义配置文件
python quick_ablation.py --experiment stages --config my_config.yaml
```

### 推荐的实验顺序

**对于论文撰写，建议按以下顺序进行实验**：

1. **第一阶段**（核心组件验证）：
   ```bash
   python quick_ablation.py --experiment stages
   python quick_ablation.py --experiment loss_components
   ```

2. **第二阶段**（性能优化）：
   ```bash
   python quick_ablation.py --experiment channels
   python quick_ablation.py --experiment loss_weights
   ```

3. **第三阶段**（训练策略）：
   ```bash
   python quick_ablation.py --experiment training
   ```

4. **完整实验**（如果时间充足）：
   ```bash
   python quick_ablation.py --experiment all
   ```

### 高级用法

如果需要更精细的控制，可以直接使用完整的消融实验脚本：

```bash
# 运行完整的消融实验系统
python ablation_study.py
```

## 📊 结果分析

### 输出文件结构

```
outputs/quick_ablation_YYYYMMDD/
├── stages_ablation_results.json          # 阶段数消融结果
├── channels_ablation_results.json        # 通道数消融结果
├── loss_components_ablation_results.json # 损失组件消融结果
├── stages_ablation.png                   # 阶段数消融图表
├── channels_ablation.png                 # 通道数消融图表
└── loss_components_ablation.png          # 损失组件消融图表
```

### 关键指标解读

#### 1. 网络架构消融

**深度展开阶段数**：
- **最优范围**：通常在8-10个阶段
- **关注指标**：PSNR、SSIM、参数量、推理时间
- **论文要点**：证明深度展开的有效性和最优配置

**隐藏通道数**：
- **最优范围**：通常在64-128通道
- **关注指标**：性能-复杂度权衡
- **论文要点**：证明网络容量的合理性

#### 2. 损失函数消融

**组件贡献度**：
- **物理约束**：提升泛化能力和物理一致性
- **边缘保持**：改善图像细节和结构
- **时间稳定性**：增强时间序列一致性
- **论文要点**：证明复合损失函数的必要性

#### 3. 训练策略消融

**批次大小**：
- **最优范围**：通常在4-8
- **权衡因素**：性能 vs 内存使用 vs 训练时间

**学习率**：
- **最优范围**：通常在1e-4到5e-4
- **关注点**：收敛速度和最终性能

### 结果可视化

实验会自动生成以下类型的图表：

1. **趋势图**：显示参数变化对性能的影响
2. **对比图**：不同配置的性能对比
3. **权衡图**：性能-复杂度权衡分析
4. **雷达图**：多维度性能分析

## ❓ 常见问题

### Q1: 实验运行时间太长怎么办？

**A**: 可以通过以下方式缩短实验时间：

1. **减少训练轮数**：
   ```yaml
   # 在config/ablation_experiments.yaml中修改
   base_config:
     training:
       epochs: 20  # 从50减少到20
   ```

2. **减少数据集大小**：
   ```yaml
   base_config:
     dataset:
       train_size: 500  # 从1000减少到500
   ```

3. **运行关键实验**：
   ```bash
   # 只运行最重要的实验
   python quick_ablation.py --experiment stages
   python quick_ablation.py --experiment loss_components
   ```

### Q2: 内存不足怎么办？

**A**: 尝试以下解决方案：

1. **减少批次大小**：
   ```yaml
   base_config:
     dataset:
       batch_size: 2  # 从4减少到2
   ```

2. **减少图像尺寸**：
   ```yaml
   base_config:
     dataset:
       image_size: [128, 128]  # 从[256, 256]减少
   ```

3. **使用CPU训练**（如果GPU内存不足）：
   ```python
   # 在代码中强制使用CPU
   device = torch.device('cpu')
   ```

### Q3: 如何解读实验结果？

**A**: 关注以下几个方面：

1. **性能趋势**：参数变化对PSNR/SSIM的影响
2. **最优配置**：性能最佳的参数设置
3. **性能饱和点**：继续增加参数不再带来显著提升的点
4. **权衡分析**：性能与复杂度/时间的平衡

### Q4: 实验结果不理想怎么办？

**A**: 检查以下几个方面：

1. **数据质量**：确保训练数据质量良好
2. **超参数设置**：检查学习率、批次大小等设置
3. **模型实现**：确认模型实现正确
4. **训练稳定性**：检查训练过程是否稳定收敛

## 🔧 进阶使用

### 自定义实验配置

1. **复制配置文件**：
   ```bash
   cp config/ablation_experiments.yaml config/my_experiments.yaml
   ```

2. **修改配置参数**：
   ```yaml
   # 例如：修改阶段数测试范围
   architecture_ablation:
     stages_ablation:
       values: [4, 6, 8, 10]  # 自定义测试值
   ```

3. **使用自定义配置**：
   ```bash
   python quick_ablation.py --experiment stages --config config/my_experiments.yaml
   ```

### 添加新的评估指标

在 `quick_ablation.py` 中添加自定义指标：

```python
def _calculate_custom_metric(self, original, restored):
    """
    计算自定义评估指标
    """
    # 实现您的自定义指标
    return custom_score
```

### 并行实验执行

对于大规模实验，可以考虑并行执行：

```bash
# 在不同终端中并行运行不同实验
# 终端1
python quick_ablation.py --experiment stages

# 终端2
python quick_ablation.py --experiment channels

# 终端3
python quick_ablation.py --experiment loss_components
```

### 结果汇总和分析

使用提供的分析脚本汇总所有实验结果：

```python
# 创建结果汇总脚本
from pathlib import Path
import json
import pandas as pd

def summarize_results():
    """汇总所有实验结果"""
    results_dir = Path("outputs")
    all_results = {}
    
    for result_file in results_dir.glob("**/quick_ablation_*/"):
        # 读取和汇总结果
        pass
    
    return all_results
```

## 📝 论文撰写建议

### 实验部分结构

1. **实验设置**：
   - 数据集描述
   - 评估指标说明
   - 实验环境配置

2. **消融实验**：
   - 网络架构消融
   - 损失函数消融
   - 训练策略消融

3. **结果分析**：
   - 定量结果对比
   - 可视化结果展示
   - 关键发现总结

### 关键图表建议

1. **阶段数消融图**：展示深度展开的有效性
2. **损失组件对比图**：证明复合损失的必要性
3. **性能-复杂度权衡图**：展示模型设计的合理性
4. **消融实验汇总表**：定量对比所有配置

### 论文贡献点

基于消融实验，您可以强调以下贡献：

1. **系统性验证**：全面验证了MRA-Net各组件的有效性
2. **设计合理性**：证明了网络架构和损失函数设计的合理性
3. **最优配置**：提供了性能最佳的模型配置
4. **泛化能力**：验证了模型在不同设置下的鲁棒性

## 📞 技术支持

如果在使用过程中遇到问题，请：

1. **查看日志文件**：`logs/quick_ablation_*.log`
2. **检查配置文件**：确认参数设置正确
3. **参考示例结果**：对比预期的实验结果
4. **查看详细报告**：`outputs/reports/ablation_study_analysis.md`

---

**祝您的MRA-Net论文撰写顺利！** 🎉

通过系统性的消融实验，您将能够充分证明MRA-Net的有效性和优越性，为论文提供强有力的实验支撑。