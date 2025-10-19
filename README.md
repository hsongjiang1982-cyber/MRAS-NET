# MRAS-NetV2: 基于角谱传播理论的运动模糊图像复原系统

## 项目概述

MRAS-NetV2是一个专为活体微生物高速定量相位成像研究设计的深度学习图像复原系统。该项目基于角谱传播理论和物理引导的深度反卷积方法，提供了从数据生成到模型训练再到算法比较的完整解决方案。

## 系统特点

- 🔬 **物理精确建模**: 基于角谱传播理论的精确光学成像模型
- 🧠 **深度学习架构**: 创新的MRAS-Net架构，结合物理约束和深度展开算法
- 📊 **完整工作流**: 数据准备、模型训练、算法比较一体化解决方案
- ⚙️ **灵活配置**: 完整的YAML配置系统和实验管理
- 📈 **性能评估**: 多维度算法性能对比和可视化分析
- 🚀 **高效计算**: 支持GPU加速和并行处理

## 快速开始

### 环境要求

- **Python**: 3.8+ (推荐 3.9 或 3.10)
- **核心依赖**: NumPy, SciPy, Matplotlib, OpenCV, PyTorch
- **科学计算**: scikit-image, h5py, PyYAML
- **系统要求**: 8GB+ RAM, 10GB+ 存储空间

### 安装步骤

1. **克隆项目**
```bash
git clone <repository-url>
cd MRAS-NetV2
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **验证安装**
```bash
python data_preparation.py
```

## 主要功能模块

### 1. 数据准备 (`data_preparation.py`)

**核心功能：**
- 🔬 **微生物模型生成**: 球形和椭圆形微生物相位图像
- 📡 **PSF计算**: 基于角谱传播理论的点扩散函数计算
- 🌀 **模糊模拟**: 离焦、运动模糊等多种模糊类型
- 🔊 **噪声建模**: 高斯、泊松、相机噪声等真实噪声模拟
- 📊 **数据集生成**: 大规模训练/验证/测试数据集创建

**使用方法：**
```bash
python data_preparation.py
```

**输出：**
- `outputs/datasets/` - HDF5格式数据集文件
- `dataset_paths.json` - 数据集路径配置

### 2. 模型训练 (`model_generation.py`)

**核心功能：**
- 🧠 **MRAS-Net训练**: 基于角谱传播理论的深度学习模型
- 🔗 **U-Net训练**: 经典编码器-解码器架构对比模型
- 📈 **训练监控**: 实时损失和指标可视化
- 💾 **模型保存**: 自动保存最佳和最终模型
- 📊 **性能评估**: 训练过程中的模型性能评估

**使用方法：**
```bash
python model_generation.py
```

**输出：**
- `outputs/models/` - 训练好的模型文件
- 训练历史图表和日志

### 3. 算法比较 (`algorithm_comparison.py`)

**核心功能：**
- 🔍 **四算法对比**: 维纳滤波、Richardson-Lucy、U-Net、MRAS-Net
- 📏 **多维度评估**: PSNR、SSIM、MSE、MAE、边缘保持指数
- ⏱️ **性能测量**: 处理时间和内存使用分析
- 📊 **可视化报告**: 自动生成对比图表和分析报告
- 📈 **统计分析**: 双样本t检验等统计显著性分析

**使用方法：**
```bash
python algorithm_comparison.py
```

**输出：**
- `outputs/comparison/charts/` - 可视化图表
- `outputs/comparison/results/` - 详细结果数据
- `outputs/comparison/reports/` - 性能分析报告

### 4. 系统主入口 (`main.py`)

**核心功能：**
- 🎯 **统一入口**: 提供系统级的命令行接口
- 📊 **数据集生成**: 集成化的数据集创建和配置管理
- 🧪 **实验管理**: 创建、运行和管理实验配置
- 📈 **高级分析**: 调用可视化分析工具进行数据集分析
- ⚙️ **配置管理**: 灵活的YAML配置文件管理

**使用方法：**
```bash
# 生成数据集
python main.py generate --total-pairs 1000 --output-dir outputs/test

# 分析数据集
python main.py analyze --dataset-path outputs/datasets/dataset_20240101_120000.h5

# 创建实验
python main.py create-experiment --name my_exp --template advanced

# 运行实验
python main.py run-experiment --name my_exp

# 列出实验
python main.py list-experiments

# 显示系统信息
python main.py info
```

**输出：**
- 自动化的数据集生成和分析流程
- 实验配置文件和结果管理
- 集成的可视化报告生成

### 5. 数据分析 (`view_h5_data.py`)

**核心功能：**
- 📊 **数据集查看**: H5文件结构和内容展示
- 📈 **质量分析**: PSNR、SSIM、MSE、SNR等质量指标计算
- 🖼️ **样本可视化**: 图像对比和统计图表生成
- 📋 **分析报告**: 自动生成详细的数据集分析报告
- 💾 **信息导出**: JSON格式的数据集信息导出

**使用方法：**
```bash
# 完整分析
python view_h5_data.py --file outputs/datasets/train_dataset.h5 --all

# 质量分析
python view_h5_data.py --file outputs/datasets/train_dataset.h5 --quality-analysis

# 生成报告
python view_h5_data.py --file outputs/datasets/train_dataset.h5 --generate-report outputs/reports
```

**输出：**
- 控制台显示数据集信息和统计结果
- `dataset_analysis_report.md` - 详细分析报告
- 可选的样本图像保存

### 6. 快速比较 (`quick_comparison.py`)

**核心功能：**
- ⚡ **快速评估**: 小规模数据集的算法性能快速对比
- 📊 **简化报告**: 生成简洁的比较结果和图表
- 🎯 **原型验证**: 适用于算法原型的快速验证

**使用方法：**
```bash
python quick_comparison.py
```

## 完整工作流程

### 方式一：使用主入口（推荐）

**步骤1：生成数据集**
```bash
python main.py generate --total-pairs 1000 --output-dir outputs/experiment1
```

**步骤2：分析数据集**
```bash
python main.py analyze --dataset-path outputs/experiment1/datasets/dataset_*.h5
```

**步骤3：创建实验**
```bash
python main.py create-experiment --name exp1 --template advanced
```

**步骤4：运行实验**
```bash
python main.py run-experiment --name exp1
```

### 方式二：使用独立脚本

**步骤1：数据准备**
```bash
python data_preparation.py
```
选择"1. 生成完整数据集"，建议生成1000-5000对图像。

**步骤2：数据分析（可选）**
```bash
python view_h5_data.py --file outputs/datasets/train_dataset.h5 --all --generate-report outputs/reports
```
分析生成的数据集质量，确保数据符合训练要求。

**步骤3：模型训练**
```bash
python model_generation.py
```
选择"3. 训练两个模型"，使用步骤1生成的数据集。

**步骤4：算法比较**
```bash
python algorithm_comparison.py
```
使用训练好的模型进行四算法性能对比。

**步骤5：快速验证（可选）**
```bash
python quick_comparison.py
```
对小规模数据进行快速算法验证。

## 核心算法原理

### 1. 微生物相位函数

球形细胞的相位函数：
```
φ(x,y) = k₀ * Δn * h * sqrt(1 - (r/R)²)
```

### 2. 角谱传播PSF计算

```
PSF(x,y,z) = |F⁻¹{P(kₓ,kᵧ) * exp(ikᵧz)}|²
```

### 3. 深度展开Richardson-Lucy

结合传统迭代算法和深度学习的优势，实现物理约束的图像复原。

### 4. 物理一致性损失

```
L_total = L_reconstruction + λ₁*L_physics + λ₂*L_perceptual
```

## 项目结构

```
MRA-NetV2/
├── main.py                      # 系统主入口（推荐使用）
├── data_preparation.py          # 数据准备脚本
├── model_generation.py          # 模型训练脚本
├── algorithm_comparison.py      # 算法比较脚本
├── quick_comparison.py          # 快速比较脚本
├── view_h5_data.py              # 数据分析工具
├── README.md                    # 项目说明（本文件）
├── USAGE_GUIDE.md              # 详细使用指南
├── requirements.txt             # 依赖包列表
├── config/                      # 配置文件目录
│   ├── config.yaml             # 主配置文件
│   ├── model_config.yaml       # 模型配置
│   ├── training_config.yaml    # 训练配置
│   ├── comparison_config.yaml  # 比较配置
│   └── experiments/            # 实验配置目录
├── tools/                       # 核心工具模块
│   ├── mra_net_model.py        # MRAS-Net模型实现
│   ├── unet_model.py           # U-Net模型实现
│   ├── dataset_generator.py    # 数据集生成器
│   ├── blur_simulator.py       # 模糊仿真器
│   ├── psf_calculator.py       # PSF计算模块
│   ├── algorithm_evaluator.py  # 算法评估器
│   ├── report_generator.py     # 报告生成器
│   └── ...
├── outputs/                     # 输出目录
│   ├── datasets/               # 数据集
│   ├── models/                 # 训练模型
│   ├── comparison/             # 比较结果
│   ├── charts/                 # 可视化图表
│   └── reports/                # 分析报告
└── logs/                        # 日志文件
```

## 算法性能对比

| 算法 | PSNR | SSIM | 处理速度 | 适用场景 |
|------|------|------|----------|----------|
| 维纳滤波 | 中等 | 中等 | 最快 | 实时处理 |
| Richardson-Lucy | 良好 | 良好 | 慢 | 已知PSF复原 |
| U-Net | 最佳 | 良好 | 快 | 通用图像复原 |
| **MRAS-Net** | 优秀 | **最佳** | 中等 | **科学成像** |

## 评估指标说明

- **PSNR**: 峰值信噪比，衡量图像重建质量
- **SSIM**: 结构相似性指数，评估感知质量
- **MSE**: 均方误差，像素级差异度量
- **MAE**: 平均绝对误差，鲁棒性指标
- **Edge Preservation**: 边缘保持指数，细节保持能力

## 配置说明

主要配置文件 `config/config.yaml`：

```yaml
# 数据集配置
dataset:
  total_pairs: 1000          # 总图像对数量
  train_ratio: 0.7           # 训练集比例
  val_ratio: 0.15            # 验证集比例
  test_ratio: 0.15           # 测试集比例

# 图像参数
image:
  size: [512, 512]           # 图像尺寸
  pixel_size: 0.1            # 像素尺寸 (μm)
  wavelength: 0.532          # 波长 (μm)
  numerical_aperture: 1.4    # 数值孔径

# 微生物参数
microbe:
  types: ['sphere', 'ellipse']     # 微生物类型
  size_range: [1.0, 10.0]         # 尺寸范围 (μm)
  refractive_index_range: [1.35, 1.45]  # 折射率范围
  num_microbes_range: [1, 20]     # 数量范围

# 训练参数
training:
  batch_size: 16             # 批次大小
  learning_rate: 0.001       # 学习率
  epochs: 100                # 训练轮数
  device: 'cuda'             # 计算设备
```

## 性能优化

### 计算优化
- FFT加速卷积计算
- GPU并行处理
- 内存映射大数据集
- PSF缓存机制

### 训练优化
- 学习率调度
- 早停机制
- 梯度裁剪
- 混合精度训练

## 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减小batch_size
   - 减少图像尺寸
   - 使用CPU训练

2. **数据集加载失败**
   - 检查HDF5文件完整性
   - 确认文件路径正确
   - 验证数据格式

3. **训练收敛慢**
   - 调整学习率
   - 检查数据质量
   - 优化损失函数权重

4. **比较结果异常**
   - 确认模型文件存在
   - 检查测试数据格式
   - 验证评估指标计算

### 日志分析

查看 `logs/` 目录下的日志文件：
- 错误信息和堆栈跟踪
- 训练过程监控
- 性能统计信息
- 参数配置记录

## 扩展开发

### 添加新的微生物模型
```python
from tools.microbe_models import MicrobeModel

class CustomMicrobeModel(MicrobeModel):
    def generate_phase_function(self, x, y):
        # 实现自定义相位函数
        pass
```

### 添加新的损失函数
```python
class CustomLoss(nn.Module):
    def forward(self, pred, target):
        # 实现自定义损失
        pass
```

## 学术引用

如果您在研究中使用了本系统，请引用：

```bibtex
@software{mras_netv2_2025,
  title={MRAS-NetV2: Angular Spectrum Propagation-based Motion Blur Image Restoration},
  author={MRAS-Net Research Team},
  year={2025},

  version={2.0.0}
}
```

## 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件

## 技术支持
 

---

**版本**: MRAS-NetV2 v2.0.0  
**更新日期**: 2025年1月  
**作者**: MRAS-Net研究团队  

**免责声明**: 本系统专为科研用途设计。
