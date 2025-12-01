# DISTS、PI、FID指标计算工具使用说明

## 简介

这个独立工具用于计算图像超分辨率的三个重要指标：
- **DISTS** (Deep Image Structure and Texture Similarity)
- **PI** (Perceptual Index = 0.5 × ((10 - MA) + NIQE))
- **FID** (Fréchet Inception Distance)

## 特点

- ✅ 不修改原有代码，完全独立运行
- ✅ 自动匹配SR和GT图像文件
- ✅ 支持自定义文件命名模式
- ✅ 支持批量计算和结果保存
- ✅ 显示每个指标的均值和标准差
- ✅ GPU加速计算

## 快速开始

### 基本用法

```bash
python meters/calculate_dists_pi_fid.py \
  --sr_path results/xxx/visualization/dataset_name \
  --gt_path path/to/gt/images \
  --metrics dists pi fid
```

### 参数说明

| 参数 | 必需 | 说明 | 默认值 |
|------|------|------|--------|
| `--sr_path` | ✅ | 超分结果目录 | - |
| `--gt_path` | ✅ | Ground Truth图像目录 | - |
| `--metrics` | ❌ | 要计算的指标列表 | `dists pi fid` |
| `--pattern` | ❌ | 文件匹配模式 | `*.png` |
| `--suffix` | ❌ | SR文件名后缀（自动去除以匹配GT） | `''` |
| `--save_json` | ❌ | 保存结果到JSON文件 | 不保存 |
| `--verbose` | ❌ | 打印每张图片的结果 | False |

## 指标说明

### 1. DISTS (Deep Image Structure and Texture Similarity)

- **类型**: 全参考 (Full-Reference)
- **范围**: [0, ∞)，越小越好
- **实现**: 使用**pyiqa**库（与原UPSR项目一致）
- **特点**: 基于深度学习的感知质量指标，同时评估结构和纹理相似度
- **pyiqa指标名**: `dists`

### 2. PI (Perceptual Index)

- **类型**: 无参考 (No-Reference)
- **范围**: [0, ∞)，越小越好
- **公式**: `PI = 0.5 × ((10 - MA) + NIQE)`
- **实现**: 使用**pyiqa**库（与原UPSR项目一致）
- **组成**:
  - **NIQE**: 自然图像质量评估 (pyiqa指标名: `niqe`)
  - **MA**: 美学评分，使用`musiq-ava`近似 (输出0-1，×10转为0-10)
- **特点**: 结合自然度和美学，常用于GAN评估
- **注意**: pyiqa不直接支持MA，这里使用`musiq-ava`作为近似

### 3. FID (Fréchet Inception Distance)

- **类型**: 分布距离
- **范围**: [0, ∞)，越小越好
- **实现**: 使用**torchvision Inception V3**（pyiqa不支持FID）
- **特点**: 测量SR和GT在Inception特征空间的分布距离
- **注意**: 
  - 需要足够数量的图像（建议≥50张）才有统计意义
  - pyiqa不支持FID指标，本工具使用标准Inception V3实现
  - 与原UPSR项目一致（原项目也跳过FID在线验证）

## 使用示例

### 示例1: 计算所有指标

```bash
python meters/calculate_dists_pi_fid.py \
  --sr_path results/000_UPSR_RealSR_x4/visualization/LSDIR-test \
  --gt_path D:\DataSets\LSDIR-test\HR\val \
  --metrics dists pi fid \
  --save_json meters/results_all.json
```

### 示例2: 只计算PI指标

```bash
python meters/calculate_dists_pi_fid.py \
  --sr_path results/xxx/visualization/dataset \
  --gt_path D:\DataSets\xxx\HR \
  --metrics pi
```

输出会包含：
- PI (总指标)
- NIQE (自然度)
- MA (美学评分)

### 示例3: 处理特殊命名的文件

如果SR文件名是 `0000001_000_UPSR_RealSR_x4.png`，GT文件名是 `0000001.png`：

```bash
# 方法1: 使用自动检测（推荐）
python meters/calculate_dists_pi_fid.py \
  --sr_path results/xxx/visualization/dataset \
  --gt_path D:\DataSets\xxx\HR \
  --metrics dists pi fid

# 方法2: 手动指定后缀
python meters/calculate_dists_pi_fid.py \
  --sr_path results/xxx/visualization/dataset \
  --gt_path D:\DataSets\xxx\HR \
  --metrics dists pi fid \
  --suffix _000_UPSR_RealSR_x4
```

### 示例4: 使用JPG格式

```bash
python meters/calculate_dists_pi_fid.py \
  --sr_path results/xxx/visualization/dataset \
  --gt_path D:\DataSets\xxx\HR \
  --metrics dists pi fid \
  --pattern "*.jpg"
```

## 输出说明

### 终端输出

```
================================================================================
Calculate DISTS, PI, and FID Metrics
================================================================================
SR Path: results/xxx/visualization/dataset
GT Path: D:\DataSets\xxx\HR
Metrics: ['dists', 'pi', 'fid']
================================================================================

[Step 1] Finding files...
Found 250 matched SR-GT pairs

[Step 2] Loading images...
Successfully loaded 250 image pairs

Using device: cuda

[DISTS] Calculating...
DISTS: 100%|████████████████| 250/250 [00:30<00:00,  8.21it/s]

DISTS: 0.1234 ± 0.0156

[PI] Calculating NIQE and MA...
PI (NIQE+MA): 100%|████████████████| 250/250 [01:20<00:00,  3.12it/s]

PI: 3.4567 ± 0.2345
  NIQE: 4.5678 ± 0.3456
  MA: 6.7890 ± 0.4567

[FID] Calculating (this may take a while)...
Extracting SR features...
Extracting features: 100%|████████████████| 8/8 [00:45<00:00,  5.67s/it]
Extracting GT features...
Extracting features: 100%|████████████████| 8/8 [00:45<00:00,  5.67s/it]

FID: 45.6789

================================================================================
Summary:
================================================================================
DISTS ↓: 0.1234
PI    ↓: 3.4567
NIQE  ↓: 4.5678
MA    ↑: 6.7890
FID   ↓: 45.6789
================================================================================
```

### JSON输出格式

```json
{
  "metrics": {
    "dists": {
      "per_image": [0.1234, 0.1345, ...],
      "mean": 0.1234,
      "std": 0.0156
    },
    "pi": {
      "per_image": [3.4567, 3.5678, ...],
      "mean": 3.4567,
      "std": 0.2345
    },
    "niqe": {
      "per_image": [4.5678, 4.6789, ...],
      "mean": 4.5678,
      "std": 0.3456
    },
    "ma": {
      "per_image": [6.7890, 6.8901, ...],
      "mean": 6.7890,
      "std": 0.4567
    },
    "fid": 45.6789
  },
  "config": {
    "sr_path": "results/xxx/visualization/dataset",
    "gt_path": "D:\\DataSets\\xxx\\HR",
    "num_images": 250,
    "metrics_calculated": ["dists", "pi", "fid"]
  }
}
```

## 常见问题

### 1. 找不到匹配的GT文件

**问题**: `Warning: No GT found for xxx.png`

**解决方案**:
- 检查GT路径是否正确
- 检查文件扩展名是否匹配（png/jpg/jpeg等）
- 使用 `--suffix` 参数指定SR文件名后缀
- 查看SR和GT的文件命名规则

### 2. 显存不足

**问题**: CUDA out of memory

**解决方案**:
- FID计算时降低batch_size（修改代码中的 `batch_size=32` → `batch_size=16`）
- 使用CPU计算（自动回退）
- 减少图像数量

### 3. FID值异常

**问题**: FID值过大或为负数

**原因**: 图像数量太少（<50张）导致统计不稳定

**解决方案**: 增加图像数量至至少50张

### 4. 计算速度慢

**优化建议**:
- 确保使用GPU（自动检测CUDA）
- DISTS: ~8 img/s
- PI: ~3 img/s（NIQE较慢）
- FID: 取决于图像总数，约需2-5分钟

## 技术细节

### DISTS计算（使用pyiqa）
- **库**: `pyiqa.create_metric('dists')`
- **输入**: RGB tensor, 范围[0,1], shape=[1,3,H,W]
- **输出**: 标量，越小越好
- **实现**: 基于VGG特征的结构和纹理相似度
- **与原UPSR一致**: ✅ 完全相同

### PI计算（使用pyiqa）
- **NIQE**: `pyiqa.create_metric('niqe')`
  - 输入: RGB tensor, 范围[0,1]
  - 输出: 标量，越小越好（通常0-100+）
- **MA**: `pyiqa.create_metric('musiq-ava')`
  - 输入: RGB tensor, 范围[0,1]
  - 输出: 标量范围[0,1]，需要×10转为[0,10]
  - 注意: pyiqa不直接支持MA，使用musiq-ava近似
- **PI公式**: `PI = 0.5 × ((10 - MA) + NIQE)`
- **与原UPSR一致**: ✅ 使用相同的pyiqa指标

### FID计算（自定义实现）
- **原因**: pyiqa不支持FID指标
- **实现**: 
  - 模型: torchvision预训练Inception V3
  - 输入: RGB图像resize到299×299，归一化到[-1,1]
  - 特征: 提取2048维向量（移除最后的FC层）
  - 距离: Fréchet距离 = ||μ_sr - μ_gt||² + Tr(Σ_sr + Σ_gt - 2√(Σ_sr·Σ_gt))
- **与原UPSR一致**: ✅ 原项目也跳过FID在线验证（需要文件夹路径）
- **标准实现**: 基于论文标准算法，与clean-fid等库一致

## 依赖环境

```
torch
torchvision
pyiqa          # 关键: DISTS和PI指标依赖pyiqa库
numpy
scipy          # FID计算需要
opencv-python
tqdm
```

**安装pyiqa**（如果未安装）:
```bash
pip install pyiqa
```

## 与原UPSR项目的一致性

本工具与原UPSR项目保持完全一致：

✅ **DISTS**: 使用`pyiqa.create_metric('dists')`，与原项目完全相同  
✅ **PI/NIQE**: 使用`pyiqa.create_metric('niqe')`，与原项目完全相同  
✅ **MA**: 使用`pyiqa.create_metric('musiq-ava')`作为近似  
✅ **FID**: 原项目也不支持在线FID计算，本工具提供离线实现  

**验证方法**:
```python
# 原UPSR项目的指标创建方式 (basicsr/models/upsr_real_model.py)
import pyiqa
dists_metric = pyiqa.create_metric('dists', device='cuda')
niqe_metric = pyiqa.create_metric('niqe', device='cuda')

# 本工具使用完全相同的方式
# meters/calculate_dists_pi_fid.py
dists_metric = pyiqa.create_metric('dists', device=device)
niqe_metric = pyiqa.create_metric('niqe', device=device)
```

## 性能参考

- **250张512×512图像**:
  - DISTS: ~30秒
  - PI: ~80秒
  - FID: ~90秒
  - 总计: ~3.5分钟

## 引用

如果使用这些指标，请引用相应论文：

**DISTS**:
```
@article{ding2020image,
  title={Image quality assessment: Unifying structure and texture similarity},
  author={Ding, Keyan and Ma, Kede and Wang, Shiqi and Simoncelli, Eero P},
  journal={IEEE TPAMI},
  year={2020}
}
```

**PI**:
```
@inproceedings{blau2018perception,
  title={The perception-distortion tradeoff},
  author={Blau, Yochai and Michaeli, Tomer},
  booktitle={CVPR},
  year={2018}
}
```

**FID**:
```
@inproceedings{heusel2017gans,
  title={Gans trained by a two time-scale update rule converge to a local nash equilibrium},
  author={Heusel, Martin and Ramsauer, Hubert and Unterthiner, Thomas and Nessler, Bernhard and Hochreiter, Sepp},
  booktitle={NeurIPS},
  year={2017}
}
```
