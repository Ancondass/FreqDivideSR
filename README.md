# FreqDivideSR — Evaluation & Reproduction Package

本仓库包含论文 *FreqDivideSR: A Frequency-Band Decoupled Transformer-NODE Network for Lightweight Image Super-Resolution* 的**评测复现材料**：评测脚本、评测协议、数据准备说明与环境配置。

按论文匿名评审要求，模型定义、训练代码与预训练权重**不包含在本仓库中**，获取方式见下文「权重与模型代码」。

## 1. 环境

```bash
conda create -n freqdiv_sr python=3.11 -y
conda activate freqdiv_sr
pip install -r requirements.txt
```

实验环境参考：Python 3.11，PyTorch 2.x，单张 24 GB GPU。

## 2. 数据准备

四个标准基准与 RealSR 的下载与目录结构：

```
data/
├── DIV2K/                    # 训练集 (DIV2K 800 train)，仅用于权重训练，评测不需要
├── Set5/Set14/BSD100/Urban100/
│   └── {HR_crop, LR_bicubic}/{X2,X3,X4}/*.png
└── Canon/Test/               # RealSR(V3) Canon 部分
    └── {2,3,4}/Canon_*_{HR,LR*}.png   # HR 与 LR 同目录, 靠文件名后缀配对
```

- Set5/Set14/BSD100/Urban100：使用标准 MATLAB bicubic（imresize）退化的 LR 图像，与论文口径一致。
- RealSR (V3)：官方发布页 https://github.com/jwstyhss/RealSR （Google Drive / 百度网盘），只需 Canon/Test 部分。
- LPIPS 评测使用 HR 原图与 SR 输出，不裁边界（见下文协议）。

## 3. 权重与模型代码

模型定义（`sr_model.py`、`ode_solver.py`、`transformer_global.py`、`config.py`）与预训练权重（×2/×3/×4 三个 checkpoint）未随本仓库发布。获取方式（二选一，按投稿阶段决定）：

- 匿名评审期间：随补充材料提供匿名下载链接（权重文件 `ckpt_x{2,3,4}.pth`，放到本目录 `weights/` 下）。
- 录用后：公开完整代码仓库，含训练脚本。

将权重放入 `weights/` 后，评测脚本通过 `--ckpt-x2/--ckpt-x3/--ckpt-x4` 指定，或使用默认探测路径 `weights/ckpt_x{s}.pth`。

## 4. 评测协议（与论文一致）

| 项目 | 口径 |
|---|---|
| PSNR/SSIM | MATLAB 实现，YCbCr 的 Y 通道，4 像素边界裁剪 |
| 自集成 (SE) | 8 方向几何自集成（×2/×3/×4 主表均用） |
| LPIPS | AlexNet 主干，RGB 输入，不裁边界 |
| RealSR | 真实变焦配对，Y 通道 4 px，每尺度 50 对 |

## 5. 复现论文各表的命令

> 运行前提：环境 + 数据 + 权重就绪（第 2、3 节）。

```bash
# Table 1 主表 (Set5/Set14/BSD100/Urban100, ×2/×3/×4, PSNR/SSIM, 自集成)
python test_benchmark.py --self-ensemble

# Table 6 LPIPS + SR 输出图存档 (生成 lpips_summary.md)
python test_save_sr.py --self-ensemble

# RealSR 退化鲁棒性 (×2/×3/×4 各 50 对; --save-sr 同时存 SR 输出)
python test_realsr.py --save-sr

# 计算复杂度 (MACs, 对应 Table 4)
python measure_complexity.py

# 积分步数扫描 (Table 5)
python measure_step_sweep.py

# 逐图配对统计检验 (显著性报告)
python paired_ablation_test.py
```

消融与统计检验脚本（`test_set5.py` 内含 Y 通道 PSNR/SSIM 与自集成实现，是其余评测脚本的公共依赖）。

## 6. 目录结构

```
.
├── README.md
├── requirements.txt
├── eval/
│   ├── test_set5.py            # Y 通道 PSNR/SSIM + 自集成 (公共评测组件)
│   ├── test_benchmark.py       # 四标准基准主评测
│   ├── test_save_sr.py         # LPIPS + SR 输出存档
│   ├── test_realsr.py          # RealSR 真实退化评测
│   ├── measure_complexity.py   # MACs 统计
│   ├── measure_step_sweep.py   # 积分步数扫描
│   ├── paired_ablation_test.py # 逐图配对统计检验
│   ├── ensemble.py             # 自集成工具
│   └── bic_baseline.py         # bicubic 基线
├── weights/                    # (预留) ckpt_x2/x3/x4.pth
└── scripts/
    └── prepare_data.md         # 数据目录准备说明
```
