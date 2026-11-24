# 数据目录准备说明

评测脚本通过 `--data-root` 或内置默认路径探测数据。以服务器路径 `/root/shared-nvme/data/` 为例，本地运行时传对应路径即可。

## 标准基准 (Set5 / Set14 / BSD100 / Urban100)

需要的目录结构：

```
{root}/Set5/HR_crop/X2/*.png      # HR (供 PSNR 参照)
{root}/Set5/LR_bicubic/X2/*.png   # MATLAB bicubic 退化 LR
... (每个数据集 × 每个尺度)
```

LR 生成标准：MATLAB `imresize(I, 1/s, 'bicubic')`，与主流论文一致。
若用 Python 生成，需与 MATLAB bicubic 对齐（如 `matlab_imresize` 实现），
否则 PSNR 会与已发表数值有 0.1 dB 量级漂移。

## RealSR (V3, Canon)

官方发布页：https://github.com/jwstyhss/RealSR

只需 Canon 的 Test 部分。解压后两种布局都能被脚本自动探测：

```
布局 C (V3 官方, 推荐):
{root}/Canon/Test/{2,3,4}/Canon_001_HR.png + Canon_001_LR2.png (同目录, 后缀配对)

布局 A:
{root}/Canon/Test/LR/{2,3,4}/*.png  +  {root}/Canon/Test/HR/{2,3,4}/*.png
```

每尺度 50 对。

## LPIPS

`test_save_sr.py` 先把 SR 输出存盘（`sr_outputs/SR/{dataset}/X{s}/`），再对
HR 与 SR 逐图计算 LPIPS（AlexNet 主干、RGB、不裁边界），汇总写入
`sr_outputs/lpips_summary.md`。
