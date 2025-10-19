#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json

# 读取最新对比结果
with open('outputs/comparison_20251003_153830/results/detailed_comparison_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

stats = data['statistics']['metrics']
print("=== 最新对比结果 ===")
print("PSNR平均值:")
for alg in stats.keys():
    psnr = stats[alg]['psnr']['mean']
    print(f"{alg}: {psnr:.3f} dB")

print("\nSSIM平均值:")
for alg in stats.keys():
    ssim = stats[alg]['ssim']['mean']
    print(f"{alg}: {ssim:.3f}")

# 计算与U-Net的差距
if 'Unet' in stats and 'MRAS-Net' in stats:
    unet_psnr = stats['Unet']['psnr']['mean']
    mras_psnr = stats['MRAS-Net']['psnr']['mean']
    gap = unet_psnr - mras_psnr
    print(f"\nMRAS-Net与U-Net的PSNR差距: {gap:.3f} dB")
    print(f"需要提升: {gap:.3f} dB才能超越U-Net")
