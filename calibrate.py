# -*- coding: utf-8 -*-
"""
相机标定辅助脚本（半自动）
用法：
    python calibrate.py --image input_charger.jpg

它会做两件事：
  1) 显示图片尺寸，让你对画面有个概念
  2) 用 OpenCV 检测尺子刻度线，自动估算 像素/毫米 (ppm)
  3) 输出建议的 --ppm 参数，直接拿去跑 opencv_demo.py

如果自动检测不准（光线/角度问题），脚本会提示你手动标定方法。
"""

import argparse
import os
import sys
import cv2
import numpy as np


def auto_calibrate(gray):
    """
    尝试从图片里的尺子刻度自动估算 ppm。
    思路：尺子上有等间距的刻度线 → 用边缘检测找竖直/水平方向的周期性边缘，
    通过 FFT 或相邻边缘间距统计出「1mm 占多少像素」。
    """
    # 边缘检测
    edges = cv2.Canny(gray, 50, 150)

    # 霍夫线检测，找尺子上的刻度线（大部分是短线段，近似垂直或水平）
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180,
                            threshold=40, minLineLength=8, maxLineGap=5)

    if lines is None or len(lines) < 10:
        return None, None, "刻度线太少，无法自动标定（建议用手工标定）"

    # 兼容不同 OpenCV 版本的返回格式
    raw_lines = []
    for l in lines:
        if hasattr(l[0], '__len__') and len(l[0]) >= 4:
            raw_lines.append(list(l[0]))
        elif len(l) >= 4:
            raw_lines.append(list(l))
    lines_arr = np.array(raw_lines).reshape(-1, 4)

    # 分离近似水平线和近似垂直线
    h_lines, v_lines = [], []
    for x1, y1, x2, y2 in lines_arr:
        if abs(y2 - y1) < abs(x2 - x1):   # 更偏水平
            h_lines.append([x1, y1, x2, y2])
        else:
            v_lines.append([x1, y1, x2, y2])

    # 取线段更多的那一组（尺子长边方向）
    target = v_lines if len(v_lines) > len(h_lines) else h_lines
    if len(target) < 10:
        return None, None, "有效刻度线不足"

    # 对目标方向的线段按位置排序，计算相邻间距
    is_vert = len(v_lines) >= len(h_lines)
    positions = []
    for (x1, y1, x2, y2) in target:
        pos = (y1 + y2) / 2 if is_vert else (x1 + x2) / 2
        positions.append(pos)

    positions.sort()
    # 计算相邻间距
    gaps = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
    if not gaps:
        return None, None, "无法计算间距"

    # 过滤掉明显异常的大间距（可能是跳过了几个刻度）
    median_gap = np.median(gaps)
    valid_gaps = [g for g in gaps if 0.3 * median_gap < g < 3 * median_gap]
    if not valid_gaps:
        return None, None, "过滤后无有效间距"

    avg_gap_px = np.mean(valid_gaps)
    # 尺子最小刻度通常是 1mm，所以 avg_gap_px ≈ 1mm 的像素数
    ppm = avg_gap_px

    # 画检测结果用于调试
    color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for (x1, y1, x2, y2) in target[:50]:  # 只画前 50 条避免太密
        cv2.line(color, (x1, y1), (x2, y2), (0, 255, 0), 1)

    return ppm, color, f"自动标定成功：约 {ppm:.1f} 像素/毫米（基于 {len(valid_gaps)} 个刻度间距）"


def manual_calibrate_guide(img_path):
    """
    当自动标定失败时，输出手工标定步骤。
    """
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    print(f"\n{'='*60}")
    print("  手工标定指南")
    print(f"{'='*60}")
    print(f"  图片尺寸: {w} x {h} 像素")
    print()
    print("  步骤：")
    print("  1. 用任意看图软件打开这张照片")
    print("  2. 在尺子上找到两个清晰的刻度（比如 0 和 50mm 的标记）")
    print("  3. 记下这两个点在图片里的像素坐标 (x1,y1) 和 (x2,y2)")
    print("     （Windows 自带画图工具 → 鼠标悬停左下角就能看到坐标）")
    print("  4. 计算像素距离:")
    print("     pixel_dist = sqrt((x2-x1)^2 + (y2-y1)^2)")
    print("  5. 计算 ppm:")
    print("     ppm = pixel_dist / 实际毫米数")
    print()
    print("  例：如果 50mm 在图上量出来是 680 像素")
    print("      ppm = 680 / 50 = 13.6")
    print()
    print("  然后运行:")
    print(f"  python opencv_demo.py --image {img_path} --ppm 你的ppm值")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="相机标定：从带尺子的照片算出 ppm")
    parser.add_argument("--image", required=True, help="包含待测物+尺子的照片")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"[错误] 文件不存在: {args.image}"); sys.exit(1)

    gray = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print(f"[错误] 无法读取图片"); sys.exit(1)

    print(f"[信息] 图片尺寸: {gray.shape[1]} x {gray.shape[0]} 像素\n")

    ppm, debug_img, msg = auto_calibrate(gray)
    if ppm is not None:
        print(f"[结果] {msg}")
        print(f"\n  建议命令：")
        print(f"  python opencv_demo.py --image {args.image} --ppm {ppm:.1f}")

        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
        os.makedirs(out_dir, exist_ok=True)
        dbg_path = os.path.join(out_dir, "calibration_debug.png")
        cv2.imwrite(dbg_path, debug_img)
        print(f"\n  标定调试图（绿色=检测到的刻度线）: {dbg_path}")

        # 同时给出一个范围供微调
        print(f"\n  提示：如果测量结果偏大/偏小，可微调 ±10%：")
        print(f"       偏大 → 减小 ppm（如 {ppm * 0.9:.1f}）")
        print(f"       偏小 → 增大 ppm（如 {ppm * 1.1:.1f}）")
    else:
        print(f"[警告] {msg}")
        manual_calibrate_guide(args.image)


if __name__ == "__main__":
    main()
