# -*- coding: utf-8 -*-
"""
工件尺寸测量 + 缺陷检测 Demo（机器视觉 / AOI 入门）
作者：杨杰（学习用，可直接运行，注释全中文）

依赖：
    pip install opencv-python numpy

运行方式：
    python opencv_demo.py                          # 内置合成图演示（OK + NG 各一张）
    python opencv_demo.py --image 工件.jpg          # 用自己的真实照片
    python opencv_demo.py --image 工件.jpg --ppm 15 # 指定标定系数（像素/毫米）
    python opencv_demo.py --out C:/temp/out        # 输出到指定目录（避免中文路径问题）

本 Demo 覆盖的 AOI 核心知识点：
    1) 标定（calibration）：像素 -> 真实毫米的换算
    2) 阈值分割 + 轮廓检测：把工件从背景里抠出来
    3) 尺寸测量：包围盒 + 圆孔直径
    4) 缺陷检测：用「中值滤波模板差分」找表面划痕/异色斑点
"""

import argparse
import os
import sys
import cv2
import numpy as np

PIXELS_PER_MM = 10.0


# ---------- 中文路径兼容 ----------
def imwrite_unicode(path, img):
    """cv2.imwrite 不支持中文路径，用 imencode+tofile 绕过。"""
    ext = os.path.splitext(path)[1].lower()
    enc_map = {".jpg": ".jpg", ".jpeg": ".jpg", ".png": ".png",
               ".bmp": ".bmp", ".tiff": ".tiff"}
    suffix = enc_map.get(ext, ".png")
    ok, buf = cv2.imencode(suffix, img)
    if ok:
        buf.tofile(path)
        return True
    return False


def imread_unicode(path):
    """cv2.imread 不支持中文路径，用 numpy+imdecode 绕过。"""
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def imread_gray_unicode(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)


# ---------- 合成图生成 ----------
def generate_sample(out_dir, with_defect=True, tag="ng"):
    """
    合成一张工件图：金属矩形件 + 圆孔 +（可选）划痕 + 50mm 标定尺。
    用于没有真实照片时也能跑通全流程。
    """
    w, h = 640, 480
    img = np.ones((h, w), np.uint8) * 220
    cv2.rectangle(img, (180, 120), (460, 360), 130, -1)
    cv2.circle(img, (320, 240), 35, 230, -1)
    if with_defect:
        cv2.line(img, (250, 180), (300, 230), 60, 3)
    bar_len = int(50 * PIXELS_PER_MM)
    cv2.rectangle(img, (40, 420), (40 + bar_len, 440), 80, -1)
    cv2.putText(img, "50mm ref", (40, 410),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, 60, 1)
    path = os.path.join(out_dir, f"sample_{tag}.png")
    imwrite_unicode(path, img)
    return path, img


# ---------- 尺寸测量 ----------
def measure_dimensions(gray):
    """
    阈值分割 -> 轮廓检测 -> 包围盒尺寸 -> ROI内圆孔检测。
    返回 dict 或 None。
    """
    # 自适应阈值比固定阈值更鲁棒（适应不同光照）
    if gray.mean() > 128:
        _, th = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    else:
        th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 51, 10)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    # 在工件 ROI 内找圆孔（亮区域）
    roi = gray[y:y + h, x:x + w]
    _, th2 = cv2.threshold(roi, 180, 255, cv2.THRESH_BINARY)
    holes, _ = cv2.findContours(th2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hole_diam_px = 0.0
    for hh in holes:
        if cv2.contourArea(hh) > 100:
            (_, _), r = cv2.minEnclosingCircle(hh)
            hole_diam_px = max(hole_diam_px, 2 * r)

    return {"bbox": (x, y, w, h), "w_px": w, "h_px": h,
            "hole_diam_px": hole_diam_px, "contour": c}


# ---------- 缺陷检测 ----------
def detect_defects(gray, bbox):
    """
    中值滤波模板差分法（工业 AOI 常用）：
    正常表面平滑 → 中值滤波后几乎不变；
    划痕/脏点等缺陷是突变 → 滤波前后差异大 → 被检出。
    """
    x, y, w, h = bbox
    roi = gray[y:y + h, x:x + w]
    smooth = cv2.medianBlur(roi, 9)
    diff = cv2.absdiff(roi, smooth)
    _, dth = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    dth = cv2.morphologyEx(dth, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(dth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    defects = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 30]
    return [(x + dx, y + dy, dw, dh) for (dx, dy, dw, dh) in defects]


# ---------- 主处理流程 ----------
def process(img, name, ppm, out_dir):
    """测量 + 缺陷检测 + 画结果 + 存盘。"""
    color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img.copy()
    if len(color.shape) == 3 and color.shape[2] == 3:
        gray_for_proc = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    else:
        gray_for_proc = img

    res = measure_dimensions(gray_for_proc)
    if res is None:
        print(f"[{name}] 未检测到工件，跳过")
        return

    x, y, w, h = res["bbox"]
    cv2.rectangle(color, (x, y), (x + w, y + h), (0, 200, 0), 2)
    w_mm = res["w_px"] / ppm
    h_mm = res["h_px"] / ppm
    cv2.putText(color, f"W={w_mm:.1f}mm H={h_mm:.1f}mm",
                (x, max(y - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
    if res["hole_diam_px"] > 0:
        d_mm = res["hole_diam_px"] / ppm
        cv2.putText(color, f"hole={d_mm:.1f}mm",
                    (x, min(y + h + 25, color.shape[0] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    defects = detect_defects(gray_for_proc, res["bbox"])
    for (dx, dy, dw, dh) in defects:
        cv2.rectangle(color, (dx, dy), (dx + dw, dy + dh), (0, 0, 255), 2)

    verdict = "NG（有缺陷）" if defects else "OK（合格）"
    cv2.putText(color, f"判定:{verdict}  缺陷数:{len(defects)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    out = os.path.join(out_dir, f"result_{name}.png")
    imwrite_unicode(out, color)
    print(f"[{name}] 尺寸 W={w_mm:.1f}mm H={h_mm:.1f}mm  "
          f"圆孔≈{res['hole_diam_px']/ppm:.1f}mm  "
          f"缺陷数={len(defects)}  -> {verdict}")
    print(f"       结果图: {out}")
    return out


def main():
    parser = argparse.ArgumentParser(description="工件尺寸测量 + 缺陷检测 Demo")
    parser.add_argument("--image", default=None, help="工件照片路径")
    parser.add_argument("--ppm", type=float, default=PIXELS_PER_MM,
                        help="标定系数 像素/毫米（默认 10.0）")
    parser.add_argument("--out", default=None,
                        help="输出目录（默认与脚本同级的 output/）")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = args.out or os.path.join(here, "output")
    os.makedirs(out_dir, exist_ok=True)

    if args.image and os.path.exists(args.image):
        img = imread_gray_unicode(args.image)
        if img is None:
            print("[错误] 图片读取失败，检查路径"); sys.exit(1)
        name = os.path.splitext(os.path.basename(args.image))[0]
        process(img, name, args.ppm, out_dir)
    else:
        print("[信息] 未提供 --image，使用内置合成图演示（一张合格、一张带缺陷）\n")
        _, ok = generate_sample(out_dir, with_defect=False, tag="ok")
        process(ok, "ok", args.ppm, out_dir)
        _, ng = generate_sample(out_dir, with_defect=True, tag="ng")
        process(ng, "ng", args.ppm, out_dir)


if __name__ == "__main__":
    main()
