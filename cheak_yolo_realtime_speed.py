# -*- coding: utf-8 -*-
"""
check_yolo_realtime_speed.py

YOLOv11n-seg(nail_seg_v1)を、撮影スクリプトと同じ2カメラ構成で
リアルタイムに動かしたとき、どれくらいのFPSが出るかを確認する。

VGG16との組み合わせは一切せず、
「YOLOセグメンテーションだけ」の速度を純粋に測ることが目的。
"""

import os
import cv2
import time
import threading
import numpy as np
from ultralytics import YOLO

# =========================================================
# 設定
# =========================================================
YOLO_WEIGHT_PATH = r"C:\Users\Owner\PycharmProjects\YOLO_nail_seg\runs\segment\runs\segment\nail_seg_v1\weights\best.pt"

CAM_NAIL = 1   # axis_satsuei_material と同じ番号に合わせる
CAM_TIP  = 0

# concat画像のサイズ（撮影スクリプトと同じ）
OUT_H = 150
OUT_W_LEFT = 150
OUT_W_RIGHT = 140

# ROI（撮影スクリプトと同じ値）
N_CX, N_CY = 499, 250
N_W0, N_H0 = 282, 409
T_CX, T_CY = 324, 550
T_W0, T_H0 = 182, 136
N_W_SCALE, N_H_SCALE = 1.3, 0.9
T_W_SCALE, T_H_SCALE = 1.7, 1.0


def crop_with_center_wh_safe(img, cx, cy, w, h):
    H, W = img.shape[:2]
    w = int(max(1, min(w, W)))
    h = int(max(1, min(h, H)))
    cx = int(max(w // 2, min(cx, W - w // 2)))
    cy = int(max(h // 2, min(cy, H - h // 2)))
    x1 = int(cx - w / 2)
    y1 = int(cy - h / 2)
    x2 = x1 + w
    y2 = y1 + h
    return img[y1:y2, x1:x2], (cx, cy, w, h)


def _resize_no_pad_center_crop(img, out_w, out_h):
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((out_h, out_w, 3), dtype=np.uint8)
    target = out_w / out_h
    cur = w / h
    if cur > target:
        new_w = int(h * target)
        x0 = (w - new_w) // 2
        cropped = img[:, x0:x0 + new_w]
    else:
        new_h = int(w / target)
        y0 = (h - new_h) // 2
        cropped = img[y0:y0 + new_h, :]
    interp = cv2.INTER_AREA if (cropped.shape[0] > out_h or cropped.shape[1] > out_w) else cv2.INTER_LINEAR
    return cv2.resize(cropped, (out_w, out_h), interpolation=interp)


def make_concat_bgr(img_nail_bgr, img_tip_bgr):
    n_w = int(N_W0 * N_W_SCALE)
    n_h = int(N_H0 * N_H_SCALE)
    t_w = int(T_W0 * T_W_SCALE)
    t_h = int(T_H0 * T_H_SCALE)

    roi_n, _ = crop_with_center_wh_safe(img_nail_bgr, N_CX, N_CY, n_w, n_h)
    roi_t, _ = crop_with_center_wh_safe(img_tip_bgr,  T_CX, T_CY, t_w, t_h)

    roi_n = _resize_no_pad_center_crop(roi_n, OUT_W_LEFT,  OUT_H)
    roi_t = _resize_no_pad_center_crop(roi_t, OUT_W_RIGHT, OUT_H)

    return cv2.hconcat([roi_n, roi_t])


class YoloSpeedTest:
    def __init__(self):
        print("=== YOLOモデル読み込み中 ===")
        self.yolo = YOLO(YOLO_WEIGHT_PATH)
        print("YOLO読み込み完了")

        # カメラ初期化
        self.cap_nail = cv2.VideoCapture(CAM_NAIL, cv2.CAP_MSMF)
        time.sleep(0.8)
        self.cap_tip = cv2.VideoCapture(CAM_TIP, cv2.CAP_MSMF)
        for cap in (self.cap_nail, self.cap_tip):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 30)
        if (not self.cap_nail.isOpened()) or (not self.cap_tip.isOpened()):
            raise RuntimeError("カメラを開けませんでした")

        # ★ カメラスレッド（撮影スクリプトと同じ方式）
        self._frame_nail = None
        self._frame_tip = None
        self._lock_nail = threading.Lock()
        self._lock_tip = threading.Lock()
        self._cam_running = True
        threading.Thread(target=self._read_nail, daemon=True).start()
        threading.Thread(target=self._read_tip, daemon=True).start()
        time.sleep(0.5)

        # FPS計測用
        self._fps_cam_ema = 0.0       # カメラ取得〜concat画像作成までのFPS
        self._fps_yolo_ema = 0.0      # YOLO推論だけのFPS（推論時間の逆数）
        self._fps_total_ema = 0.0     # ループ全体のFPS

        print("準備OK。'q'キーで終了")

    def _read_nail(self):
        while self._cam_running:
            ret, frame = self.cap_nail.read()
            if ret and frame is not None:
                with self._lock_nail:
                    self._frame_nail = frame

    def _read_tip(self):
        while self._cam_running:
            ret, frame = self.cap_tip.read()
            if ret and frame is not None:
                with self._lock_tip:
                    self._frame_tip = frame

    def run(self):
        prev_loop_time = time.perf_counter()

        try:
            while True:
                loop_start = time.perf_counter()

                with self._lock_nail:
                    base_n = self._frame_nail
                with self._lock_tip:
                    base_t = self._frame_tip

                if base_n is None or base_t is None:
                    continue

                concat_bgr = make_concat_bgr(base_n, base_t)

                # ── YOLO推論時間だけを計測 ──────────────────
                yolo_start = time.perf_counter()
                results = self.yolo.predict(
                    source=concat_bgr,
                    verbose=False,
                    device=0  # GPU使用
                )
                yolo_end = time.perf_counter()

                yolo_dt = yolo_end - yolo_start
                if yolo_dt > 0:
                    self._fps_yolo_ema = self._fps_yolo_ema * 0.9 + (1.0 / yolo_dt) * 0.1

                # ── マスク可視化（オプション：確認用） ───────
                annotated = results[0].plot()  # マスク付き画像を生成

                # ── ループ全体のFPS ─────────────────────────
                loop_end = time.perf_counter()
                loop_dt = loop_end - loop_start
                if loop_dt > 0:
                    self._fps_total_ema = self._fps_total_ema * 0.9 + (1.0 / loop_dt) * 0.1

                # ── 表示 ─────────────────────────────────────
                disp = cv2.resize(annotated, (annotated.shape[1] * 3, annotated.shape[0] * 3))
                cv2.putText(disp, f"YOLO only FPS: {self._fps_yolo_ema:.1f}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(disp, f"Loop total FPS: {self._fps_total_ema:.1f}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.imshow("YOLO speed test", disp)

                print(f"\rYOLO推論FPS: {self._fps_yolo_ema:5.1f}   "
                      f"ループ全体FPS: {self._fps_total_ema:5.1f}   "
                      f"(YOLO 1回あたり {yolo_dt*1000:.1f} ms)", end="")

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

        finally:
            self._cam_running = False
            self.cap_nail.release()
            self.cap_tip.release()
            cv2.destroyAllWindows()
            print("\n終了しました")


if __name__ == "__main__":
    tester = YoloSpeedTest()
    tester.run()