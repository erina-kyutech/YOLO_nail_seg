# -*- coding: utf-8 -*-
"""
学習済みYOLOv11n-segで18万枚を一括推論し、
マスク済み画像を3種類保存するスクリプト(v2: finger_tip_only追加)。

出力:
  masked_nail_only/         ← 爪のみマスク(既存)
  masked_nail_and_tip/      ← 爪+指先マスク(既存)
  masked_finger_tip_only/   ← 指先のみマスク(新規)

既存の2フォルダは再生成しない(既にある場合はスキップ)。
finger_tip_onlyだけ新規生成する。
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# =========================================================
# 設定
# =========================================================
BEST_PT = Path(
    r"C:\Users\Owner\Downloads\runs\segment\runs\segment\nail_seg_v1\weights\best.pt"
)

SRC_ROOT         = Path(r"C:\Users\Owner\PycharmProjects\datas\record0-10xyz")
DST_NAIL_ONLY    = Path(r"C:\Users\Owner\PycharmProjects\datas\masked_nail_only")
DST_NAIL_AND_TIP = Path(r"C:\Users\Owner\PycharmProjects\datas\masked_nail_and_tip")
DST_TIP_ONLY     = Path(r"C:\Users\Owner\PycharmProjects\datas\masked_finger_tip_only")

IFUKU_START    = 1
IFUKU_END      = 180
CONF_THRESHOLD = 0.25
IMG_SIZE       = 320
BATCH_SIZE     = 32
DEVICE         = 0

CLASS_NAIL      = 1
CLASS_FINGERTIP = 0

# 既存フォルダを再生成するかどうか
REGENERATE_NAIL_ONLY    = False  # 既にあるので再生成しない
REGENERATE_NAIL_AND_TIP = False  # 既にあるので再生成しない
REGENERATE_TIP_ONLY     = True   # 新規生成する


# =========================================================
# ユーティリティ
# =========================================================
def get_masks(result, H: int, W: int):
    nail_mask = np.zeros((H, W), dtype=bool)
    tip_mask  = np.zeros((H, W), dtype=bool)

    if result.masks is None:
        return nail_mask, tip_mask

    for mask_tensor, box in zip(result.masks.data, result.boxes):
        class_id     = int(box.cls.item())
        mask_np      = mask_tensor.cpu().numpy()
        mask_resized = cv2.resize(mask_np, (W, H), interpolation=cv2.INTER_NEAREST)
        mask_bool    = mask_resized > 0.5

        if class_id == CLASS_NAIL:
            nail_mask |= mask_bool
        elif class_id == CLASS_FINGERTIP:
            tip_mask  |= mask_bool

    return nail_mask, tip_mask


def apply_mask(img_bgr: np.ndarray, mask_bool: np.ndarray) -> np.ndarray:
    result = np.zeros_like(img_bgr)
    result[mask_bool] = img_bgr[mask_bool]
    return result


def save_image(path: Path, img: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


# =========================================================
# メイン
# =========================================================
def main():
    print(f"モデル読み込み: {BEST_PT}")
    if not BEST_PT.exists():
        raise FileNotFoundError(f"best.pt が見つかりません: {BEST_PT}")

    model = YOLO(str(BEST_PT))

    print("画像パスを収集中...")
    all_img_paths = []
    for ifuku_id in range(IFUKU_START, IFUKU_END + 1):
        img_dir = SRC_ROOT / f"ifuku{ifuku_id}" / "360deg"
        if not img_dir.exists():
            print(f"[WARN] フォルダが見つかりません(スキップ): {img_dir}")
            continue
        paths = sorted(img_dir.glob("*.png"),
                       key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
        all_img_paths.extend(paths)

    total = len(all_img_paths)
    print(f"合計画像数: {total} 枚")

    processed = 0
    skipped   = 0
    no_mask   = 0

    for batch_start in range(0, total, BATCH_SIZE):
        batch_paths = all_img_paths[batch_start: batch_start + BATCH_SIZE]
        batch_strs  = [str(p) for p in batch_paths]

        results = model(
            batch_strs,
            imgsz=IMG_SIZE,
            conf=CONF_THRESHOLD,
            device=DEVICE,
            verbose=False,
        )

        for img_path, result in zip(batch_paths, results):
            try:
                rel = img_path.relative_to(SRC_ROOT)
            except ValueError:
                skipped += 1
                continue

            bgr = cv2.imread(str(img_path))
            if bgr is None:
                skipped += 1
                continue

            H, W = bgr.shape[:2]
            nail_mask, tip_mask = get_masks(result, H, W)
            combined_mask = nail_mask | tip_mask

            # マスク未検出の場合は元画像をコピー
            if not nail_mask.any() and not tip_mask.any():
                no_mask += 1
                if REGENERATE_NAIL_ONLY:
                    save_image(DST_NAIL_ONLY / rel, bgr)
                if REGENERATE_NAIL_AND_TIP:
                    save_image(DST_NAIL_AND_TIP / rel, bgr)
                if REGENERATE_TIP_ONLY:
                    save_image(DST_TIP_ONLY / rel, bgr)
            else:
                if REGENERATE_NAIL_ONLY:
                    save_image(DST_NAIL_ONLY / rel,    apply_mask(bgr, nail_mask))
                if REGENERATE_NAIL_AND_TIP:
                    save_image(DST_NAIL_AND_TIP / rel, apply_mask(bgr, combined_mask))
                if REGENERATE_TIP_ONLY:
                    save_image(DST_TIP_ONLY / rel,     apply_mask(bgr, tip_mask))

            processed += 1

        done = batch_start + len(batch_paths)
        print(f"  進捗: {done}/{total} ({100*done/total:.1f}%)"
              f" | スキップ: {skipped} | マスク未検出: {no_mask}")

    print("\n=== 完了 ===")
    print(f"処理完了       : {processed} 枚")
    print(f"スキップ       : {skipped} 枚")
    print(f"マスク未検出   : {no_mask} 枚")
    if REGENERATE_TIP_ONLY:
        print(f"指先のみ保存先 : {DST_TIP_ONLY}")


if __name__ == "__main__":
    main()
