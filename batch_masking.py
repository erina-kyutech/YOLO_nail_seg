# -*- coding: utf-8 -*-
"""
学習済みYOLOv11n-segで18万枚を一括推論し、
マスク済み画像を2種類保存するスクリプト。

出力:
  masked_nail_only/ifukuN/360deg/*.png      ← 爪のみマスク
  masked_nail_and_tip/ifukuN/360deg/*.png   ← 爪+指先マスク

元画像は一切変更しない。
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# =========================================================
# 設定
# =========================================================

# 学習済みモデルのパス
BEST_PT = Path(
    r"C:\Users\Owner\Downloads\runs\segment\runs\segment\nail_seg_v1\weights\best.pt"
)

# 元画像のルート
SRC_ROOT = Path(r"C:\Users\Owner\PycharmProjects\datas\record0-10xyz")

# 出力先ルート
DST_NAIL_ONLY    = Path(r"C:\Users\Owner\PycharmProjects\datas\masked_nail_only")
DST_NAIL_AND_TIP = Path(r"C:\Users\Owner\PycharmProjects\datas\masked_nail_and_tip")

# ifuku番号の範囲
IFUKU_START = 1
IFUKU_END   = 180

# 推論設定
CONF_THRESHOLD = 0.25
IMG_SIZE       = 320
BATCH_SIZE     = 32   # 一度にYOLOに渡す画像枚数(VRAMに余裕があれば増やせる)
DEVICE         = 0    # GPU番号

# クラスID
CLASS_NAIL      = 1
CLASS_FINGERTIP = 0


# =========================================================
# ユーティリティ
# =========================================================
def get_combined_mask(result, H: int, W: int) -> tuple[np.ndarray, np.ndarray]:
    """
    YOLO推論結果から nail_mask と combined_mask を返す。
    どちらも (H, W) の bool配列。
    """
    nail_mask     = np.zeros((H, W), dtype=bool)
    combined_mask = np.zeros((H, W), dtype=bool)

    if result.masks is None:
        return nail_mask, combined_mask

    for mask_tensor, box in zip(result.masks.data, result.boxes):
        class_id = int(box.cls.item())
        mask_np  = mask_tensor.cpu().numpy()
        mask_resized = cv2.resize(mask_np, (W, H), interpolation=cv2.INTER_NEAREST)
        mask_bool = mask_resized > 0.5

        combined_mask |= mask_bool
        if class_id == CLASS_NAIL:
            nail_mask |= mask_bool

    return nail_mask, combined_mask


def apply_mask(img_bgr: np.ndarray, mask_bool: np.ndarray) -> np.ndarray:
    """マスク領域以外を黒(0)にしたBGR画像を返す"""
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

    # 全画像パスを収集
    print("画像パスを収集中...")
    all_img_paths = []
    for ifuku_id in range(IFUKU_START, IFUKU_END + 1):
        img_dir = SRC_ROOT / f"ifuku{ifuku_id}" / "360deg"
        if not img_dir.exists():
            print(f"[WARN] フォルダが見つかりません(スキップ): {img_dir}")
            continue
        paths = sorted(img_dir.glob("*.png"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
        all_img_paths.extend(paths)

    total = len(all_img_paths)
    print(f"合計画像数: {total} 枚")

    if total == 0:
        raise RuntimeError("処理対象の画像が見つかりません。SRC_ROOTを確認してください。")

    # バッチ処理
    processed  = 0
    skipped    = 0
    no_mask    = 0

    for batch_start in range(0, total, BATCH_SIZE):
        batch_paths = all_img_paths[batch_start: batch_start + BATCH_SIZE]
        batch_strs  = [str(p) for p in batch_paths]

        # YOLO推論(バッチ)
        results = model(
            batch_strs,
            imgsz=IMG_SIZE,
            conf=CONF_THRESHOLD,
            device=DEVICE,
            verbose=False,
        )

        for img_path, result in zip(batch_paths, results):
            # 出力パスを構築 (元のフォルダ構成をそのまま維持)
            # img_path: .../record0-10xyz/ifukuN/360deg/M.png
            try:
                rel = img_path.relative_to(SRC_ROOT)  # ifukuN/360deg/M.png
            except ValueError:
                print(f"[WARN] 相対パス解決失敗(スキップ): {img_path}")
                skipped += 1
                continue

            dst_nail     = DST_NAIL_ONLY    / rel
            dst_combined = DST_NAIL_AND_TIP / rel

            # 元画像読み込み
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                print(f"[WARN] 読み込み失敗(スキップ): {img_path}")
                skipped += 1
                continue

            H, W = bgr.shape[:2]

            # マスク取得
            nail_mask, combined_mask = get_combined_mask(result, H, W)

            # マスクが全く検出されなかった場合は元画像をそのままコピー
            # (学習データとしての整合性を保つため)
            if not nail_mask.any() and not combined_mask.any():
                no_mask += 1
                save_image(dst_nail,     bgr)
                save_image(dst_combined, bgr)
            else:
                save_image(dst_nail,     apply_mask(bgr, nail_mask))
                save_image(dst_combined, apply_mask(bgr, combined_mask))

            processed += 1

        # 進捗表示
        done = batch_start + len(batch_paths)
        print(f"  進捗: {done}/{total} ({100*done/total:.1f}%) "
              f"| スキップ: {skipped} | マスク未検出: {no_mask}")

    print("\n=== 完了 ===")
    print(f"処理完了 : {processed} 枚")
    print(f"スキップ : {skipped} 枚")
    print(f"マスク未検出(元画像コピー): {no_mask} 枚")
    print(f"爪のみマスク保存先    : {DST_NAIL_ONLY}")
    print(f"爪+指先マスク保存先   : {DST_NAIL_AND_TIP}")


if __name__ == "__main__":
    main()
