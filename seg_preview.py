# -*- coding: utf-8 -*-
"""
学習済みYOLOv11n-segで推論し、セグメンテーション結果を目視確認するスクリプト。

val画像9枚に対して以下の4種類の画像を並べて表示・保存する:
  1. 元画像
  2. マスクオーバーレイ (finger_tip=赤, nail=青)
  3. 爪だけマスク(nail領域のみ背景0)
  4. 爪+指先マスク(両方の領域のみ背景0)
"""

from __future__ import annotations
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from ultralytics import YOLO

# =========================================================
# 設定 (パスだけ確認してください)
# =========================================================

# 学習済みモデルのパス
BEST_PT = Path(
    r"C:\Users\Owner\Downloads\runs\segment\runs\segment\nail_seg_v1\weights\best.pt"
)

# 確認する画像フォルダ (val画像)
VAL_IMG_DIR = Path(
    r"C:\Users\Owner\PycharmProjects\Tactile_sensors_20230929"
    r"\yolo_seg_dataset\images\val"
)

# 結果保存先
SAVE_DIR = Path(
    r"C:\Users\Owner\PycharmProjects\YOLO_nail_seg\seg_preview"
)

# クラス設定
CLASS_NAMES = {0: "finger_tip", 1: "nail"}
CLASS_COLORS = {
    0: (255, 80, 80),    # finger_tip: 赤
    1: (80, 120, 255),   # nail: 青
}

# 推論設定
CONF_THRESHOLD = 0.25   # 信頼度閾値
IMG_SIZE = 320


# =========================================================
# メイン
# =========================================================
def apply_mask_to_image(img_rgb: np.ndarray, mask_bool: np.ndarray) -> np.ndarray:
    """マスク領域以外を黒(0)にした画像を返す"""
    result = np.zeros_like(img_rgb)
    result[mask_bool] = img_rgb[mask_bool]
    return result


def overlay_masks(img_rgb: np.ndarray, masks: list, class_ids: list, alpha: float = 0.45) -> np.ndarray:
    """マスクをカラーオーバーレイで重ねた画像を返す"""
    overlay = img_rgb.copy().astype(np.float32)
    for mask, class_id in zip(masks, class_ids):
        color = CLASS_COLORS.get(class_id, (255, 255, 255))
        colored = np.zeros_like(img_rgb, dtype=np.float32)
        colored[mask] = color
        overlay = overlay * (1 - alpha * mask[:, :, None]) + colored * alpha * mask[:, :, None]
    return np.clip(overlay, 0, 255).astype(np.uint8)


def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"モデル読み込み: {BEST_PT}")
    if not BEST_PT.exists():
        raise FileNotFoundError(f"best.pt が見つかりません: {BEST_PT}")

    model = YOLO(str(BEST_PT))

    img_paths = sorted(VAL_IMG_DIR.glob("*.png")) + sorted(VAL_IMG_DIR.glob("*.jpg"))
    if not img_paths:
        raise FileNotFoundError(f"画像が見つかりません: {VAL_IMG_DIR}")

    print(f"確認する画像数: {len(img_paths)}")

    for img_path in img_paths:
        print(f"処理中: {img_path.name}")

        # 元画像読み込み
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"[WARN] 読み込み失敗: {img_path}")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]

        # 推論
        results = model(str(img_path), imgsz=IMG_SIZE, conf=CONF_THRESHOLD, verbose=False)
        result = results[0]

        # マスク取得
        masks_all = []
        class_ids_all = []
        nail_mask = np.zeros((H, W), dtype=bool)
        finger_tip_mask = np.zeros((H, W), dtype=bool)

        if result.masks is not None:
            for i, (mask_tensor, box) in enumerate(zip(result.masks.data, result.boxes)):
                class_id = int(box.cls.item())
                # マスクを元画像サイズにリサイズ
                mask_np = mask_tensor.cpu().numpy()
                mask_resized = cv2.resize(mask_np, (W, H), interpolation=cv2.INTER_NEAREST)
                mask_bool = mask_resized > 0.5

                masks_all.append(mask_bool)
                class_ids_all.append(class_id)

                if class_id == 1:  # nail
                    nail_mask |= mask_bool
                elif class_id == 0:  # finger_tip
                    finger_tip_mask |= mask_bool

        combined_mask = nail_mask | finger_tip_mask

        # 4種類の画像を生成
        img_overlay = overlay_masks(rgb, masks_all, class_ids_all)
        img_nail_only = apply_mask_to_image(rgb, nail_mask)
        img_combined = apply_mask_to_image(rgb, combined_mask)

        # 4枚並べてプロット
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle(img_path.name, fontsize=13)

        axes[0].imshow(rgb)
        axes[0].set_title("元画像")
        axes[0].axis("off")

        axes[1].imshow(img_overlay)
        axes[1].set_title("マスクオーバーレイ")
        axes[1].axis("off")
        # 凡例
        patches = [
            mpatches.Patch(color=[c/255 for c in CLASS_COLORS[0]], label="finger_tip"),
            mpatches.Patch(color=[c/255 for c in CLASS_COLORS[1]], label="nail"),
        ]
        axes[1].legend(handles=patches, loc="upper right", fontsize=8)

        axes[2].imshow(img_nail_only)
        axes[2].set_title("爪のみマスク(背景0)")
        axes[2].axis("off")

        axes[3].imshow(img_combined)
        axes[3].set_title("爪+指先マスク(背景0)")
        axes[3].axis("off")

        plt.tight_layout()

        save_path = SAVE_DIR / f"preview_{img_path.stem}.png"
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> 保存: {save_path}")

    print(f"\n=== 完了 ===")
    print(f"確認結果: {SAVE_DIR}")


if __name__ == "__main__":
    main()
