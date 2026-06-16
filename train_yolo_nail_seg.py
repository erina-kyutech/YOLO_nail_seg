# -*- coding: utf-8 -*-
"""
YOLOv11n-seg を爪・指先セグメンテーション用にファインチューニングするスクリプト。

クラス:
  0: finger_tip
  1: nail

データセット:
  C:/Users/Owner/PycharmProjects/Tactile_sensors_20230929/yolo_seg_dataset/

出力:
  runs/segment/nail_seg_v1/ 以下に重みファイル・学習ログが保存される
"""

from pathlib import Path
from ultralytics import YOLO

# =========================================================
# 設定
# =========================================================

# data.yaml のパス
DATA_YAML = Path(
    r"C:\Users\Owner\PycharmProjects\Tactile_sensors_20230929"
    r"\yolo_seg_dataset\data.yaml"
)

# 使用するベースモデル
# yolo11n-seg: 最軽量(nano)、ファインチューニング向け
# yolo11s-seg: 少し大きい(small)、精度が少し上がる可能性あり
BASE_MODEL = "yolo11n-seg.pt"

# 学習パラメータ
EPOCHS      = 100       # エポック数 (63枚なら100で十分)
BATCH_SIZE  = 16        # バッチサイズ (RTX A5000なら余裕)
IMG_SIZE    = 320       # 入力解像度 (元画像290x150に近い値。32の倍数)
DEVICE      = 0         # GPU番号 (0 = 1枚目のGPU)
PROJECT     = "runs/segment"
NAME        = "nail_seg_v1"

# early stopping: val_lossが改善しなければ何エポックで止めるか
PATIENCE    = 30


# =========================================================
# メイン
# =========================================================
def main():
    print("=== YOLOv11n-seg ファインチューニング ===")
    print(f"データセット : {DATA_YAML}")
    print(f"ベースモデル : {BASE_MODEL}")
    print(f"エポック数   : {EPOCHS}")
    print(f"バッチサイズ : {BATCH_SIZE}")
    print(f"入力解像度   : {IMG_SIZE}")

    if not DATA_YAML.exists():
        raise FileNotFoundError(f"data.yaml が見つかりません: {DATA_YAML}")

    # モデル読み込み (初回は自動ダウンロード)
    model = YOLO(BASE_MODEL)

    # 学習
    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        device=DEVICE,
        project=PROJECT,
        name=NAME,
        patience=PATIENCE,

        # データ拡張 (63枚の少量データなので積極的に使う)
        augment=True,
        hsv_h=0.015,    # 色相のランダム変化 (爪色の変動をカバー)
        hsv_s=0.3,      # 彩度のランダム変化
        hsv_v=0.2,      # 明度のランダム変化
        fliplr=0.5,     # 左右反転
        flipud=0.0,     # 上下反転なし (指の向きが変わるため)
        scale=0.3,      # スケールのランダム変化
        translate=0.1,  # 平行移動
        degrees=5.0,    # 回転 (小さめ)
        mosaic=0.5,     # モザイク拡張 (少量データに有効)

        # その他
        save=True,
        save_period=10,  # 10エポックごとに重みを保存
        plots=True,      # 学習曲線・結果画像を保存
        verbose=True,
    )

    print("=== 学習完了 ===")
    print(f"結果は {PROJECT}/{NAME}/ に保存されました")

    # ベストモデルで検証
    print("=== ベストモデルで検証 ===")
    best_weight = Path(PROJECT) / NAME / "weights" / "best.pt"
    if best_weight.exists():
        model_best = YOLO(str(best_weight))
        metrics = model_best.val(data=str(DATA_YAML), device=DEVICE)
        print(f"mAP50      : {metrics.seg.map50:.4f}")
        print(f"mAP50-95   : {metrics.seg.map:.4f}")
    else:
        print(f"[WARN] best.pt が見つかりません: {best_weight}")


if __name__ == "__main__":
    main()
