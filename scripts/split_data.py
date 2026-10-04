"""Chia dữ liệu CoverType thành train / eval theo data/split_metadata.csv.

Chạy từ thư mục gốc của repo (sinh viên chạy đúng lệnh này sau khi git pull):

    python scripts/split_data.py

Đầu vào : data/covtype.csv.gz        (581 012 dòng: 54 đặc trưng + cột Cover_Type 1..7)
          data/split_metadata.csv    (cột row_id, split; split ∈ {train, eval})
Đầu ra  : data/processed/train.npz   và   data/processed/eval.npz
          mỗi file có: X (float32, N x 54), y (int64, 0..6), row_id (int64), feature_names

Lưu ý:
  - Chỉ có hai tập: train và eval. Không có tập validation. Bạn tự tách validation
    từ train (xem GUIDE, Part 0). Tập eval chỉ dùng để chấm điểm cuối.
  - Script KHÔNG chuẩn hoá dữ liệu. Việc chuẩn hoá bằng thống kê của train là phần của bạn.
  - Nhãn được đổi từ 1..7 về 0..6 và ép sang int64 (cross-entropy của PyTorch cần int64).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


if hasattr(sys.stdout, "reconfigure"):
    # Windows terminals may use a legacy code page that cannot print the
    # Vietnamese diagnostics below.  Keep the required command portable.
    sys.stdout.reconfigure(encoding="utf-8")

LABEL = "Cover_Type"
N_TOTAL, N_TRAIN, N_EVAL = 581_012, 464_809, 116_203   # kiểm tra metadata không bị sửa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/covtype.csv.gz")
    ap.add_argument("--meta", default="data/split_metadata.csv")
    ap.add_argument("--out", default="data/processed")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    meta = pd.read_csv(args.meta)

    # ---- kiểm tra tính toàn vẹn
    assert len(df) == N_TOTAL, f"covtype.csv.gz phải có {N_TOTAL} dòng, hiện có {len(df)}"
    assert len(meta) == N_TOTAL, "split_metadata.csv phải có đúng 1 dòng cho mỗi mẫu"
    assert meta["row_id"].is_unique and set(meta["row_id"]) == set(range(N_TOTAL)), "row_id phải là 0..N-1, không trùng"
    assert set(meta["split"]) == {"train", "eval"}, "cột split chỉ được chứa 'train' và 'eval'"
    n_train, n_eval = (meta["split"] == "train").sum(), (meta["split"] == "eval").sum()
    assert (n_train, n_eval) == (N_TRAIN, N_EVAL), f"kích thước tập lệch: train={n_train}, eval={n_eval}"

    feature_names = [c for c in df.columns if c != LABEL]
    assert len(feature_names) == 54

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = meta.sort_values("row_id").reset_index(drop=True)
    for name in ("train", "eval"):
        ids = meta.loc[meta["split"] == name, "row_id"].to_numpy()
        part = df.iloc[ids]
        X = part[feature_names].to_numpy(dtype=np.float32)
        y = (part[LABEL].to_numpy() - 1).astype(np.int64)
        np.savez_compressed(out / f"{name}.npz", X=X, y=y, row_id=ids.astype(np.int64),
                            feature_names=np.array(feature_names))
        print(f"{name:5s}: X {X.shape} {X.dtype}, y {y.shape} {y.dtype}, nhãn {y.min()}..{y.max()} -> {out / (name + '.npz')}")

    # ---- thống kê để bạn đối chiếu
    tr = np.load(out / "train.npz")["y"]; ev = np.load(out / "eval.npz")["y"]
    ctr, cev = np.bincount(tr, minlength=7), np.bincount(ev, minlength=7)
    print("\nlớp   train(%)   eval(%)")
    for c in range(7):
        print(f"{c:>3d}   {100*ctr[c]/ctr.sum():7.3f}   {100*cev[c]/cev.sum():7.3f}")
    print(f"\nđoán luôn lớp đa số (lớp {ctr.argmax()}) cho accuracy trên eval = {cev[ctr.argmax()]/cev.sum():.4f}")


if __name__ == "__main__":
    main()
