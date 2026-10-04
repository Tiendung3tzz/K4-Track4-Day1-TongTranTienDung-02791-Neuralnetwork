"""Utilities for loading, splitting, standardizing, and batching the data.

Nhiệm vụ: nạp tập train/eval đã chia sẵn, tách validation từ train, chuẩn hoá, đưa lên thiết bị.

Điều kiện trước: đã chạy `python scripts/split_data.py` (tạo data/processed/train.npz, eval.npz).

Quy ước dữ liệu (xem README mục 2 và 3):
    X : float32, shape (N, 54)   — 10 cột đầu là số liên tục, 44 cột sau là nhị phân (one-hot)
    y : int64,   shape (N,)      — nhãn 0..6
Tập eval CHỈ dùng để chấm điểm cuối. Không dùng nó để chọn cấu hình, chuẩn hoá hay dừng sớm.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

N_NUMERIC = 10  # số cột liên tục cần chuẩn hoá (cột 0..9)


def load_split(processed_dir: str = "data/processed"):
    """Nạp train và eval từ file .npz.

    Trả về: X_train_full, y_train_full, X_eval, y_eval, eval_row_id
    Các bước:
      1. np.load(f"{processed_dir}/train.npz") -> khoá "X", "y"
      2. np.load(f"{processed_dir}/eval.npz")  -> khoá "X", "y", "row_id"
      3. assert shape/dtype đúng quy ước ở đầu file
    """
    processed_path = Path(processed_dir)
    with np.load(processed_path / "train.npz", allow_pickle=False) as train:
        if not {"X", "y"}.issubset(train.files):
            raise KeyError("train.npz phải có khoá X và y")
        X_train_full = np.array(train["X"], copy=True)
        y_train_full = np.array(train["y"], copy=True)
    with np.load(processed_path / "eval.npz", allow_pickle=False) as eval_data:
        if not {"X", "y", "row_id"}.issubset(eval_data.files):
            raise KeyError("eval.npz phải có khoá X, y và row_id")
        X_eval = np.array(eval_data["X"], copy=True)
        y_eval = np.array(eval_data["y"], copy=True)
        eval_row_id = np.array(eval_data["row_id"], copy=True)

    for name, X, y in (("train", X_train_full, y_train_full), ("eval", X_eval, y_eval)):
        if X.ndim != 2 or X.shape[1] != 54 or X.dtype != np.float32:
            raise ValueError(f"{name} X phải có shape (N, 54) và dtype float32, nhận {X.shape}, {X.dtype}")
        if y.shape != (X.shape[0],) or y.dtype != np.int64:
            raise ValueError(f"{name} y phải có shape (N,) và dtype int64, nhận {y.shape}, {y.dtype}")
        if len(y) and (y.min() < 0 or y.max() > 6):
            raise ValueError(f"{name} y phải chứa nhãn 0..6")
    if eval_row_id.shape != (X_eval.shape[0],) or eval_row_id.dtype != np.int64:
        raise ValueError("eval_row_id phải có shape (N_eval,) và dtype int64")
    if len(np.unique(eval_row_id)) != len(eval_row_id):
        raise ValueError("eval_row_id không được trùng")
    return X_train_full, y_train_full, X_eval, y_eval, eval_row_id


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    """Tách validation TỪ train (không đụng eval). Phân tầng theo nhãn.

    Trả về: X_tr, y_tr, X_val, y_val
    Gợi ý: sklearn.model_selection.train_test_split(..., stratify=y, random_state=seed)
    Dùng CÙNG seed và val_fraction cho mọi thí nghiệm để so sánh công bằng.
    """
    X, y = np.asarray(X), np.asarray(y)
    if X.ndim != 2 or len(X) != len(y):
        raise ValueError("X phải là ma trận và có cùng số dòng với y")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction phải nằm giữa 0 và 1")
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_fraction, random_state=seed, stratify=y
    )
    return X_tr, y_tr, X_val, y_val


def fit_standardizer(X_tr):
    """Tính mean và std của N_NUMERIC cột đầu CHỈ trên tập train (sau khi tách val).

    Trả về: mean (shape (10,)), std (shape (10,))
    Câu hỏi: vì sao không được tính trên toàn bộ dữ liệu hay trên eval?
    """
    X_tr = np.asarray(X_tr)
    if X_tr.ndim != 2 or X_tr.shape[1] < N_NUMERIC:
        raise ValueError(f"X_tr phải có ít nhất {N_NUMERIC} cột")
    numeric = X_tr[:, :N_NUMERIC].astype(np.float64, copy=False)
    mean = numeric.mean(axis=0)
    std = numeric.std(axis=0)
    std = np.where(std == 0.0, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_standardizer(X, mean, std):
    """Trả về bản sao của X, trong đó 10 cột đầu được (x - mean) / std; 44 cột nhị phân giữ nguyên.

    Chú ý: không sửa X tại chỗ nếu bạn còn dùng lại nó; chú ý std = 0 (nếu có).
    """
    X, mean, std = np.asarray(X), np.asarray(mean), np.asarray(std)
    if X.ndim != 2 or X.shape[1] < N_NUMERIC:
        raise ValueError(f"X phải có ít nhất {N_NUMERIC} cột")
    if mean.shape != (N_NUMERIC,) or std.shape != (N_NUMERIC,):
        raise ValueError("mean và std phải có shape (10,)")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std <= 0):
        raise ValueError("mean/std phải hữu hạn và std phải dương")
    result = np.array(X, dtype=np.float32, copy=True)
    result[:, :N_NUMERIC] = (result[:, :N_NUMERIC].astype(np.float64) - mean) / std
    return result


def prepare_data(device: str, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed") -> dict:
    """Gộp các bước trên và đưa TOÀN BỘ dữ liệu lên `device` một lần (không dùng DataLoader).

    Trả về dict gồm các tensor trên device:
        X_tr, y_tr, X_val, y_val, X_eval, y_eval        (y là int64)
    và các mảng numpy: eval_row_id
    Các bước:
      1. load_split -> make_val_split -> fit_standardizer (chỉ trên X_tr)
      2. apply_standardizer cho X_tr, X_val, X_eval bằng CÙNG mean/std
      3. torch.tensor(..., device=device); X là float32, y là int64
      4. in ra kích thước các tập và accuracy của chiến lược "luôn đoán lớp đa số" trên val
    """
    X_full, y_full, X_eval, y_eval, eval_row_id = load_split(processed_dir)
    X_tr, y_tr, X_val, y_val = make_val_split(X_full, y_full, val_fraction, seed)
    mean, std = fit_standardizer(X_tr)
    X_tr = apply_standardizer(X_tr, mean, std)
    X_val = apply_standardizer(X_val, mean, std)
    X_eval = apply_standardizer(X_eval, mean, std)
    data = {
        "X_tr": torch.tensor(X_tr, dtype=torch.float32, device=device),
        "y_tr": torch.tensor(y_tr, dtype=torch.int64, device=device),
        "X_val": torch.tensor(X_val, dtype=torch.float32, device=device),
        "y_val": torch.tensor(y_val, dtype=torch.int64, device=device),
        "X_eval": torch.tensor(X_eval, dtype=torch.float32, device=device),
        "y_eval": torch.tensor(y_eval, dtype=torch.int64, device=device),
        "eval_row_id": eval_row_id,
        "mean": mean,
        "std": std,
    }
    majority_class = int(np.bincount(y_tr, minlength=7).argmax())
    print(f"X_tr={tuple(data['X_tr'].shape)}, X_val={tuple(data['X_val'].shape)}, X_eval={tuple(data['X_eval'].shape)}")
    print(f"majority class on train={majority_class}, validation accuracy={np.mean(y_val == majority_class):.4f}")
    return data


def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = True):
    """Generator trả về từng cặp (xb, yb), thay cho DataLoader.

    Các bước:
      1. nếu shuffle: perm = torch.randperm(len(X), generator=generator, device=X.device); ngược lại arange
      2. for i in range(0, N, batch_size): idx = perm[i:i+batch_size]; yield X[idx], y[idx]
    Chú ý: batch cuối có thể nhỏ hơn batch_size; hãy quyết định bạn xử lý thế nào và ghi lại.
    """
    if X.ndim == 0 or y.ndim == 0 or len(X) != len(y):
        raise ValueError("X và y phải có cùng số mẫu")
    if batch_size <= 0:
        raise ValueError("batch_size phải là số nguyên dương")
    if shuffle:
        try:
            perm = torch.randperm(len(X), generator=generator, device=X.device)
        except RuntimeError:
            if generator is None:
                raise
            perm = torch.randperm(len(X), generator=generator, device="cpu").to(X.device)
    else:
        perm = torch.arange(len(X), device=X.device)
    for start in range(0, len(X), batch_size):
        idx = perm[start:start + batch_size]
        yield X[idx], y[idx]
