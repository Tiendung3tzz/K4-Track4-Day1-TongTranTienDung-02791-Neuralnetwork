"""Training, evaluation, and prediction utilities for the experiments.

Gồm: đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.
Mọi thí nghiệm chỉ là *đổi dict cfg* rồi gọi lại run_experiment (xem GUIDE, Part 2).

Mọi chỉ số (loss, accuracy, macro-F1) dùng cùng định nghĩa với scripts/evaluate.py.
"""
from __future__ import annotations

import contextlib
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, clip_gradients

# Cấu hình mặc định = BASELINE (M-base). `lr` do bạn tự chọn bằng val rồi điền vào.
DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=None,                   # Chọn bằng validation trước khi chạy baseline.
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip; hoặc số, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    seed=1,
)


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0.

    cm: ma trận nhầm lẫn (7, 7), hàng = nhãn thật, cột = dự đoán.
    """
    cm = np.asarray(cm)
    if cm.shape != (7, 7):
        raise ValueError(f"cm phải có shape (7, 7), nhận {cm.shape}")
    cm = cm.astype(np.float64, copy=False)
    tp = np.diag(cm)
    precision_den = cm.sum(axis=0)
    recall_den = cm.sum(axis=1)
    denom = precision_den + recall_den
    f1 = np.divide(2.0 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(f1.mean())


@torch.no_grad()
def predict(model, X, batch_size: int = 8192) -> torch.Tensor:
    """Trả về nhãn dự đoán int64 (N,) = argmax của logits.

    Các bước: model.eval(); duyệt X theo từng lô (không cần xáo); gom argmax(dim=1); torch.cat.
    """
    if X.ndim != 2:
        raise ValueError("X phải là tensor hai chiều")
    if batch_size <= 0:
        raise ValueError("batch_size phải dương")
    was_training = model.training
    model.eval()
    chunks = [model(X[start:start + batch_size]).argmax(dim=1) for start in range(0, len(X), batch_size)]
    model.train(was_training)
    if not chunks:
        return torch.empty(0, dtype=torch.int64, device=X.device)
    return torch.cat(chunks).to(dtype=torch.int64)


@torch.no_grad()
def evaluate(model, X, y, loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """Trả về dict(loss, acc, macro_f1) ở chế độ eval() (dropout tắt) và no_grad.

    Các bước:
      1. model.eval()
      2. tính logits theo từng lô; cộng dồn tổng loss (reduction="sum") rồi chia N cuối cùng
      3. pred = argmax; acc = (pred == y).mean()
      4. dựng ma trận nhầm lẫn 7x7 -> macro_f1_from_confusion
    Dùng hàm này cho: train loss (trên toàn bộ hoặc một tập con CỐ ĐỊNH của train), val, và eval cuối cùng.
    """
    if len(X) != len(y):
        raise ValueError("X và y phải có cùng số mẫu")
    if len(X) == 0:
        raise ValueError("không thể đánh giá tập rỗng")
    was_training = model.training
    model.eval()
    total_loss = 0.0
    predictions = []
    targets = []
    try:
        for start in range(0, len(X), batch_size):
            xb, yb = X[start:start + batch_size], y[start:start + batch_size]
            logits = model(xb)
            if loss_name.lower() == "ce":
                total_loss += float(F.cross_entropy(logits, yb, reduction="sum").item())
            elif loss_name.lower() == "mse":
                target = F.one_hot(yb, num_classes=logits.shape[1]).to(dtype=logits.dtype)
                total_loss += float(F.mse_loss(logits, target, reduction="sum").item())
            else:
                raise ValueError(f"loss không hợp lệ: {loss_name!r}")
            predictions.append(logits.argmax(dim=1))
            targets.append(yb)
        pred = torch.cat(predictions)
        target = torch.cat(targets)
        cm = torch.bincount(
            (target.to(torch.int64) * 7 + pred.to(torch.int64)), minlength=49
        ).reshape(7, 7).detach().cpu().numpy()
        if loss_name.lower() == "mse":
            loss = total_loss / (len(X) * 7)
        else:
            loss = total_loss / len(X)
        acc = float((pred == target).float().mean().item())
        return {"loss": float(loss), "acc": acc, "macro_f1": macro_f1_from_confusion(cm), "confusion": cm}
    finally:
        model.train(was_training)


def compute_loss(logits, y, loss_name: str):
    """"ce"  : cross-entropy nhận logit thô và nhãn int64 (F.cross_entropy).
       "mse" : MSE giữa logit và one-hot của y (ghi rõ bạn lấy trung bình thế nào).
    """
    name = loss_name.lower()
    if name == "ce":
        return F.cross_entropy(logits, y)
    if name == "mse":
        target = F.one_hot(y, num_classes=logits.shape[-1]).to(dtype=logits.dtype)
        return F.mse_loss(logits, target)
    raise ValueError(f"loss không hợp lệ: {loss_name!r}")


def run_experiment(cfg: dict, data: dict) -> dict:
    """Huấn luyện một cấu hình và trả về lịch sử + tóm tắt.

    Args:
        cfg : dict cấu hình (xem DEFAULT_CFG)
        data: kết quả của data.prepare_data (tensor X_tr, y_tr, X_val, y_val, X_eval, y_eval trên device)

    Trả về dict:
        {"cfg": cfg,
         "history": {"epoch": [...], "train_loss": [...], "val_loss": [...], "val_acc": [...],
                     "val_macro_f1": [...], "grad_norm": [...], "epoch_time_s": [...]},
         "summary": {"step0_loss", "best_val_loss", "best_epoch", "final_train_loss", "final_val_loss",
                     "val_acc", "val_macro_f1", "time_per_epoch_s", "peak_mem_MB", "diverged"},
         "best_state": state_dict của epoch có val_loss thấp nhất (giữ trong RAM để dự đoán eval)}
    (tên khoá của summary trùng tên cột trong experiments.xlsx)

    Các bước:
      0. set_seed(cfg["seed"]); tạo model = MLP(...), assert count_params(model) == EXPECTED_PARAMS[hidden]
         chuyển model lên device; tạo optimizer = build_optimizer(...)
         nếu precision == "fp16": scaler = torch.amp.GradScaler(...)
      1. step0_loss = evaluate(model, X_val, y_val)["loss"]   # TRƯỚC bước cập nhật đầu tiên; kỳ vọng ≈ ln 7
      2. for epoch in 1..epochs:
           model.train()
           for xb, yb in iterate_batches(X_tr, y_tr, cfg["batch"], generator):
               with torch.autocast(...)  nếu precision != "fp32":   # chỉ bọc forward + loss
                   logits = model(xb); loss = compute_loss(logits, yb, cfg["loss"])
               optimizer.zero_grad(set_to_none=True)
               backward (qua scaler nếu fp16)
               nếu fp16 và có clip: scaler.unscale_(optimizer)  TRƯỚC khi clip
               gn = clip_gradients(model.parameters(), cfg["clip_norm"])   # chuẩn TRƯỚC khi cắt; ghi lại
               bước cập nhật (scaler.step(optimizer); scaler.update() nếu fp16, ngược lại optimizer.step())
               nếu loss là NaN/inf: đặt diverged=True và dừng sớm, ĐỪNG để notebook treo
           cuối epoch (dùng evaluate, chế độ eval):
               train_loss trên toàn bộ train (hoặc 1 tập con CỐ ĐỊNH ~50 000 mẫu), val_loss/val_acc/val_macro_f1
               grad_norm trung bình của epoch; thời gian epoch (torch.cuda.synchronize() nếu dùng GPU)
               nếu val_loss tốt nhất từ trước tới giờ: lưu best_state (bản sao state_dict) và best_epoch
      3. tổng hợp summary tại best_epoch (val_acc, val_macro_f1 lấy ở best_epoch); peak_mem_MB nếu có GPU
    TUYỆT ĐỐI không đưa X_eval vào hàm này để chọn epoch/cấu hình. Chỉ dùng val.
    """
    cfg = {**DEFAULT_CFG, **cfg}
    cfg["hidden"] = tuple(cfg["hidden"])
    if cfg["lr"] is None:
        raise ValueError("cfg['lr'] phải được chọn bằng validation trước khi chạy")
    set_seed(cfg["seed"])
    X_tr, y_tr = data["X_tr"], data["y_tr"]
    X_val, y_val = data["X_val"], data["y_val"]
    device = X_tr.device
    model = MLP(hidden=cfg["hidden"], dropout=cfg["dropout"], init=cfg["init"]).to(device)
    expected = EXPECTED_PARAMS.get(tuple(cfg["hidden"]))
    if expected is not None:
        assert count_params(model) == expected
    optimizer = build_optimizer(
        cfg["optimizer"], model.parameters(), cfg["lr"],
        weight_decay=cfg["weight_decay"], momentum=cfg.get("momentum", 0.9),
    )
    precision = cfg.get("precision", "fp32").lower()
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("precision phải là fp32, fp16 hoặc bf16")
    if precision == "fp16":
        amp_dtype = torch.float16
    elif precision == "bf16":
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = None
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(precision == "fp16" and device.type == "cuda")
    )
    generator = torch.Generator(device="cpu")
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_acc": [],
               "val_macro_f1": [], "grad_norm": [], "epoch_time_s": []}
    diverged = False
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    step0_loss = evaluate(model, X_val, y_val, cfg["loss"])["loss"]
    best_val = float("inf")
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    train_limit = int(cfg.get("train_eval_samples", 50_000))
    train_limit = max(1, min(train_limit, len(X_tr)))
    for epoch in range(1, int(cfg["epochs"]) + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        model.train()
        generator.manual_seed(int(cfg["seed"]) + epoch)
        epoch_norms = []
        for xb, yb in iterate_batches(X_tr, y_tr, int(cfg["batch"]), generator=generator, shuffle=True):
            optimizer.zero_grad(set_to_none=True)
            context = (torch.autocast(device_type=device.type, dtype=amp_dtype)
                       if amp_dtype is not None else contextlib.nullcontext())
            with context:
                logits = model(xb)
                loss = compute_loss(logits, yb, cfg["loss"])
            if not torch.isfinite(loss).item():
                diverged = True
                break
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()
            grad_norm = clip_gradients(model.parameters(), cfg.get("clip_norm"))
            if not np.isfinite(grad_norm):
                diverged = True
                break
            epoch_norms.append(grad_norm)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if diverged:
            break
        train_metrics = evaluate(model, X_tr[:train_limit], y_tr[:train_limit], cfg["loss"])
        val_metrics = evaluate(model, X_val, y_val, cfg["loss"])
        elapsed = time.perf_counter() - started
        history["epoch"].append(epoch)
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["grad_norm"].append(float(np.mean(epoch_norms)) if epoch_norms else float("nan"))
        history["epoch_time_s"].append(float(elapsed))
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    if history["epoch"]:
        best_idx = int(np.argmin(history["val_loss"]))
        best_epoch = history["epoch"][best_idx]
        best_val_loss = history["val_loss"][best_idx]
        best_val_acc = history["val_acc"][best_idx]
        best_val_f1 = history["val_macro_f1"][best_idx]
        final_train_loss = history["train_loss"][-1]
        final_val_loss = history["val_loss"][-1]
        time_per_epoch = float(np.mean(history["epoch_time_s"]))
    else:
        best_epoch = 0
        best_val_loss = final_train_loss = final_val_loss = float("nan")
        best_val_acc = best_val_f1 = float("nan")
        time_per_epoch = float("nan")
    peak_mem = (torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                if device.type == "cuda" else 0.0)
    summary = {
        "step0_loss": float(step0_loss), "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch), "final_train_loss": float(final_train_loss),
        "final_val_loss": float(final_val_loss), "val_acc": float(best_val_acc),
        "val_macro_f1": float(best_val_f1), "time_per_epoch_s": time_per_epoch,
        "peak_mem_MB": float(peak_mem), "diverged": bool(diverged),
    }
    return {"cfg": cfg, "history": history, "summary": summary, "best_state": best_state}


def write_predictions(row_id, preds, path: str) -> None:
    """Ghi file nộp cho scripts/evaluate.py: CSV có tiêu đề `row_id,pred`.

    row_id : mảng row_id của tập eval (data["eval_row_id"])
    preds  : nhãn dự đoán int64 0..6 (cùng thứ tự với row_id)
    Phải đủ mọi dòng của tập eval, mỗi row_id đúng một lần.
    """
    row_id = np.asarray(row_id)
    preds = np.asarray(preds)
    if row_id.ndim != 1 or preds.ndim != 1 or len(row_id) != len(preds):
        raise ValueError("row_id và preds phải là cùng chiều dài")
    if len(np.unique(row_id)) != len(row_id):
        raise ValueError("row_id không được trùng")
    if len(preds) and (preds.min() < 0 or preds.max() > 6):
        raise ValueError("pred phải nằm trong khoảng 0..6")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("row_id", "pred"))
        writer.writerows((int(rid), int(pred)) for rid, pred in zip(row_id, preds))


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> None:
    """Dùng MỘT LẦN cho cấu hình cuối cùng (và baseline): nạp best_state, dự đoán eval, ghi predictions.

    Các bước:
      1. model = MLP(...); model.load_state_dict(result["best_state"]); lên device
      2. preds = predict(model, data["X_eval"])  # fp32, eval mode
      3. write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
      4. chạy `python scripts/evaluate.py --pred <pred_path>` và ghi kết quả vào bảng/báo cáo
    """
    cfg_full = {**DEFAULT_CFG, **cfg}
    cfg_full["hidden"] = tuple(cfg_full["hidden"])
    device = data["X_eval"].device
    model = MLP(
        hidden=cfg_full["hidden"], dropout=cfg_full["dropout"], init=cfg_full["init"]
    ).to(device)
    model.load_state_dict(result["best_state"])
    preds = predict(model, data["X_eval"])
    write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
