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
    seed = int(seed); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0.

    cm: ma trận nhầm lẫn (7, 7), hàng = nhãn thật, cột = dự đoán.
    """
    cm = np.asarray(cm, dtype=np.float64)
    if cm.shape != (7, 7): raise ValueError("cm phải có shape (7, 7)")
    tp = np.diag(cm); denom = cm.sum(0) + cm.sum(1)
    return float(np.divide(2 * tp, denom, out=np.zeros(7), where=denom > 0).mean())


@torch.no_grad()
def predict(model, X, batch_size: int = 8192) -> torch.Tensor:
    """Trả về nhãn dự đoán int64 (N,) = argmax của logits.

    Các bước: model.eval(); duyệt X theo từng lô (không cần xáo); gom argmax(dim=1); torch.cat.
    """
    if X.ndim != 2 or batch_size <= 0: raise ValueError("X/batch_size không hợp lệ")
    was_training = model.training; model.eval()
    try:
        chunks = [model(X[i:i + batch_size]).argmax(1) for i in range(0, len(X), batch_size)]
        return torch.cat(chunks).to(torch.int64) if chunks else torch.empty(0, dtype=torch.int64, device=X.device)
    finally: model.train(was_training)


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
    if len(X) != len(y) or not len(X): raise ValueError("tập đánh giá không hợp lệ")
    was_training = model.training; model.eval(); total = 0.0; preds = []; targets = []
    try:
        for i in range(0, len(X), batch_size):
            logits, yb = model(X[i:i + batch_size]), y[i:i + batch_size]
            if loss_name == "ce": total += float(F.cross_entropy(logits, yb, reduction="sum"))
            elif loss_name == "mse":
                oh = F.one_hot(yb, logits.shape[1]).to(logits.dtype)
                total += float(F.mse_loss(logits, oh, reduction="sum"))
            else: raise ValueError(f"loss không hợp lệ: {loss_name!r}")
            preds.append(logits.argmax(1)); targets.append(yb)
        pred, target = torch.cat(preds), torch.cat(targets)
        cm = torch.bincount(target.to(torch.int64) * 7 + pred.to(torch.int64), minlength=49).reshape(7, 7).cpu().numpy()
        loss = total / (len(X) * (7 if loss_name == "mse" else 1))
        return {"loss": float(loss), "acc": float((pred == target).float().mean()), "macro_f1": macro_f1_from_confusion(cm), "confusion": cm}
    finally: model.train(was_training)


def compute_loss(logits, y, loss_name: str):
    """"ce"  : cross-entropy nhận logit thô và nhãn int64 (F.cross_entropy).
       "mse" : MSE giữa logit và one-hot của y (ghi rõ bạn lấy trung bình thế nào).
    """
    if loss_name == "ce": return F.cross_entropy(logits, y)
    if loss_name == "mse": return F.mse_loss(logits, F.one_hot(y, logits.shape[-1]).to(logits.dtype))
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
    cfg = {**DEFAULT_CFG, **cfg}; cfg["hidden"] = tuple(cfg["hidden"])
    if cfg["lr"] is None: raise ValueError("cfg['lr'] phải được chọn")
    set_seed(cfg["seed"]); X_tr, y_tr = data["X_tr"], data["y_tr"]; X_val, y_val = data["X_val"], data["y_val"]; device = X_tr.device
    model = MLP(cfg["hidden"], cfg["dropout"], cfg["init"]).to(device)
    if tuple(cfg["hidden"]) in EXPECTED_PARAMS: assert count_params(model) == EXPECTED_PARAMS[tuple(cfg["hidden"])]
    optimizer = build_optimizer(cfg["optimizer"], model.parameters(), cfg["lr"], cfg["weight_decay"], cfg.get("momentum", .9))
    precision = cfg.get("precision", "fp32"); dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(precision); scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16" and device.type == "cuda")
    gen = torch.Generator(device="cpu"); history = {k: [] for k in ("epoch", "train_loss", "val_loss", "val_acc", "val_macro_f1", "grad_norm", "epoch_time_s")}; diverged = False; step0_loss = evaluate(model, X_val, y_val, cfg["loss"])["loss"]; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; best = float("inf")
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    n_eval = min(max(1, int(cfg.get("train_eval_samples", 50_000))), len(X_tr))
    for epoch in range(1, int(cfg["epochs"]) + 1):
        if device.type == "cuda": torch.cuda.synchronize(device)
        start = time.perf_counter(); model.train(); gen.manual_seed(int(cfg["seed"]) + epoch); norms = []
        for xb, yb in iterate_batches(X_tr, y_tr, int(cfg["batch"]), gen, True):
            optimizer.zero_grad(set_to_none=True); context = torch.autocast(device.type, dtype=dtype) if dtype else contextlib.nullcontext()
            with context: loss = compute_loss(model(xb), yb, cfg["loss"])
            if not torch.isfinite(loss): diverged = True; break
            if scaler.is_enabled(): scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            else: loss.backward()
            gn = clip_gradients(model.parameters(), cfg.get("clip_norm"));
            if not np.isfinite(gn): diverged = True; break
            norms.append(gn)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
        if diverged: break
        tr, va = evaluate(model, X_tr[:n_eval], y_tr[:n_eval], cfg["loss"]), evaluate(model, X_val, y_val, cfg["loss"]); elapsed = time.perf_counter() - start
        for key, value in (("epoch", epoch), ("train_loss", tr["loss"]), ("val_loss", va["loss"]), ("val_acc", va["acc"]), ("val_macro_f1", va["macro_f1"]), ("grad_norm", float(np.mean(norms)) if norms else float("nan")), ("epoch_time_s", elapsed)): history[key].append(value)
        if va["loss"] < best: best = va["loss"]; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if history["epoch"]:
        i = int(np.argmin(history["val_loss"])); best_epoch = history["epoch"][i]; best_val_loss = history["val_loss"][i]; best_acc = history["val_acc"][i]; best_f1 = history["val_macro_f1"][i]; final_tr = history["train_loss"][-1]; final_va = history["val_loss"][-1]; tpe = float(np.mean(history["epoch_time_s"]))
    else: best_epoch = 0; best_val_loss = best_acc = best_f1 = final_tr = final_va = tpe = float("nan")
    peak = torch.cuda.max_memory_allocated(device) / 1024 ** 2 if device.type == "cuda" else 0.0
    summary = {"step0_loss": float(step0_loss), "best_val_loss": float(best_val_loss), "best_epoch": int(best_epoch), "final_train_loss": float(final_tr), "final_val_loss": float(final_va), "val_acc": float(best_acc), "val_macro_f1": float(best_f1), "time_per_epoch_s": tpe, "peak_mem_MB": float(peak), "diverged": bool(diverged)}
    return {"cfg": cfg, "history": history, "summary": summary, "best_state": best_state}


def write_predictions(row_id, preds, path: str) -> None:
    """Ghi file nộp cho scripts/evaluate.py: CSV có tiêu đề `row_id,pred`.

    row_id : mảng row_id của tập eval (data["eval_row_id"])
    preds  : nhãn dự đoán int64 0..6 (cùng thứ tự với row_id)
    Phải đủ mọi dòng của tập eval, mỗi row_id đúng một lần.
    """
    row_id, preds = np.asarray(row_id), np.asarray(preds)
    if row_id.ndim != 1 or preds.ndim != 1 or len(row_id) != len(preds) or len(np.unique(row_id)) != len(row_id): raise ValueError("row_id/preds không hợp lệ")
    if len(preds) and (preds.min() < 0 or preds.max() > 6): raise ValueError("pred phải trong 0..6")
    out = Path(path); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(("row_id", "pred")); w.writerows((int(a), int(b)) for a, b in zip(row_id, preds))


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> None:
    """Dùng MỘT LẦN cho cấu hình cuối cùng (và baseline): nạp best_state, dự đoán eval, ghi predictions.

    Các bước:
      1. model = MLP(...); model.load_state_dict(result["best_state"]); lên device
      2. preds = predict(model, data["X_eval"])  # fp32, eval mode
      3. write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
      4. chạy `python scripts/evaluate.py --pred <pred_path>` và ghi kết quả vào bảng/báo cáo
    """
    full = {**DEFAULT_CFG, **cfg}; full["hidden"] = tuple(full["hidden"]); model = MLP(full["hidden"], full["dropout"], full["init"]).to(data["X_eval"].device); model.load_state_dict(result["best_state"]); write_predictions(data["eval_row_id"], predict(model, data["X_eval"]).cpu().numpy(), pred_path)
