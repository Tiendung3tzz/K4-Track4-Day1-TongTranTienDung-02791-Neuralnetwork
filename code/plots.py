"""Plotting helpers for experiment histories.

Ảnh biểu đồ là sản phẩm nộp (xem README mục 6): mỗi thí nghiệm một ảnh figures/<exp_id>.png.
Khi notebook chạy trong code/, lưu vào "../figures/" (ví dụ path = f"../figures/{exp_id}.png").
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_run(result: dict, path: str) -> None:
    """Vẽ MỘT thí nghiệm thành một ảnh PNG có ít nhất 3 ô:
         (1) train_loss và val_loss theo epoch (cùng một trục)
         (2) val_acc (và nên có val_macro_f1) theo epoch
         (3) grad_norm theo epoch (đo TRƯỚC khi clip)
    Yêu cầu: tiêu đề ghi exp_id và cấu hình chính (optimizer, lr, batch, ...), có nhãn trục và chú thích.
    Các bước: fig, axes = plt.subplots(1, 3, figsize=...); plot; set_title/xlabel/legend;
              fig.savefig(path, dpi=..., bbox_inches="tight"); plt.close(fig)
    Gợi ý: đánh dấu best_epoch bằng đường thẳng đứng.
    """
    history = result.get("history", {}); epochs = history.get("epoch", [])
    if not epochs: raise ValueError("result không có history để vẽ")
    cfg = result.get("cfg", {}); exp_id = cfg.get("exp_id", "experiment")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].plot(epochs, history.get("train_loss", []), label="train loss"); axes[0].plot(epochs, history.get("val_loss", []), label="val loss"); axes[0].set(title="Loss", xlabel="epoch", ylabel="loss"); axes[0].legend()
    axes[1].plot(epochs, history.get("val_acc", []), label="val accuracy"); axes[1].plot(epochs, history.get("val_macro_f1", []), label="val macro-F1"); axes[1].set(title="Validation metrics", xlabel="epoch", ylabel="score"); axes[1].legend()
    axes[2].plot(epochs, history.get("grad_norm", []), label="grad norm"); axes[2].set(title="Gradient norm (before clip)", xlabel="epoch", ylabel="L2 norm"); axes[2].legend()
    best = result.get("summary", {}).get("best_epoch")
    if best in epochs:
        for axis in axes: axis.axvline(best, color="black", linestyle="--", alpha=.35)
    fig.suptitle(f"{exp_id} — optimizer={cfg.get('optimizer')} lr={cfg.get('lr')} batch={cfg.get('batch')}"); fig.tight_layout()
    output = Path(path); output.parent.mkdir(parents=True, exist_ok=True); fig.savefig(output, dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_compare(results: list[dict], metric: str, path: str, title: str = "") -> None:
    """Vẽ chồng một chỉ số (ví dụ "val_loss", "val_macro_f1", "grad_norm") của nhiều thí nghiệm
    trên cùng một trục, mỗi thí nghiệm một đường, chú thích bằng exp_id.

    Dùng cho ảnh figures/compare_<nhóm>.png (ví dụ compare_optimizer.png).
    """
    if not results: raise ValueError("cần ít nhất một result")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for result in results:
        history = result.get("history", {})
        if metric not in history: raise KeyError(f"history không có metric {metric!r}")
        ax.plot(history.get("epoch", []), history[metric], label=result.get("cfg", {}).get("exp_id", "experiment"))
    ax.set(xlabel="epoch", ylabel=metric, title=title or metric); ax.legend(); ax.grid(alpha=.25); output = Path(path); output.parent.mkdir(parents=True, exist_ok=True); fig.tight_layout(); fig.savefig(output, dpi=150, bbox_inches="tight"); plt.close(fig)
