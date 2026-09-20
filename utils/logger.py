import os
import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt
from utils.metrics import plot_confusion_matrix_figure

class WandbLogger:
    """
    Trình quản lý Ghi nhận nhật ký (Logging) tập trung sử dụng Weights & Biases (wandb).
    """
    def __init__(self, config: dict):
        self.config = config
        wandb_cfg = config.get("wandb", {})
        
        self.enabled = wandb_cfg.get("mode", "online") != "disabled"
        if not self.enabled:
            print("[WandbLogger] WandB logging đã bị TẮT (disabled).")
            return

        project = wandb_cfg.get("project", "dl-flexible-pipeline")
        entity = wandb_cfg.get("entity", None)
        run_name = wandb_cfg.get("run_name", None)
        mode = wandb_cfg.get("mode", "online")
        tags = wandb_cfg.get("tags", [])

        print(f"[WandbLogger] Đang khởi tạo WandB: Project='{project}', Run Name='{run_name}', Mode='{mode}'")
        wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            config=config,
            mode=mode,
            tags=tags
        )

    def log(self, metrics: dict, step: int = None):
        """Log các giá trị scalar (Loss, Accuracy, Learning Rate...)"""
        if self.enabled and wandb.run is not None:
            if step is not None:
                wandb.log(metrics, step=step)
            else:
                wandb.log(metrics)

    def log_confusion_matrix(self, y_true, y_pred, class_names=None, title="Confusion Matrix", step=None):
        """
        Ghi ma trận nhầm lẫn dưới dạng biểu đồ tương tác của WandB và dạng hình ảnh Heatmap.
        """
        if not self.enabled or wandb.run is None:
            return

        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)

        # 1. WandB Interactive Plot
        try:
            wandb.log({
                f"{title}_interactive": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=y_true_arr,
                    preds=y_pred_arr,
                    class_names=class_names
                )
            }, step=step)
        except Exception as e:
            print(f"[WandbLogger] Warning: Không thể log interactive confusion matrix: {e}")

        # 2. Seaborn Matplotlib Heatmap Image
        try:
            fig, _ = plot_confusion_matrix_figure(y_true_arr, y_pred_arr, class_names=class_names)
            wandb.log({f"{title}_image": wandb.Image(fig)}, step=step)
            plt.close(fig)
        except Exception as e:
            print(f"[WandbLogger] Warning: Không thể log heatmap image confusion matrix: {e}")

    def log_predictions(self, images, y_true, y_pred, class_names=None, max_samples=16, step=None):
        """
        Log các hình ảnh dự đoán mẫu kèm Nhãn Thực tế (GT) vs Nhãn Dự đoán (Pred).
        """
        if not self.enabled or wandb.run is None:
            return

        wandb_images = []
        num_samples = min(len(images), max_samples)

        for i in range(num_samples):
            img = images[i]
            if isinstance(img, torch.Tensor):
                # Unnormalize / Format tensor image
                img_np = img.detach().cpu().numpy()
                if img_np.ndim == 3 and img_np.shape[0] in [1, 3]:
                    img_np = np.transpose(img_np, (1, 2, 0))
                # Min-max scaling về range [0, 1]
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
            else:
                img_np = np.array(img)

            gt_label = class_names[y_true[i]] if class_names and y_true[i] < len(class_names) else str(y_true[i])
            pred_label = class_names[y_pred[i]] if class_names and y_pred[i] < len(class_names) else str(y_pred[i])
            
            caption = f"GT: {gt_label} | Pred: {pred_label}"
            wandb_images.append(wandb.Image(img_np, caption=caption))

        wandb.log({"prediction_samples": wandb_images}, step=step)

    def finish(self):
        if self.enabled and wandb.run is not None:
            print("[WandbLogger] Hoàn tất và đóng phiên WandB.")
            wandb.finish()
