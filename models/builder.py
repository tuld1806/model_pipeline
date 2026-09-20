import os
import inspect
import torch
import torch.nn as nn
import models.arch as arch_module

def build_model(config: dict) -> nn.Module:
    """
    Hàm Factory khởi tạo mô hình linh hoạt dựa trên cấu hình config.yaml
    - Tự động nạp Class mô hình từ models/arch.py
    - Nạp weights từ file .pth nếu được chỉ định trong weights_path
    """
    model_cfg = config.get("model", {})
    model_name = model_cfg.get("name", "SimpleResNet")
    weights_path = model_cfg.get("weights_path", None)
    strict_load = model_cfg.get("strict_load", True)
    
    # 1. Tìm kiếm Class model trong module arch.py
    if hasattr(arch_module, model_name):
        model_cls = getattr(arch_module, model_name)
    else:
        raise AttributeError(f"Không tìm thấy model class '{model_name}' trong models/arch.py. "
                             f"Các class khả thi: {[name for name, _ in inspect.getmembers(arch_module, inspect.isclass)]}")

    # 2. Thu thập tham số khởi tạo (num_classes, in_channels, kwargs...)
    init_kwargs = {}
    if "num_classes" in model_cfg:
        init_kwargs["num_classes"] = model_cfg["num_classes"]
    if "in_channels" in model_cfg:
        init_kwargs["in_channels"] = model_cfg["in_channels"]
    
    # Bổ sung các kwargs khác nếu có trong config
    extra_kwargs = model_cfg.get("kwargs", {})
    if isinstance(extra_kwargs, dict):
        init_kwargs.update(extra_kwargs)

    # Khởi tạo mô hình
    print(f"[ModelBuilder] Đang khởi tạo mô hình '{model_name}' với các tham số: {init_kwargs}")
    model = model_cls(**init_kwargs)

    # 3. Nạp trọng số từ file .pth nếu được cung cấp
    if weights_path and weights_path.strip():
        if os.path.exists(weights_path):
            print(f"[ModelBuilder] Nạp trọng số pre-trained từ file: {weights_path}")
            checkpoint = torch.load(weights_path, map_location="cpu")
            
            # Xử lý trường hợp file .pth lưu dưới dạng dictionary checkpoint
            if isinstance(checkpoint, dict):
                if "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                elif "model" in checkpoint:
                    state_dict = checkpoint["model"]
                elif "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
                
            # Loại bỏ tiền tố 'module.' nếu model trước đó được lưu từ DataParallel
            clean_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace("module.", "") if k.startswith("module.") else k
                clean_state_dict[new_key] = v

            missing_keys, unexpected_keys = model.load_state_dict(clean_state_dict, strict=strict_load)
            if missing_keys:
                print(f"[ModelBuilder] Warning: Missing keys khi load weights: {missing_keys}")
            if unexpected_keys:
                print(f"[ModelBuilder] Warning: Unexpected keys khi load weights: {unexpected_keys}")
            print(f"[ModelBuilder] -> Nạp weights thành công!")
        else:
            print(f"[ModelBuilder] ERROR: Đường dẫn weights_path '{weights_path}' không tồn tại!")

    return model
