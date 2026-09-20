import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from data.transforms import build_transforms

class SyntheticDataset(Dataset):
    """
    Dataset giả lập ngẫu nhiên dùng để test pipeline nhanh chóng không cần tải dữ liệu ngoài.
    """
    def __init__(self, num_samples=200, in_channels=3, img_size=(224, 224), num_classes=10, transform=None):
        self.num_samples = num_samples
        self.in_channels = in_channels
        self.img_size = img_size
        self.num_classes = num_classes
        self.transform = transform
        
        # Sinh dữ liệu ngẫu nhiên sẵn
        self.data = torch.randn(num_samples, in_channels, *img_size)
        self.targets = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]
        return x, y


class ImageFolderDataset(Dataset):
    """
    Dataset tổng quát đọc dữ liệu ảnh từ thư mục phân lớp (ImageFolder structure).
    Cấu trúc: path/class_name/image.jpg
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.classes = []

        if os.path.exists(root_dir):
            self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
            class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
            
            for cls_name in self.classes:
                cls_dir = os.path.join(root_dir, cls_name)
                for root, _, files in os.walk(cls_dir):
                    for file in files:
                        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                            path = os.path.join(root, file)
                            self.samples.append((path, class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if len(self.samples) == 0:
            raise RuntimeError(f"Thư mục '{self.root_dir}' không chứa hình ảnh hợp lệ nào!")
        path, target = self.samples[idx]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)

        return img, target


def build_dataloaders(config: dict):
    """
    Tạo DataLoaders cho Train, Val, và Test dựa trên config.yaml
    """
    data_cfg = config.get("data", {})
    dataset_type = data_cfg.get("dataset_type", "synthetic")
    batch_size = data_cfg.get("batch_size", 32)
    num_workers = data_cfg.get("num_workers", 2)
    pin_memory = data_cfg.get("pin_memory", True)
    
    train_transform, val_test_transform = build_transforms(config)

    if dataset_type == "synthetic":
        print("[DataLoader] Đang sử dụng SyntheticDataset giả lập để chạy thử nghiệm...")
        in_ch = config.get("model", {}).get("in_channels", 3)
        img_sz = tuple(data_cfg.get("image_size", [224, 224]))
        num_cls = config.get("model", {}).get("num_classes", 10)

        train_dataset = SyntheticDataset(num_samples=300, in_channels=in_ch, img_size=img_sz, num_classes=num_cls)
        val_dataset   = SyntheticDataset(num_samples=100, in_channels=in_ch, img_size=img_sz, num_classes=num_cls)
        test_dataset  = SyntheticDataset(num_samples=100, in_channels=in_ch, img_size=img_sz, num_classes=num_cls)
    
    elif dataset_type == "image_folder":
        train_path = data_cfg.get("train_path", "data/train")
        val_path   = data_cfg.get("val_path", "data/val")
        test_path  = data_cfg.get("test_path", "data/test")

        train_dataset = ImageFolderDataset(train_path, transform=train_transform)
        val_dataset   = ImageFolderDataset(val_path, transform=val_test_transform)
        test_dataset  = ImageFolderDataset(test_path, transform=val_test_transform)
    else:
        raise ValueError(f"dataset_type '{dataset_type}' chưa được hỗ trợ. Vui lòng chọn 'synthetic' hoặc 'image_folder'")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    print(f"[DataLoader] Hoàn tất tạo DataLoaders: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)} samples.")
    return train_loader, val_loader, test_loader
