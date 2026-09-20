from torchvision import transforms

def build_transforms(config: dict):
    """
    Xây dựng các hàm biến đổi dữ liệu (Augmentation & Normalization) cho Train, Val, Test.
    """
    data_cfg = config.get("data", {})
    img_size = tuple(data_cfg.get("image_size", [224, 224]))

    train_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_test_transform
