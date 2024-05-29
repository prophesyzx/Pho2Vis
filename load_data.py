import torch
import torch.nn
import os
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd


class ImageDataset(Dataset):
    def __init__(self, image_dir: str, label_path: str, transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    ) -> None:
        self.image_dir = image_dir
        self.label_path = label_path
        self.transform = transform
        self.image_paths = [os.path.join(image_dir, img)
                            for img in os.listdir(image_dir)]
        df = pd.read_csv(label_path)
        labels = [i for i in df["Visibility"].values.tolist()for _ in range(2)]
        rh = [i for i in df["RH"].values.tolist() for _ in range(2)]
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
        self.RH = torch.tensor(rh, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        rh = self.RH[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label, rh


if __name__ == '__main__':
    image_dir = "new_May_25/train_mask_rename"
    label_path = "new_May_25/train_labels.csv"
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    train_set = ImageDataset(image_dir, label_path, transform)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
