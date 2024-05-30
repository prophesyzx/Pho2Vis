import torch
import torch.nn
import os
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
from sklearn.impute import KNNImputer


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
                            for img in sorted(os.listdir(image_dir))]
        df = pd.read_csv(label_path)

        # 初始化 KNN 插补器
        imputer = KNNImputer(n_neighbors=10)
        # 对数据进行插补
        df[['AOD']] = imputer.fit_transform(df[['AOD']])
        df['Visibility'] = df['Visibility'].ffill()
        df['RH'] = df['RH'].ffill()

        labels = self._normalize(df["Visibility"].values.tolist(), 'Visibility')
        rh = self._normalize(df["RH"].values.tolist(), 'RH')
        aod = self._normalize(df["AOD"].values.tolist(), 'AOD')

        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
        self.RH = torch.tensor(rh, dtype=torch.float32).unsqueeze(1)
        self.AOD = torch.tensor(aod, dtype=torch.float32).unsqueeze(1)
        
        # 确保没有 NaN 值
        assert not torch.isnan(self.labels).any(), "Labels contain NaN"
        assert not torch.isnan(self.RH).any(), "RH contains NaN"
        assert not torch.isnan(self.AOD).any(), "AOD contains NaN"
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path_1 = self.image_paths[2*idx]
        image_path_2 = self.image_paths[2*idx + 1]
        image_1 = Image.open(image_path_1).convert('RGB')
        image_2 = Image.open(image_path_2).convert('RGB')
        
        aod = self.AOD[idx]
        rh = self.RH[idx]
        label = self.labels[idx]

        if self.transform:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        
        image = torch.cat((image_1, image_2), dim=2)

        return aod, rh, image, label
    
    def _normalize(self,data,type):
        min_max_dict = {'Visibility':(2369.3, 15000.0),
                        'RH':(5.0, 99.0),
                        'AOD':(0.041306392, 1.58347051)}
        X_max = min_max_dict[type][1]
        X_min = min_max_dict[type][0]
        normalized_list = [(x - X_min) / (X_max - X_min) for x in data]
        return normalized_list

