import argparse
import math
import os
import random
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR, CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import torch.nn.functional as F
from load_data import ImageDataset
from MyModel import DenseNetWithRH


def main(args):
    # 加载数据
    image_dir = "new_May_25/train_mask_rename"
    label_path = "new_May_25/train_labels.csv"
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    train_set = ImageDataset(image_dir, label_path, transform)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)

    test_dir = ""
    test_path = ""
    test_set = ImageDataset(test_dir, test_path)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    # 加载预训练的DenseNet模型并修改为回归模型
    model = DenseNetWithRH()

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 定义学习率调度器（余弦退火）
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    num_epochs = args.strat_epoch + args.epoch
    for i in range(args.strat_epoch, num_epochs):
        train_loss = train(train_loader, model, criterion,optimizer, i, args)

        test_loss = eval(test_loader, model, criterion, args)


def train(train_loader, model, criterion, optimizer, scheduler, epoch, args, **kwargs):
    running_loss = 0.0
    total_sample = 0
    model.train()

    start_time = time.time()
    for i, (images, target, _) in enumerate(train_loader):
        step_time = time.time()
        output = model(images)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # 累加总损失
        running_loss += loss.item() * images.size(0)
        total_sample += images.size(0)
        end_time = time.time()
        if (i + 1) % 10 == 0:
            avg_loss = running_loss / total_sample
            step_time = end_time - step_time
            total_time = end_time - start_time
            print(f"Epoch: {epoch}|Step: [{i+1}/{len(train_loader)}]|Loss: {loss.item():.6f}|Average Loss: {avg_loss:.6f}"
                  f"|Step Time: {step_time:.3f}s|Total Time: {total_time:.3f}s")

    epoch_loss = running_loss / len(train_loader)
    epoch_time = time.time() - start_time
    print(
        f'Epoch: [{epoch + 1}/{args.num_epochs}]|Total Loss: {epoch_loss:.6f}|Elapsed Time: {epoch_time:.3f}s')

    return epoch_loss


def eval(test_loader, model, criterion, args, **kwargs):
    model.eval()

    with torch.no_grad():
        running_loss = 0.0
        total_sample = 0
        start_time = time.time()
        for i, (images, target, _) in enumerate(test_loader):
            step_time = time.time()
            output = model(images)
            loss = criterion(output, target)

            # 累加总损失
            running_loss += loss.item() * images.size(0)
            total_sample += images.size(0)
            end_time = time.time()
            if (i + 1) % 10 == 0:
                avg_loss = running_loss / total_sample
                step_time = end_time - step_time
                total_time = end_time - start_time
                print(f"Testing...|Step: [{i+1}/{len(test_loader)}]|Loss: {loss.item():.6f}|Average Loss: {avg_loss:.6f}"
                      f"|Step Time: {step_time:.3f}s|Total Time: {total_time:.3f}s")

        epoch_loss = running_loss / len(test_loader)
        epoch_time = time.time() - start_time
        print(
            f'{"*"*10} Total Loss: {epoch_loss:.6f}|Elapsed Time: {epoch_time:.3f}s {"*"*10}')
        return epoch_loss
