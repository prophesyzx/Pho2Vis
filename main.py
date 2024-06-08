import argparse
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from torch.utils.data import DataLoader
from torchvision import transforms
from load_data import ImageDataset
from model import DenseNet
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to best checkpoint (default: none)')
args = parser.parse_args()
def main(args):
    # 加载数据
    image_dir = "/kaggle/input/pho2vit/new_May_25/train_mask_rename"
    label_path = "/kaggle/input/pho2vit/new_May_25/train_labels.csv"
    
    
    transform1 = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transform2 = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    train_set = ImageDataset(image_dir, label_path, transform1)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)

    test_dir = "/kaggle/input/pho2vit/new_May_25/test_mask_rename"
    test_path = "/kaggle/input/pho2vit/new_May_25/test_labels.csv"
    test_set = ImageDataset(test_dir, test_path, transform2)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    # 加载预训练的DenseNet模型并修改为回归模型
    model = DenseNet().to(device)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), 1e-3,
                                momentum=0.9,
                                weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    num_epochs = 100
    best_loss = 10000000000000
    best_predictions = []
    train_losses = []
    test_losses = []
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        state_dict = checkpoint['model_state_dict']
        model.load_state_dict(state_dict)
        i = checkpoint['epoch']
        test_loss, all_predictions, all_targets = eval(test_loader, model, criterion, i)
        best_predictions = all_predictions
        with open("target_and_pediction.txt", "w", encoding="utf-8") as f:
            f.write(f"Target\tPrediction\n")
            all_targets = [x * (15000-2369.3) + 2369.3 for x in all_targets]
            best_predictions = [x * (15000-2369.3) + 2369.3 for x in best_predictions]
            for i in range(len(all_targets)):
                f.write(f"{all_targets[i]}\t{best_predictions[i]}\n")
    else:
        for i in range(num_epochs):
            train_loss,checkpoint = train(train_loader, model, criterion, optimizer, scheduler, i)

            test_loss, all_predictions, all_targets = eval(test_loader, model, criterion, i)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            if test_loss < best_loss:
                best_loss = test_loss
                best_predictions = all_predictions
                torch.save(checkpoint, "best_train_model.pth")

        with open("model_history.txt", "w", encoding='utf-8') as w:
            w.write(f"Epoch\tTrain Loss\tTest Loss\n")
            for i in range(num_epochs):
                w.write(f"{i}\t{train_losses[i]}\t{test_losses[i]}\n")
                
        with open("target_and_pediction.txt", "w", encoding="utf-8") as f:
            f.write(f"Target\tPrediction\n")
            all_targets = [x * (15000-2369.3) + 2369.3 for x in all_targets]
            best_predictions = [x * (15000-2369.3) + 2369.3 for x in best_predictions]
            for i in range(len(all_targets)):
                f.write(f"{all_targets[i]}\t{best_predictions[i]}\n")


def train(train_loader, model, criterion, optimizer, scheduler, epoch):
    running_loss = 0.0
    model.train()

    start_time = time.time()
    for i, (aod, rh, images, target) in enumerate(train_loader):
        images, target, rh, aod = images.to(device), target.to(device), rh.to(device), aod.to(device)
        output = model(aod, rh, images)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 累加总损失
        running_loss += loss.item()
        end_time = time.time()
        if (i + 1) % 5 == 0:
            total_time = end_time - start_time
            print(f"Epoch: {epoch} | Step: [{i+1}/{len(train_loader)}] | Loss: {loss.item()} | Elapsed Time: {total_time}s")

    scheduler.step()
    epoch_loss = running_loss / len(train_loader)
    epoch_time = time.time() - start_time
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': epoch_loss
    }
    print(
        f'{"*"*10} Epoch: [{epoch + 1}] | Loss: {epoch_loss} | Elapsed Time: {epoch_time}s {"*"*10}')

    return epoch_loss, checkpoint


def eval(test_loader, model, criterion, epoch):
    model.eval()

    with torch.no_grad():
        running_loss = 0.0
        all_predictions = []
        all_targets = []
        start_time = time.time()
        for i, (aod, rh, images, target) in enumerate(test_loader):
            images, target, rh, aod = images.to(device), target.to(device), rh.to(device), aod.to(device)
            output = model(aod, rh, images)
            loss = criterion(output, target)

            # 累加总损失
            running_loss += loss.item()
            
            all_predictions.append(output.cpu().numpy())
            all_targets.append(target.cpu().numpy())
            
            end_time = time.time()
            if (i + 1) % 5 == 0:
                total_time = end_time - start_time
                print(f"Testing Epoch: {epoch}... | Step: [{i+1}/{len(test_loader)}] | Loss: {loss.item()} | Elapsed Time: {total_time}s")

        epoch_loss = running_loss / len(test_loader)
        epoch_time = time.time() - start_time
        print(
            f'{"*"*10} Epoch: [{epoch + 1}] | Test Loss: {epoch_loss} | Elapsed Time: {epoch_time}s {"*"*10}')
        all_predictions = np.concatenate(all_predictions, axis=0).ravel()
        all_targets = np.concatenate(all_targets, axis=0).ravel()
        return epoch_loss, all_predictions.tolist(), all_targets.tolist()


if __name__ == '__main__':
    main(args)
