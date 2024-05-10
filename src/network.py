# Python packages
from termcolor import colored
from typing import Dict
import copy
import torch.nn as nn
import torch.optim as optim

# PyTorch & Pytorch Lightning
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from torch import nn
from torchvision import models ,datasets , transforms
from torchvision.models.alexnet import AlexNet
import torch

# Custom packages
from src.metric import MyAccuracy
from src.metric import MyF1Score
import src.config as cfg
from src.util import show_setting


# [TODO: Optional] Rewrite this class if you want
class MyNetwork(AlexNet):
    def __init__(self, num_classes = 200):
        super(AlexNet , self).__init__()
 
        # [TODO] Modify feature extractor part in AlexNet
        # 특징 추출
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),  # 배치 정규화 추가
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(192),  # 배치 정규화 추가
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(384),  # 배치 정규화 추가
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),  # 배치 정규화 추가
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),  # 배치 정규화 추가
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(4096),  # 배치 정규화 추가
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(4096),  # 배치 정규화 추가
            nn.Linear(4096, num_classes),
        )
       def forward(self, x: torch.Tensor) -> torch.Tensor:
          [TODO: Optional] Modify this as well if you want
          x = self.features(x)
          x = self.avgpool(x)
          x = torch.flatten(x, 1)
          x = self.classifier(x)
          return x
    
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=200):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 16 * 16, 384)
        self.fc2 = nn.Linear(384, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(model, device, train_loader, optimizer, epoch, l2_alpha=0.05):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
        
        # L2 공간 유지 손실 추가
        l2_loss = sum(p.pow(2.0).sum() for p in model.parameters())
        loss += l2_alpha * l2_loss

        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                  f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 데이터셋 증강
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ])
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    
    model = SimpleCNN(num_classes=200).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(1, 11):
        train(model, device, train_loader, optimizer, epoch)

class SimpleClassifier(LightningModule):
    def __init__(self,
                 model_name: str = 'resnet18',
                 num_classes: int = 200,
                 optimizer_params: Dict = dict(),
                 scheduler_params: Dict = dict(),
        ):
        super().__init__()

        # Network
        if model_name == 'MyNetwork':
            self.model = MyNetwork()
        elif model_name == 'SimpleCNN':
            self.model = SimpleCNN()
        else:
            models_list = models.list_models()
            assert model_name in models_list, f'Unknown model name: {model_name}. Choose one from {", ".join(models_list)}'
            self.model = models.get_model(model_name, num_classes=num_classes)

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # Metric
        self.accuracy = MyAccuracy()
        self.f1score = MyF1Score()

        # Hyperparameters
        self.save_hyperparameters()

    def on_train_start(self):
        show_setting(cfg)

    def configure_optimizers(self):
        optim_params = copy.deepcopy(self.hparams.optimizer_params)
        optim_type = optim_params.pop('type')
        optimizer = getattr(torch.optim, optim_type)(self.parameters(), **optim_params)

        scheduler_params = copy.deepcopy(self.hparams.scheduler_params)
        scheduler_type = scheduler_params.pop('type')
        scheduler = getattr(torch.optim.lr_scheduler, scheduler_type)(optimizer, **scheduler_params)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch)
        self.f1score.update(scores, y)
        f1 = self.f1score.compute()
        accuracy = self.accuracy(scores, y)
        self.log_dict({'loss/train': loss, 'accuracy/train': accuracy,'f1score/train':f1},
                      on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch)
        accuracy = self.accuracy(scores, y)
        self.f1score.update(scores, y)
        f1 = self.f1score.compute()
        self.log_dict({'loss/val': loss, 'accuracy/val': accuracy,'f1score/val': f1},
                      on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self._wandb_log_image(batch, batch_idx, scores, frequency = cfg.WANDB_IMG_LOG_FREQ)

    def _common_step(self, batch):
        x, y = batch
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        return loss, scores, y

    def _wandb_log_image(self, batch, batch_idx, preds, frequency = 100):
        if not isinstance(self.logger, WandbLogger):
            if batch_idx == 0:
                self.print(colored("Please use WandbLogger to log images.", color='blue', attrs=('bold',)))
            return

        if batch_idx % frequency == 0:
            x, y = batch
            preds = torch.argmax(preds, dim=1)
            self.logger.log_image(
                key=f'pred/val/batch{batch_idx:5d}_sample_0',
                images=[x[0].to('cpu')],
                caption=[f'GT: {y[0].item()}, Pred: {preds[0].item()}'])
