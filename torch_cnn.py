import os, sys, glob, argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

import time, datetime
import pdb, traceback

import cv2
from PIL import Image
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
# timm: pytorch image models, SOTA models
import timm

from joblib import dump, load

train_jpg = glob.glob('./train/*/*')
np.random.shuffle(train_jpg)
train_jpg = np.array(train_jpg)

if os.path.exists('train_jpg.pkl'):
    train_jpg_dict = load('train_jpg.pkl')
else:
    train_jpg_dict = {}
    for path in tqdm(train_jpg):
        # 加入resize
        train_jpg_dict[path] = Image.open(path).convert('RGB')
    dump(train_jpg_dict, 'train_jpg.pkl')


class QRDataset(Dataset):
    def __init__(self, img_path, transform=None):
        self.img_path = img_path
        self.transform = transform

    def __getitem__(self, index):
        start_time = time.time()
        # img = Image.open(self.img_path[index]).convert('RGB')

        if self.img_path[index] in train_jpg_dict:
            img = train_jpg_dict[self.img_path[index]]
        else:
            img = Image.open(self.img_path[index]).convert('RGB')

        lbl_dict = {'angry': 0, 'disgusted': 1, 'fearful': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprised': 6}

        if self.transform is not None:
            img = self.transform(img)

        if 'test' in self.img_path[index]:
            return img, torch.from_numpy(np.array(0))
        else:
            lbl_int = lbl_dict[self.img_path[index].split('\\')[-2]]
            return img, torch.from_numpy(np.array(lbl_int))

    def __len__(self):
        return len(self.img_path)


class XunFeiNet(nn.Module):
    def __init__(self):
        super(XunFeiNet, self).__init__()

        #         model = models.resnet18(True)
        #         model.avgpool = nn.AdaptiveAvgPool2d(1)
        #         model.fc = nn.Linear(512, 7)

        model = timm.create_model('efficientnet_b1', pretrained='imagenet')
        model.classifier = nn.Linear(1280, 7)
        self.model = model

    def forward(self, img):
        out = self.model(img)
        return out


def validate(val_loader, model, criterion):
    model.eval()
    acc1 = []
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()

            output = model(input)
            loss = criterion(output, target)
            acc1.append((output.argmax(1) == target).float().mean().item())

        print(' * Val Acc@1 {0}'.format(np.mean(acc1)))
        return np.mean(acc1)


def predict(test_loader, model, tta=10):
    model.eval()

    test_pred_tta = None
    for _ in range(tta):
        test_pred = []
        with torch.no_grad():
            end = time.time()
            for i, (input, target) in enumerate(test_loader):
                input = input.cuda()
                target = target.cuda()

                output = model(input)
                output = output.data.cpu().numpy()

                test_pred.append(output)
        test_pred = np.vstack(test_pred)

        if test_pred_tta is None:
            test_pred_tta = test_pred
        else:
            test_pred_tta += test_pred

    return test_pred_tta


def train(train_loader, model, criterion, optimizer, epoch):
    model.train()

    end = time.time()
    acc1 = []
    for i, (input, target) in enumerate(train_loader):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        output = model(input)
        loss = criterion(output, target)

        acc1.append((output.argmax(1) == target).float().mean().item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print('Train: {0}'.format(np.mean(acc1)))


if __name__ == '__main__':
    skf = KFold(n_splits=10, random_state=233, shuffle=True)
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_jpg, train_jpg)):

        train_loader = torch.utils.data.DataLoader(
            QRDataset(train_jpg[train_idx][:],
                      transforms.Compose([
                          transforms.RandomAffine(10),
                          transforms.ColorJitter(hue=.05, saturation=.05),
                          transforms.RandomHorizontalFlip(),
                          transforms.RandomVerticalFlip(),
                          transforms.Resize((196, 196)),
                          transforms.ToTensor(),
                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                      ])
                      ), batch_size=20, shuffle=True, num_workers=0, pin_memory=True
        )

        val_loader = torch.utils.data.DataLoader(
            QRDataset(train_jpg[val_idx][:],
                      transforms.Compose([
                          transforms.Resize((196, 196)),
                          #                             transforms.Resize((256, 256)),
                          # transforms.Resize((124, 124)),
                          # transforms.RandomCrop((88, 88)),
                          transforms.ToTensor(),
                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                      ])
                      ), batch_size=10, shuffle=False, num_workers=0, pin_memory=True
        )

        model = XunFeiNet().cuda()
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.SGD(model.parameters(), 0.01)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.75)
        best_acc = 0.0
        for epoch in range(40):
            print('\nEpoch: ', epoch)

            train(train_loader, model, criterion, optimizer, epoch)
            val_acc = validate(val_loader, model, criterion)
            scheduler.step()

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), './resnet18_fold{0}.pt'.format(fold_idx))

        break






















