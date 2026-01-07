#!/usr/bin/env python
# @Time    : 2020/7/8 16:07
# @Author  : wb
# @File    : main.py

import os

# 1. 强制解决 OMP 冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 2. 获取当前脚本的绝对路径，确保文件夹一定建对位置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(BASE_DIR, 'checkpoints')
RESULT_DIR = os.path.join(BASE_DIR, 'results')

if not os.path.exists(CKPT_DIR):
    os.makedirs(CKPT_DIR)
    print(f"✅ 绝对路径文件夹已创建: {CKPT_DIR}")
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

import h5py
from config import opt
from torch.utils.data import DataLoader
from data.dataset import CWRUDataset
import models
import torch
from tensorboardX import SummaryWriter
import copy
import time
import pandas as pd
import numpy as np


def train():
    # step1: 模型
    model = getattr(models, opt.model)()
    if opt.use_gpu: model.cuda()

    # step2: 数据
    train_data = CWRUDataset(opt.train_data_root, train=True)
    val_data = CWRUDataset(opt.val_data_root, train=False)

    train_dataloader = DataLoader(train_data, opt.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, opt.batch_size, shuffle=False)

    # step3: 目标函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=opt.weight_decay)

    # 学习率调整
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.lr_decay_iters, opt.lr_decay)

    writer = SummaryWriter()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    device = torch.device(opt.device) if (opt.use_gpu and torch.cuda.is_available()) else torch.device('cpu')
    print(f'Using device: {device}')
    print(f'Initial Learning Rate: {lr}')

    # train
    for epoch in range(opt.max_epoch):
        start_time = time.time()
        model.train()

        for ii, (data, label) in enumerate(train_dataloader):
            data.resize_(data.size()[0], 1, data.size()[1])
            data, label = data.float(), label.long()
            input, target = data.to(device), label.to(device)

            optimizer.zero_grad()
            score = model(input)
            loss = criterion(score, target)
            writer.add_scalar('loss', loss.item())

            if (ii + 1) % opt.print_every == 0:
                print('Batch %d, loss = %.4f' % (ii + 1, loss.item()))

            loss.backward()
            optimizer.step()

        scheduler.step()

        # 计算准确率
        train_acc, _ = check_accuracy(model, train_dataloader, device)
        val_acc, _ = check_accuracy(model, val_dataloader, device)

        writer.add_scalars('acc', {'train_acc': train_acc, 'val_acc': val_acc}, epoch)

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        end_time = time.time()
        print('Epoch %d/%d Done. Train Acc: %.2f%%, Val Acc: %.2f%%, Time: %.2fs' %
              (epoch + 1, opt.max_epoch, 100 * train_acc, 100 * val_acc, end_time - start_time))

    print('Training finished. Best Val Acc: %.2f%%' % (100 * best_acc))

    # 加载最佳模型
    model.load_state_dict(best_model_wts)

    # 验证并保存 Excel
    val_acc, confuse_matrix = check_accuracy(model, val_dataloader, device, error_analysis=True)
    data_pd = pd.DataFrame(confuse_matrix)
    writer = pd.ExcelWriter(opt.result_file)
    data_pd.to_excel(writer)
    writer.close()
    print(f"Results saved to {opt.result_file}")

    # === 3. 暴力保存模型（直接用绝对路径，绕过 BasicModule）===
    timestamp = time.strftime('%m%d_%H_%M_%S')
    save_name = f'{opt.model}_{timestamp}.pth'
    # 拼接绝对路径
    final_save_path = os.path.join(CKPT_DIR, save_name)

    try:
        torch.save(model.state_dict(), final_save_path)
        print(f"✅ 模型已强制保存至: {final_save_path}")
    except Exception as e:
        print(f"❌ 保存依然失败，报错信息: {e}")


def test():
    model = getattr(models, opt.model)().eval()
    if opt.load_model_path and os.path.exists(opt.load_model_path):
        model.load(opt.load_model_path)

    test_dataset = CWRUDataset(opt.test_data_root, train=False)
    test_loader = DataLoader(test_dataset, opt.batch_size, shuffle=False)

    device = torch.device(opt.device) if (opt.use_gpu and torch.cuda.is_available()) else torch.device('cpu')
    model = model.to(device)

    f = h5py.File(opt.feature_filename, 'w')
    f.create_dataset('y_train', data=test_dataset.y)
    check_accuracy(model, test_loader, device, feature_file=f, error_analysis=True)
    f.close()
    print("Feature extraction finished.")


def check_accuracy(model, loader, device, feature_file=None, error_analysis=False):
    X_feature = np.empty([0, 36])
    ys = np.array([])
    y_preds = np.array([])
    confuse_matrix = None
    num_correct = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x.resize_(x.size()[0], 1, x.size()[1])
            x, y = x.float(), y.long()
            x, y = x.to(device), y.to(device)

            scores = model(x)
            preds = scores.max(1, keepdim=True)[1]
            num_correct += preds.eq(y.view_as(preds)).sum().item()

            if error_analysis:
                ys = np.append(ys, np.array(y.cpu()))
                y_preds = np.append(y_preds, np.array(preds.cpu()))

            feature_output = model.feature.cpu()
            X_feature = np.concatenate((X_feature, feature_output), axis=0)

        if feature_file is not None:
            feature_file.create_dataset('X_train', data=X_feature)

    acc = float(num_correct) / len(loader.dataset)
    if error_analysis:
        confuse_matrix = pd.crosstab(y_preds, ys, margins=True)

    return acc, confuse_matrix


if __name__ == '__main__':
    train()