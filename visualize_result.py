import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from data.dataset import CWRUDataset
import models
from config import opt
import matplotlib

# 设置绘图后端，防止 Windows 下弹窗卡死
matplotlib.use('Agg')

# 设置中文字体（如果需要显示中文）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def find_latest_checkpoint(ckpt_dir='checkpoints'):
    '''自动寻找最新的模型权重文件'''
    if not os.path.exists(ckpt_dir):
        return None
    files = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith('.pth')]
    if not files:
        return None
    # 按修改时间排序，取最后一个
    files.sort(key=os.path.getmtime)
    return files[-1]


def extract_features(model_path):
    '''提取测试集的特征层输出'''
    print(f"🔄 正在加载模型: {model_path}")
    # 动态加载模型结构
    model = getattr(models, opt.model)().eval()

    # 加载权重
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)

    device = torch.device(opt.device) if (opt.use_gpu and torch.cuda.is_available()) else torch.device('cpu')
    model = model.to(device)

    # 加载测试数据
    test_dataset = CWRUDataset(opt.test_data_root, train=False)
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)

    features = []
    labels = []

    print("⏳ 正在提取特征 (Feature Extraction)...")
    with torch.no_grad():
        for x, y in test_loader:
            x = x.float().unsqueeze(1).to(device)  # 调整维度 [Batch, 1, 400]

            # 前向传播
            _ = model(x)
            # 获取 BasicModule 中保存的中间层特征 (self.feature)
            feat = model.feature.cpu().numpy()

            features.append(feat)
            labels.append(y.numpy())

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    print(f"✅ 特征提取完成! 维度: {features.shape}")
    return features, labels


def plot_tsne(features, labels, save_path='results/tsne.png'):
    '''绘制 t-SNE 聚类图'''
    print("⏳ 正在计算 t-SNE (这可能需要几秒钟)...")
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    # 绘制散点图，使用不同颜色区分 10 类故障
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='tab10', alpha=0.7, s=30)

    # 添加图例
    legend1 = plt.legend(*scatter.legend_elements(), title="故障类别", loc="upper right")
    plt.gca().add_artist(legend1)

    plt.title("轴承故障特征 t-SNE 可视化 (准确率 98.8%)", fontsize=15)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True, linestyle='--', alpha=0.3)

    # 保存
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"🎉 t-SNE 图已保存至: {save_path}")
    plt.close()


def plot_confusion_matrix(excel_path, save_path='results/confusion_matrix.png'):
    '''绘制混淆矩阵热力图'''
    print(f"🔄 正在读取混淆矩阵: {excel_path}")
    if not os.path.exists(excel_path):
        print("❌ 错误: 找不到混淆矩阵 Excel 文件，请先运行 main.py 训练!")
        return

    df = pd.read_excel(excel_path, index_col=0)

    plt.figure(figsize=(10, 8))

    # 使用 Matplotlib 绘制热力图
    plt.imshow(df, interpolation='nearest', cmap='Blues')
    plt.title("故障诊断混淆矩阵", fontsize=15)
    plt.colorbar()

    tick_marks = np.arange(len(df.columns))
    plt.xticks(tick_marks, df.columns, rotation=45)
    plt.yticks(tick_marks, df.index)

    # 在格子里填数字
    thresh = df.values.max() / 2.
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            plt.text(j, i, format(df.values[i, j], 'd'),
                     horizontalalignment="center",
                     verticalalignment="center",
                     color="white" if df.values[i, j] > thresh else "black")

    plt.ylabel('真实标签 (True Label)', fontsize=12)
    plt.xlabel('预测标签 (Predicted Label)', fontsize=12)
    plt.tight_layout()

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"🎉 混淆矩阵图已保存至: {save_path}")
    plt.close()


if __name__ == '__main__':
    print("--- 开始可视化流程 ---")

    # 1. 自动寻找最新模型
    ckpt = find_latest_checkpoint()

    if ckpt:
        # 2. 提取特征并画 t-SNE
        feats, lbls = extract_features(ckpt)
        plot_tsne(feats, lbls)
    else:
        print("❌ 未找到模型文件 (.pth)，无法进行 t-SNE 可视化！")

    # 3. 画混淆矩阵 (依赖于训练生成的 Excel)
    plot_confusion_matrix(opt.result_file)

    print("--- 可视化结束，请去 results 文件夹查看图片 ---")