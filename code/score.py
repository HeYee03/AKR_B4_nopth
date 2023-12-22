" 评价 可用  F1 recall等图"

import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_curve, f1_score, accuracy_score
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from data import DataGenerator  # 请替换为你的数据生成模块
from net import vgg16  # 请替换为你的模型定义模块
import numpy as np

# 加载模型结构
device = torch.device("cpu")
net = vgg16( True, progress=True, num_classes=2)  # 请替换为你的模型结构
net.to(device)

# 加载模型权重
model_weights_path = "./B4model/model/AKR_B4_12_1915.pth"  # 请替换为你的.pth或.pt文件路径
net.load_state_dict(torch.load(model_weights_path, map_location=device))
net.eval()

# 加载数据并进行推断
# 3. 数据处理
annotation_path = './B4model/B4_train.txt'
with open(annotation_path, 'r') as f:
    lines = f.readlines()

np.random.seed(10101)
np.random.shuffle(lines)
np.random.seed(None)
num_val = int(len(lines) * 0.2)
num_train = len(lines) - num_val
input_shape = [224, 224]
val_data = DataGenerator(lines[num_train:], input_shape, False)
val_len = len(val_data)
#
gen_test = DataLoader(val_data, batch_size=4)

all_labels = []
all_probabilities = []

for data in gen_test:
    img, label = data
    with torch.no_grad():
        img = torch.tensor(img, dtype=torch.float32).to(device)  # Convert NumPy array to PyTorch tensor
        label = label.to(device)
        out = net(img)

        probabilities = torch.softmax(out, dim=1)[:, 1].cpu().numpy()

        all_probabilities.extend(probabilities)
        all_labels.extend(label.cpu().numpy())

# Calculate F1, Recall, Precision, and Accuracy
precision, recall, thresholds = precision_recall_curve(all_labels, all_probabilities)
f1 = [2 * p * r / (p + r) if (p + r) > 0 else 0 for p, r in zip(precision, recall)]
accuracy = [accuracy_score(all_labels, np.array(all_probabilities) > threshold) for threshold in thresholds]

# Ensure all arrays have the same length
min_length = min(len(thresholds), len(precision), len(recall), len(f1), len(accuracy))
thresholds = thresholds[:min_length]
precision = precision[:min_length]
recall = recall[:min_length]
f1 = f1[:min_length]
accuracy = accuracy[:min_length]

# Plot the graphs
plt.figure(figsize=(10, 6))
plt.plot(thresholds, precision, label='Precision', marker='o')
plt.plot(thresholds, recall, label='Recall', marker='o')
plt.plot(thresholds, f1, label='F1 Score', marker='o')
plt.plot(thresholds, accuracy, label='Accuracy', marker='o')
plt.title('Precision, Recall, F1 Score, and Accuracy vs Confidence Threshold')
plt.xlabel('Confidence Threshold')
plt.ylabel('Metrics Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()