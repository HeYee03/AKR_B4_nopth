#########################
" 仅产生混淆矩阵图 和 打印评价  "
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from data import DataGenerator  # 请替换为你的数据生成模块    ????
from net import vgg16 # 请替换为你的模型定义模块
from data import *
# 加载模型结构
device = torch.device("cpu")
net = vgg16(True, progress=True, num_classes=2)  # 请替换为你的模型结构
net.to(device)

# 加载模型权重
model_weights_path = "./B4model/model/AKR_B4_12_1915.pth"   # 请替换为你的.pth或.pt文件路径
net.load_state_dict(torch.load(model_weights_path, map_location=device))
net.eval()

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
all_predictions = []

for data in gen_test:
    img, label = data
    with torch.no_grad():
        img = img.to(device)
        label = label.to(device)
        out = net(img)
        all_labels.extend(label.cpu().numpy())
        all_predictions.extend(out.argmax(1).cpu().numpy())

# 计算混淆矩阵和评价指标
conf_matrix = confusion_matrix(all_labels, all_predictions)
accuracy = accuracy_score(all_labels, all_predictions)
precision = precision_score(all_labels, all_predictions, average='weighted')
recall = recall_score(all_labels, all_predictions, average='weighted')
f1 = f1_score(all_labels, all_predictions, average='weighted')

# 打印评价指标
print("混淆矩阵:")
print(conf_matrix)
print("准确度:", accuracy)
print("精确度:", precision)
print("召回率:", recall)
print("F1分数:", f1)

# 可视化混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
