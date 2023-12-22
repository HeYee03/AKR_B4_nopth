import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from net import vgg16
from torch.utils.data import DataLoader#工具取黑盒子，用函数来提取数据集中的数据（小批次）
from data import *


def main():
    # ... [之前的代码]
    '''数据集'''
    annotation_path = 'cls_cat_dog_test.txt'  # 读取数据集生成的文件
    with open(annotation_path, 'r') as f:
        lines = f.readlines()
    np.random.seed(10101)  # 函数用于生成指定随机数
    np.random.shuffle(lines)  # 数据打乱
    np.random.seed(None)
    num_val = int(len(lines) * 0.2)  # 十分之一数据用来测试
    num_train = len(lines) - num_val  # 输入图像大小
    input_shape = [224, 224]  # 导入图像大小
    train_data = DataGenerator(lines[:num_train], input_shape, True)
    val_data = DataGenerator(lines[num_train:], input_shape, False)
    val_len = len(val_data)
    print(val_len)  # 返回测试集长度
    # 取黑盒子工具
    """加载数据"""
    gen_train = DataLoader(train_data, batch_size=4)  # 训练集batch_size读取小样本，规定每次取多少样本
    gen_test = DataLoader(val_data, batch_size=4)  # 测试集读取小样本
    # '''构建网络'''
    device = torch.device("cpu")  # device=torch.device('cuda'if torch.cuda.is_available() else "cpu")#电脑主机的选择
    net = vgg16(True, progress=True, num_classes=2)  # 定于分类的类别
    net.to(device)
    '''选择优化器和学习率的调整方法'''
    lr = 0.0001  # 定义学习率
    optim = torch.optim.Adam(net.parameters(), lr=lr)  # 导入网络和学习率
    sculer = torch.optim.lr_scheduler.StepLR(optim, step_size=1)  # 步长为1的读取'''训练'''
    epochs = 1  # 读取数据次数，每次读取顺序方式不同

    # 初始化用于记录的列表
    train_losses = []
    test_losses = []
    test_accuracies = []

    # 训练过程
    for epoch in range(epochs):
        #net.train()
        #total_train_loss = 0
        #for img, label in gen_train:
            # ... [训练步骤代码]
            #img, label = data

        total_train_loss = 0  # 定义总损失
        for data in gen_train:
            img, label = data
            with torch.no_grad():
                img = img.to(device)
                label = label.to(device)
            optim.zero_grad()
            output = net(img)
            train_loss = nn.CrossEntropyLoss()(output, label).to(device)
            train_loss.backward()  # 反向传播
            optim.step()  # 优化器更新
            total_train_loss += train_loss  # 损失相加
        sculer.step()
        total_test = 0  # 总损失
        total_accuracy = 0  # 总精度
        #scheduler.step()

        # 验证过程
        net.eval()
        total_test_loss = 0
        total_accuracy = 0
        #with torch.no_grad():
        #    for img, label in gen_test:
                # ... [验证步骤代码]
        for data in gen_test:
            img, label = data  # 图片转数据
            with torch.no_grad():
                img = img.to(device)
                label = label.to(device)
                optim.zero_grad()  # 梯度清零
                out = net(img)  # 投入网络
                test_loss = nn.CrossEntropyLoss()(out, label).to(device)
                total_test += test_loss  # 测试损失，无反向传播
                accuracy = ((out.argmax(1) == label).sum()).clone().detach().cpu().numpy()  # 正确预测的总和比测试集的长度，即预测正确的精度
                total_accuracy += accuracy

        test_loss, test_accuracy = test(VGG, test_loader, criterion)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        # 记录损失和准确率
        avg_train_loss = total_train_loss / len(gen_train)
        avg_test_loss = total_test_loss / len(gen_test)
        avg_accuracy = total_accuracy / len(val_data)

        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
        test_accuracies.append(avg_accuracy)

        # 打印损失和精度
        # ... [打印代码]
        print("训练集上的损失：{}".format(total_train_loss))
        print("测试集上的损失：{}".format(total_test))
        print("测试集上的精度：{:.1%}".format(total_accuracy / val_len))  # 百分数精度，正确预测的总和比测试集的长度
        torch.save(net.state_dict(), "AKR_A4{}.pth".format(epoch + 1))
        print("模型已保存")

    # 绘图
    # 绘图
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot([item.detach().numpy() for item in train_losses], label='Train Loss')
    plt.plot([item.detach().numpy() for item in test_losses], label='Test Loss')
    plt.title('Training and Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot([item.detach().numpy() for item in test_accuracies], label='Test Accuracy')
    plt.title('Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def test(model, dataloader, criterion):
    model.eval()  # 将模型设置为评估模式，这会影响例如 dropout 层的行为

    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # 在评估模式下，不需要计算梯度
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total
    average_loss = test_loss / len(dataloader)

    return average_loss, accuracy


if __name__ == "__main__":
    main()
