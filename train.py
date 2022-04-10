import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import torch
import time
import copy

"""
数据分析：
预测脑信号的类别
数据格式：csv
总共有4类（0，1，2，3）
每组数据包含205个[0,1]的值
"""


# 加载原始数据
# df_test就是原始的表格数据，第一列是id，第二列是编号，第三例是信号值
df_train = pd.read_csv('./datasets/train.csv')
df_testA = pd.read_csv('./datasets/testA.csv')
# 查看训练和测试数据的前五条
print('-----1------')
print(df_train.head())
print('-----2------')
print(df_testA.head())
# 检查数据是否有NAN数据
print('-----3------')
print(df_train.isna().sum(), df_testA.isna().sum())
# 确认标签的类别及数量
print('-----4------')
print(df_train['label'].value_counts())
# 查看训练数据集特征
print('-----5------')
print(df_train.describe())
# 查看数据集信息
print('-----6------')
print(df_train.info())


# 绘制每种类别的折线图
ids = []
for id, row in df_train.groupby('label').apply(lambda x: x.iloc[2]).iterrows():
    ids.append(int(id))
    signals = list(map(float, row['heartbeat_signals'].split(',')))
    sns.lineplot(data=signals)

# plt.legend(ids)
# plt.show()

# 原始数据信息整理:
# 1.主要特征数据为1维信号振幅，总长度为205。（已经归一化到0～1了）
# 2.除波形数据外无其他辅助和先验信息
# 3.波形数据为float64格式
# 4.没有缺失值，无需填充。（未采集到的数据默认为0，故无缺失数据）
# 5.非表格数据更适合用神经网络处理


# 加载原始数据
class MyData(Data.Dataset):
    def __init__(self, feature, label):
        self.feature = feature # 特征
        self.label = label # 标签

    def __len__(self):
        return len(self.feature)

    def __getitem__(self, idx):
        return self.feature[idx], self.label[idx]


def load_data(batch_size):
    print("----------------训练开始----------------------")
    # 加载原始数据
    df_train = pd.read_csv('./datasets/train.csv')
    # 拆解heartbeat_signals
    # test_signals就是一行数据放到一个数组里，这个数组里有205个数字。总共有2000行数据，就是2000个数组。test_signals.shape:  (20000,)
    train_signals = np.array(df_train['heartbeat_signals'].apply(lambda x: np.array(list(map(float, x.split(','))), dtype=np.float32)))
    train_labels = np.array(df_train['label'].apply(lambda x: float(x)), dtype=np.float32)
    # 构建pytorch数据类
    train_data = MyData(train_signals, train_labels)
    # 构建pytorch数据集Dataloader
    train_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    return train_data, train_loader


def loss_curve(list_loss, list_acc):
    epochs = np.arange(1, len(list_loss) + 1)
    fig, ax = plt.subplots()
    ax.plot(epochs, list_loss, label='loss')
    ax.plot(epochs, list_acc, label='accuracy')
    ax.set_xlabel('epoch')
    ax.set_ylabel('%')
    ax.set_title('loss & accuray ')
    ax.legend()


class model_CNN_1(nn.Module):
    def __init__(self):
        super(model_CNN_1, self).__init__()
        self.conv_unit = nn.Sequential(
            nn.BatchNorm1d(1),
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=11, stride=1, padding=5),
            nn.LeakyReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=11, stride=1, padding=5),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(4),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(0.1),
        )
        self.dense_unit = nn.Sequential(
            nn.Linear(3072, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 4),
            nn.Softmax(dim=1)
        )
        self.fc = nn.Linear(4, 4)

    def forward(self, inputs):
        inputs = inputs.view(inputs.size()[0], 1, inputs.size()[1])
        inputs = self.conv_unit(inputs)
        inputs = inputs.view(inputs.size()[0], -1)
        inputs = self.dense_unit(inputs)
        inputs = self.fc(inputs)
        return inputs


num_epochs = 50
batch_size = 64
train_data, train_loader = load_data(batch_size)
# 定义模型、loss function
model = model_CNN_1()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2)


# 训练
def train_model():
    t1 = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # 循环50个epoch进行数据训练
    list_loss, list_acc = [], []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            predictions = model(inputs)
            loss = criterion(predictions, labels.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size()[0]
            _, pred = torch.max(predictions, 1)
            num_correct = (pred == labels).sum()
            running_acc += num_correct.item()

        epoch_loss = running_loss/train_data.__len__()
        epoch_acc = running_acc/train_data.__len__()

        print('Train {} epoch, Loss: {:.6f}, Acc:{:.6f}'.format(epoch+1, epoch_loss,epoch_acc))
        list_loss.append(running_loss/train_data.__len__())
        list_acc.append(running_acc/train_data.__len__())

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    t2 = time.time()
    total_time = t2 - t1
    print(f'TOTAL-TIME: {total_time // 60:.0f}m{total_time % 60:.0f}s')

    # 绘图查看loss 和 accuracy曲线
    loss_curve(list_loss, list_acc)

    # 加载最佳的模型权重
    model.load_state_dict(best_model_wts)
    return model


model_train = train_model()
print("----------------训练完毕----------------------")

# 保存模型
torch.save(model_train.state_dict(), 'model11_50.pt')
'''
Train 1 epoch, Loss: 0.881416, Acc:0.702030
Train 2 epoch, Loss: 0.696473, Acc:0.950920
Train 3 epoch, Loss: 0.572125, Acc:0.953500
Train 4 epoch, Loss: 0.481754, Acc:0.954980
Train 5 epoch, Loss: 0.413092, Acc:0.956550
Train 6 epoch, Loss: 0.339566, Acc:0.958600
Train 7 epoch, Loss: 0.275737, Acc:0.958700
Train 8 epoch, Loss: 0.227889, Acc:0.960270
Train 9 epoch, Loss: 0.191755, Acc:0.960520
Train 10 epoch, Loss: 0.158716, Acc:0.970020
Train 11 epoch, Loss: 0.131181, Acc:0.989430
Train 12 epoch, Loss: 0.107974, Acc:0.991120
Train 13 epoch, Loss: 0.091479, Acc:0.991710
Train 14 epoch, Loss: 0.076984, Acc:0.992280
Train 15 epoch, Loss: 0.064283, Acc:0.993580
Train 16 epoch, Loss: 0.056268, Acc:0.993460
Train 17 epoch, Loss: 0.048392, Acc:0.994040
Train 18 epoch, Loss: 0.041536, Acc:0.994840
Train 19 epoch, Loss: 0.037166, Acc:0.994790
Train 20 epoch, Loss: 0.032649, Acc:0.995410
Train 21 epoch, Loss: 0.027969, Acc:0.995930
Train 22 epoch, Loss: 0.024655, Acc:0.996110
Train 23 epoch, Loss: 0.023095, Acc:0.996060
Train 24 epoch, Loss: 0.020281, Acc:0.996550
Train 25 epoch, Loss: 0.018101, Acc:0.997000
Train 26 epoch, Loss: 0.016819, Acc:0.997070
Train 27 epoch, Loss: 0.015383, Acc:0.997230
Train 28 epoch, Loss: 0.013675, Acc:0.997770
Train 29 epoch, Loss: 0.013687, Acc:0.997620
Train 30 epoch, Loss: 0.012321, Acc:0.997780
Train 31 epoch, Loss: 0.011586, Acc:0.998070
Train 32 epoch, Loss: 0.010931, Acc:0.997900
Train 33 epoch, Loss: 0.010777, Acc:0.998000
Train 34 epoch, Loss: 0.009579, Acc:0.998450
Train 35 epoch, Loss: 0.009171, Acc:0.998410
Train 36 epoch, Loss: 0.009781, Acc:0.998000
Train 37 epoch, Loss: 0.008348, Acc:0.998360
Train 38 epoch, Loss: 0.009214, Acc:0.998170
Train 39 epoch, Loss: 0.007177, Acc:0.998820
Train 40 epoch, Loss: 0.007617, Acc:0.998460
Train 41 epoch, Loss: 0.006965, Acc:0.998690
Train 42 epoch, Loss: 0.006568, Acc:0.998830
Train 43 epoch, Loss: 0.006500, Acc:0.998790
Train 44 epoch, Loss: 0.005753, Acc:0.998990
Train 45 epoch, Loss: 0.006385, Acc:0.998740
Train 46 epoch, Loss: 0.005432, Acc:0.998950
Train 47 epoch, Loss: 0.005347, Acc:0.999020
Train 48 epoch, Loss: 0.006075, Acc:0.998770
Train 49 epoch, Loss: 0.006002, Acc:0.998780
Train 50 epoch, Loss: 0.005278, Acc:0.999010
TOTAL-TIME: 119m58s
'''


