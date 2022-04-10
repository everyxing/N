import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import torch
import time
import copy

"""
数据分析：
预测传感器的类别
数据格式：csv
总共有3类（0，1，2）
每组数据包含1440个的值
"""


# 加载数据
class MyData(Data.Dataset):
    def __init__(self, feature, label):
        self.feature = feature # 特征
        self.label = label # 标签

    def __len__(self):
        return len(self.feature)

    def __getitem__(self, idx):
        return self.feature[idx], self.label[idx]


# 加载训练数据集
def load_data(batch_size):
    df_train = pd.read_csv('../datasets/roudoukou_data_LSTM_CNN_train .csv')
    # df_train = pd.read_csv('./datasets/trainB.csv')
    # 拆解heartbeat_signals
    # test_signals就是一行数据放到一个数组里，这个数组里有205个数字。总共有2000行数据，就是2000个数组。test_signals.shape:  (20000,)
    train_signals = np.array(df_train['heartbeat_signals'].apply(lambda x: np.array(list(map(float, x.split(','))), dtype=np.float32)))
    train_labels = np.array(df_train['label'].apply(lambda x: float(x)), dtype=np.float32)
    # 构建pytorch数据类
    train_data = MyData(train_signals, train_labels)
    # 构建pytorch数据集Dataloader
    train_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    return train_data, train_loader


# 加载测试数据集
def load_test_data(batch_size):
    # 加载原始数据
    df_train = pd.read_csv('../datasets/roudoukou_data_LSTM_CNN_test.csv')
    # df_train = pd.read_csv('./datasets/trainB.csv')
    # 拆解heartbeat_signals
    # test_signals就是一行数据放到一个数组里，这个数组里有205个数字。总共有2000行数据，就是2000个数组。test_signals.shape:  (20000,)
    train_signals = np.array(df_train['heartbeat_signals'].apply(lambda x: np.array(list(map(float, x.split(','))), dtype=np.float32)))
    train_labels = np.array(df_train['label'].apply(lambda x: float(x)), dtype=np.float32)
    # 构建pytorch数据类
    train_data = MyData(train_signals, train_labels)
    # 构建pytorch数据集Dataloader
    train_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    return train_data, train_loader


# 画损失函数和准确率的图
def draw_fig(list, name, epoch):
    # 这里迭代了200次，所以x的取值范围为(0，200)，然后再将每次相对应的准确率以及损失率附在x上
    x1 = range(1, epoch+1)
    y1 = list
    if name=="loss":
        plt.cla()
        plt.title('CNN_bs{}_epoch{}_optimizer{} \n Train loss vs. epoch'.format(batch_size, epoch, opt), fontsize=20)
        plt.plot(x1, y1, '.-')
        plt.xlabel('epoch', fontsize=20)
        plt.ylabel('Train loss', fontsize=20)
        plt.grid()
        plt.savefig("./Train_loss.png")
        plt.show()
    elif name == "acc":
        plt.cla()
        plt.title('CNN_bs{}_epoch{}_optimizer{} \n Train accuracy vs. epoch'.format(batch_size, epoch, opt), fontsize=20)
        plt.plot(x1, y1, '.-')
        plt.xlabel('epoch', fontsize=20)
        plt.ylabel('Train accuracy', fontsize=20)
        plt.grid()
        plt.savefig("./Train _accuracy.png")
        plt.show()


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
            nn.Linear(23040, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 3),
            nn.Softmax(dim=1)
        )

    def forward(self, inputs):
        inputs = inputs.view(inputs.size()[0], 1, inputs.size()[1])
        inputs = self.conv_unit(inputs)
        inputs = inputs.view(inputs.size()[0], -1)
        inputs = self.dense_unit(inputs)
        return inputs


# 定义模型、loss function
model = model_CNN_1()
criterion = nn.CrossEntropyLoss()
num_epochs = 350
batch_size = 32
opt = 'SGD'     # 优化器可以选择SGD和Adam
# optimizer = optim.Adam(model.parameters(), lr=0.0001)
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001, momentum=0.9, dampening=0.5, weight_decay=0.01, nesterov=False)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2)


# 载入训练数据
train_data, train_loader = load_data(batch_size)


# 训练
def train_model():
    print("----------------训练开始----------------------")
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
            # print(data)
            inputs, labels = data
            # print('x.shape()', inputs.shape[0], inputs.shape[1])        # x.shape() 64 205
            # print('y.shape()', labels.shape[0])     # y.shape() 64
            predictions = model(inputs)
            loss = criterion(predictions, labels.long())

            # 反向传播三部曲
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
    draw_fig(list_loss, "loss", num_epochs)
    draw_fig(list_acc, "acc", num_epochs)

    # 加载最佳的模型权重
    model.load_state_dict(best_model_wts)
    return model


model_train = train_model()
print("----------------训练完毕----------------------")

# 保存模型权重
torch.save(model_train.state_dict(), 'model_CNN_bs{}_epoch{}_optimizer{}.pt'.format(batch_size, num_epochs, opt))


print("----------------开始验证----------------------")

# 载入测试数据
test_data, test_loader = load_test_data(batch_size)


device = torch.device('cpu')
model.load_state_dict(torch.load('model_CNN_bs{}_epoch{}_optimizer{}.pt'.format(batch_size, num_epochs, opt), map_location=device))
print("load success!")
model.eval()

result = []
running_acc = 0.0
for i, data in enumerate(test_loader):
    inputs, labels = data
    with torch.no_grad():
        predictions = model(inputs)
    _, pred = torch.max(predictions, 1)
    num_correct = (pred == labels).sum()
    running_acc += num_correct.item()
    print('预测值', pred)
    print('真实值', labels)
    epoch_acc = running_acc / test_data.__len__()

    print('Test Acc:{: .3f}'.format(epoch_acc))     # 'model_train_plyimage_sensordata.pt'  Test Acc: 0.952
print("----------------验证完毕----------------------")

