import torch.utils.data as Data
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pandas import DataFrame
import torch.nn.functional as F


# 测试
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
        self.fc = nn.Linear(4, 3)

    def forward(self, inputs):
        inputs = inputs.view(inputs.size()[0], 1, inputs.size()[1])
        inputs = self.conv_unit(inputs)
        inputs = inputs.view(inputs.size()[0], -1)
        inputs = self.dense_unit(inputs)
        inputs = self.fc(inputs)
        return inputs


# 加载原始数据
class MyData(Data.Dataset):
    def __init__(self, feature):
        self.feature = feature  # 特征

    def __len__(self):
        return len(self.feature)

    def __getitem__(self, idx):
        return self.feature[idx]


def load_data(batch_size):
    # 加载原始数据
    df_test = pd.read_csv('./datasets/testB.csv')
    # 拆解heartbeat_signals
    test_signals = np.array(df_test['heartbeat_signals'].apply(lambda x: np.array(list(map(float, x.split(','))), dtype=np.float32)))
    # 构建pytorch数据类
    test_data = MyData(test_signals)
    # 构建pytorch数据集Dataloader
    test_loader = Data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
    return test_data, test_loader


batch_size = 200
device = torch.device('cpu')
test_data, test_loader = load_data(batch_size)
model = model_CNN_1()

print("----------------开始验证----------------------")
model.load_state_dict(torch.load('./model_ft_50.pt', map_location=device))
print("load success!")

model.eval()

result = []
for data in test_loader:
    data = data.to(device)
    with torch.no_grad():
        out = model(data)
    pre = F.softmax(out, 1)
    pre = pre.to("cpu")
    result.append(pre)
result = torch.stack(result, 0)  # 按照轴0将list转换为tensor
# 进行数据的后处理，准备提交数据(设置阈值)
result = result.numpy()
result = result.reshape((25, 3))
# 这个阈值太大了， result都是0.14 或者0.47
print(result)
thr = [0.2, 0.51, 0.386]
# thr = [0.2, 0.5055, 0.386]
for x in result:
    # print('x=', x)  # x= [0.1748777  0.1748777  0.1748777  0.47536686]
    for i in [1, 2, 0]:
        # print('1x=', x)  # 1x= [0.1748777  0.1748777  0.1748777  0.47536686];
        # print('i=', i)  # i=1; i= 2
        # print('x[i]=', x[i])  # x[1]= 0.1748777;
        # print('thr[i]=', thr[i])  # thr[1]=0.45;thr[2]=0.3
        if x[i] > thr[i]:
            x[0:i] = 0
            # print('x[0:i]=', x[0:i])  # x[0:1]:  [0.]
            x[i + 1: 4] = 0
            # print('x[i + 1: 4]=', x[i + 1: 4])  # x[1 + 1: 4]:  [0. 0.]
            x[i] = 1    # 更新了x
            # print('x[i]=', x[i])  # x[i]:  1.0

print('result11: ', result)  # [[1,2,3,0],[1,2,1,2],[3,0,1,2]]
id = np.arange(0, 25)
df = DataFrame(result, columns=["label_0", "label_1", "label_2"])
df.insert(loc=0, column="id", value=id, allow_duplicates=False)
df.to_csv('./submit1.csv', index_label="id", index=False)
print(df.head())
print("----------------验证完毕----------------------")
