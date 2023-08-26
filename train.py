import matplotlib.pyplot as plt

import dataset
import Net
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from pytorch_msssim import MSSSIM

TrainLoader = dataset.MyDataLoader("G:/大三下科目/智慧城市/IVIF/coco/train")
train_data = TrainLoader.load_images_from_folder()
EvalLoader = dataset.MyDataLoader("G:/大三下科目/智慧城市/IVIF/coco/eval")
eval_data = EvalLoader.load_images_from_folder()
train_set = DataLoader(train_data, batch_size=4, shuffle=True, drop_last=True)
eval_set = DataLoader(eval_data, batch_size=4, shuffle=True, drop_last=True)

# 实例化网络模型
module = Net.Autoencoder()  # 自定义的网络模型类

# 实例化损失函数
# ssim_loss = ssim()
# mse_loss = nn.MSELoss()

# 实例化优化器
optimizer = optim.Adam(module.parameters(), lr=0.0001)
# 转到gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
module.to(device)
#  保存每次的loss
train_loss_values = []
eval_loss_values = []


def train():
    # 开启计算梯度
    module.train()
    ssim = MSSSIM()
    mse = torch.nn.MSELoss()
    best_train_loss = float('inf')
    best_eval_loss = float('inf')
    for epoch in range(10):
        total_train_loss = 0.0
        total_eval_loss = 0.0

        for data in train_set:
            data = data.to(device)
            # 梯度置零
            optimizer.zero_grad()

            # 前向传播
            outputs = module(data)

            # 计算损失

            loss = 100 * ssim(outputs, data) + mse(outputs, data)

            # 反向传播
            loss.backward()

            # 优化器更新参数
            optimizer.step()

            total_train_loss += loss.item()

        # 每轮次结束后计算平均损失
        avg_train_loss = total_train_loss / len(train_set)
        train_loss_values.append(avg_train_loss)  # 保存loss
        print("epoch: ")
        print(epoch)
        print("train loss:")
        print(avg_train_loss)
        print("\n")
        # 模型验证
        module.eval()

        with torch.no_grad():
            for data in eval_set:
                data = data.to(device)
                # 前向传播
                outputs = module(data)

                # 计算损失
                loss = 100 * ssim(outputs, data) + mse(outputs, data)

                total_eval_loss += loss.item()

        avg_eval_loss = total_eval_loss / len(eval_set)
        eval_loss_values.append(avg_eval_loss)  # 保存loss
        print("epoch: ")
        print(epoch)
        print("val loss:")
        print(avg_eval_loss)
        print("\n")
        # 保存模型
        if (avg_eval_loss < best_eval_loss) and (avg_train_loss < best_train_loss):
            best_train_loss = avg_train_loss
            best_eval_loss = avg_eval_loss
            torch.save(module, './pth/EncDec.pth')


# def validate():
#     # 关闭计算梯度
#
#     print("epoch:")
#     print(epoch)
#     # 根据验证损失选择合适的模型


# 调用训练函数
train()
# 画图
plt.plot(train_loss_values, label='train loss')
plt.plot(eval_loss_values, label='eval loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
