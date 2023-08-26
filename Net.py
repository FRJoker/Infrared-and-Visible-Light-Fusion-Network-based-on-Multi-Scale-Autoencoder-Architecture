import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import MyDataLoader


# 训练用网络定义


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, out_channels, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x


class Autoencoder(nn.Module):  # 训练ECB和DCB用
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        # Encoder
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.ecb10 = Encoder(16, 64)
        self.ecb20 = Encoder(64, 112)
        self.ecb30 = Encoder(112, 160)
        self.ecb40 = Encoder(160, 208)
        # Decoder
        self.dcb31 = Decoder(368, 160)
        self.dcb21 = Decoder(272, 112)
        self.dcb22 = Decoder(384, 112)
        self.dcb11 = Decoder(176, 64)
        self.dcb12 = Decoder(240, 64)
        self.dcb13 = Decoder(304, 64)
        self.conv2 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # x = self.pool(x)
        x = self.relu(self.conv1(x))
        # print("x")
        # print(x.shape)
        # xp = self.pool(x)
        x1 = self.ecb10(x)
        # print("x1")
        # print(x1.shape)
        # xp1 = self.pool(x1)
        x2 = self.ecb20(x1)
        # print("x2")
        # print(x2.shape)
        # xp2 = self.pool(x2)
        x3 = self.ecb30(x2)
        # print("x3")
        # print(x3.shape)
        # xp3 = self.pool(x3)
        x4 = self.ecb40(x3)
        # print("x4")
        # print(x4.shape)
        x5 = self.dcb11(torch.cat((x1, self.upsample(x2)), 1))
        # print("x5")
        # print(x5.shape)
        x6 = self.dcb21(torch.cat((x2, self.upsample(x3)), 1))
        # print("x6")
        # print(x6.shape)
        x7 = self.dcb31(torch.cat((x3, self.upsample(x4)), 1))
        # print("x7")
        # print(x7.shape)
        x8 = self.dcb12(torch.cat((x1, x5, self.upsample(x6)), 1))  # 240,64
        # print("x8")
        # print(x8.shape)
        x9 = self.dcb22(torch.cat((x2, x6, self.upsample(x7)), 1))   # 384,112
        # print("x9")
        # print(x9.shape)
        x10 = self.dcb13(torch.cat((x1, x5, x8, self.upsample(x9)), 1)) # 304,64
        x10 = self.conv2(self.upsample(x10))
        x10 = self.relu(x10)
        # x10 = self.upsample(x10)
        return x10

    def final_net(self, img1, img2):
        img1 = self.relu(self.conv1(img1))
        img2 = self.relu(self.conv1(img2))
        i11 = self.ecb10(img1)
        i21 = self.ecb10(img2)
        # ecb10融合求x1
        # 求l1范数
        x1_1 = torch.abs(i11)
        x1_2 = torch.abs(i21)
        x1_1_norm = torch.sum(x1_1, dim=1)
        x1_2_norm = torch.sum(x1_2, dim=1)
        C1 = x1_1.shape[1]
        # 1/C*l1范数
        x1_1_norm = (1 / C1) * x1_1_norm
        x1_2_norm = (1 / C1) * x1_2_norm
        # softmax
        x1_1_norm_exp = torch.exp(x1_1_norm)
        x1_2_norm_exp = torch.exp(x1_2_norm)
        z1 = x1_1_norm_exp + x1_2_norm_exp
        # beta_1 beta_2
        beta1_ecb10 = x1_1_norm_exp / z1
        beta2_ecb10 = x1_2_norm_exp / z1
        # 求出两个fai_hat
        x1_1_hat = beta1_ecb10 * x1_1
        x1_2_hat = beta2_ecb10 * x1_2
        # 输出
        x1 = x1_1_hat + x1_2_hat
        # ecb20融合求x2
        i12 = self.ecb20(i11)
        i22 = self.ecb20(i21)

        # 求l1范数
        x2_1 = torch.abs(i12)
        x2_2 = torch.abs(i22)
        x2_1_norm = torch.sum(x2_1, dim=1)
        x2_2_norm = torch.sum(x2_2, dim=1)
        C2 = x2_1.shape[1]
        x2_1_norm = (1 / C2) * x2_1_norm
        x2_2_norm = (1 / C2) * x2_2_norm
        # 求softmax
        x2_1_norm_exp = torch.exp(x2_1_norm)
        x2_2_norm_exp = torch.exp(x2_2_norm)
        z2 = x2_1_norm_exp + x2_2_norm_exp
        # 求出beta
        beta1_ecb20 = x2_1_norm_exp / z2
        beta2_ecb20 = x2_2_norm_exp / z2
        # 求出两个fai_hat
        x2_1_hat = beta1_ecb20 * x2_1
        x2_2_hat = beta2_ecb20 * x2_2
        # 输出
        x2 = x2_1_hat + x2_2_hat
        # ecb30融合求x3
        i13 = self.ecb30(i12)
        i23 = self.ecb30(i22)
        # 求l1范数
        x3_1 = torch.abs(i13)
        x3_2 = torch.abs(i23)
        x3_1_norm = torch.sum(x3_1, dim=1)
        x3_2_norm = torch.sum(x3_2, dim=1)
        C3 = x3_1.shape[1]
        x3_1_norm = (1 / C3) * x3_1_norm
        x3_2_norm = (1 / C3) * x3_2_norm
        # 求softmax
        x3_1_norm_exp = torch.exp(x3_1_norm)
        x3_2_norm_exp = torch.exp(x3_2_norm)
        z3 = x3_1_norm_exp + x3_2_norm_exp
        # 求出beta
        beta1_ecb30 = x3_1_norm_exp / z3
        beta2_ecb30 = x3_2_norm_exp / z3
        # 求出两个fai_hat
        x3_1_hat = beta1_ecb30 * x3_1
        x3_2_hat = beta2_ecb30 * x3_2
        # 输出
        x3 = x3_1_hat + x3_2_hat
        # ecb40融合求x4
        i14 = self.ecb40(i13)
        i24 = self.ecb40(i23)
        # 求l1范数
        x4_1 = torch.abs(i14)
        x4_2 = torch.abs(i24)
        x4_1_norm = torch.sum(x4_1, dim=1)
        x4_2_norm = torch.sum(x4_2, dim=1)
        C4 = x4_1.shape[1]
        x4_1_norm = (1 / C4) * x4_1_norm
        x4_2_norm = (1 / C4) * x4_2_norm
        # 求softmax
        x4_1_norm_exp = torch.exp(x4_1_norm)
        x4_2_norm_exp = torch.exp(x4_2_norm)
        z = x4_1_norm_exp + x4_2_norm_exp
        # 求出beta
        beta1_ecb40 = x4_1_norm_exp / z
        beta2_ecb40 = x4_2_norm_exp / z
        # 求出两个fai_hat
        x4_1_hat = beta1_ecb40 * x4_1
        x4_2_hat = beta2_ecb40 * x4_2
        # 输出
        x4 = x4_1_hat + x4_2_hat

        # DCB
        x5 = self.dcb11(torch.cat((x1, self.upsample(x2)), 1))
        x6 = self.dcb21(torch.cat((x2, self.upsample(x3)), 1))
        x7 = self.dcb31(torch.cat((x3, self.upsample(x4)), 1))
        x8 = self.dcb12(torch.cat((x1, x5, self.upsample(x6)), 1))  # 240,64
        x9 = self.dcb22(torch.cat((x2, x6, self.upsample(x7)), 1))   # 384,112
        x10 = self.dcb13(torch.cat((x1, x5, x8, self.upsample(x9)), 1)) # 304,64
        x10 = self.conv2(self.upsample(x10))
        x10 = self.relu(x10)
        return x10



# # 创建自动编码器实例
# autoencoder = Autoencoder()
#
# # 打印网络结构
# # print(autoencoder)
#
# TrainLoader = MyDataLoader("G:/大三下科目/智慧城市/IVIF/coco/train")
# train_data = TrainLoader.load_images_from_folder()
# # train_set = DataLoader(train_data, batch_size=1, shuffle=True, drop_last=True)
# # inputs, _ = train_set
# inputs = train_data[1]
# inputs = torch.unsqueeze(inputs, dim=0)
# print(autoencoder(inputs).size())
