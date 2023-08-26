from PIL import Image
import dataset
import Net
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from pytorch_msssim import MSSSIM
import numpy as np

# 导入数据集
IrLoader = dataset.MyDataLoader("G:/大三下科目/智慧城市/IVIF/test_data/ir")
Ir_data = IrLoader.load_images_from_folder()
ViLoader = dataset.MyDataLoader("G:/大三下科目/智慧城市/IVIF/test_data/vi")
Vi_data = ViLoader.load_images_from_folder()
Ir_set = DataLoader(Ir_data, batch_size=1, shuffle=False, drop_last=False)
Vi_set = DataLoader(Vi_data, batch_size=1, shuffle=False, drop_last=False)
# 加载模型
model = Net.Autoencoder()
model = torch.load("G:/大三下科目/智慧城市/IVIF/pth/EncDec.pth")
# 转到gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
i = 1
for ir, vi in zip(Ir_set, Vi_set):
    # 转到gpu
    ir = ir.to(device)
    vi = vi.to(device)
    img = model.final_net(ir, vi)
    # print(img.size())
    # 转到cpu
    img = img.to("cpu")
    img = img.squeeze(dim=0)
    # print(img.size())
    img = img.detach().numpy().astype(np.uint8)
    img = img[0]
    img1 = Image.fromarray(img)
    output_path = "G:/大三下科目/智慧城市/IVIF/result/" + str(i) + ".png"
    img1.save(output_path)
    i = i+1
